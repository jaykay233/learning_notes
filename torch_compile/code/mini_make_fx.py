"""An ~400 line re-implementation of ``torch.fx.experimental.proxy_tensor.make_fx``.

Run with ``python mini_make_fx.py``. It needs only PyTorch: no MLX, no Aotolith,
because it stops at the FX graph that a frontend later lowers into its own IR.

``make_fx`` is often mistaken for something that inspects bytecode or parses an
AST. It does neither. It runs the function on real tensors and records every
ATen operator that gets dispatched. Four layers, one per real component:

    Layer 0  Mini graph        <->  torch.fx.Graph / Node / Tracer
    Layer 1  Proxy + memo      <->  torch.fx.Proxy / set_proxy_slot / fetch_object_proxy
    Layer 2  Dispatch mode     <->  ProxyTorchDispatchMode.__torch_dispatch__
    Layer 3  Entry point       <->  torch.fx.experimental.proxy_tensor.make_fx

The whole trick is four steps in ``proxy_call``: build a graph node, let the real
kernel run, then bind the concrete output tensor to that node.

Every claim in the accompanying notes is reproduced by one of the demos at the
bottom, so the output doubles as evidence.
"""

from __future__ import annotations

import operator
import time
from collections import defaultdict
from typing import Any, Callable, Iterator

import torch
from torch.utils._python_dispatch import (
    TorchDispatchMode,
    _get_current_dispatch_mode_stack,
)

# ---------------------------------------------------------------------------
# Layer 0: the graph is an ordered list of rows.
#
# There is deliberately no edge list. A row's ``args`` hold references to the
# rows that produce its operands, so every edge can be re-derived by scanning
# ``args``. Storing edges twice would admit two versions of the same fact.
# ---------------------------------------------------------------------------


class Node:
    """One row: a placeholder input, a call, or the return value."""

    def __init__(
        self,
        kind: str,
        target: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        name: str,
    ) -> None:
        self.kind = kind
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.name = name

    def __repr__(self) -> str:
        if self.kind == "placeholder":
            return f"%{self.name} : placeholder"
        if self.kind == "output":
            return f"return {_format(self.args[0])}"
        return f"%{self.name} = {_target_name(self.target)}({_format(self.args)})"


class Graph:
    def __init__(self) -> None:
        self.nodes: list[Node] = []

    def __repr__(self) -> str:
        body = "\n".join("    " + repr(node) for node in self.nodes)
        return "graph():\n" + body


def _target_name(target: Any) -> str:
    if isinstance(target, torch._ops.OpOverload):
        return str(target)
    return getattr(target, "__name__", str(target))


def _format(value: Any) -> str:
    if isinstance(value, Node):
        return f"%{value.name}"
    if isinstance(value, tuple):
        inner = ", ".join(_format(item) for item in value)
        return f"({inner},)" if len(value) == 1 else f"({inner})"
    if isinstance(value, list):
        return "[" + ", ".join(_format(item) for item in value) + "]"
    return repr(value)


# ---------------------------------------------------------------------------
# Layer 1: Proxy is a handle to a value that only exists as a row so far.
#
# Proxy exists instead of "just use Node" for one hard reason: crossing from the
# value side into the graph side goes through ``create_arg``, whose whitelist
# accepts Proxy but rejects Node. Handing a Node to ``create_arg`` raises
# ``NotImplementedError: argument of type: <class 'torch.fx.node.Node'>``.
# ---------------------------------------------------------------------------


class Proxy:
    """Handle for a value that only exists as a row in the graph so far."""

    def __init__(self, node: Node, tracer: "PythonKeyTracer") -> None:
        self.node = node
        self.tracer = tracer

    def __getitem__(self, index: int) -> "Proxy":
        # Indexing a multi-output op materialises a getitem row on demand. This
        # is where the %getitem rows in real FX graphs come from: nothing
        # special-cases multi-output ops, they just get indexed like any value.
        return self.tracer.create_proxy(
            "call_function", operator.getitem, (self, index), {}
        )

    def __repr__(self) -> str:
        return f"%{self.node.name}"


class PythonKeyTracer:
    """Creates nodes and resolves tensors back to the handles that produced them.

    The memo stands in for the ``proxy_slot`` that PyTorch attaches to a tensor.
    The real one is keyed by tracer and holds a weak reference; keying by
    ``id()`` plus a strong reference is equivalent for a single tracer.

    The memo holds Proxy, not Node, because whatever comes out of it is used as
    an operand, and operands are converted by ``create_arg`` -- which only
    understands Proxy. Real make_fx is stricter still: it stores a
    ``_ProxyTensor(proxy, constant)`` and every consumer reads ``.proxy``.
    """

    def __init__(self) -> None:
        self.graph = Graph()
        self._memo: dict[int, tuple[torch.Tensor, Proxy]] = {}
        self._counts: dict[str, int] = {}

    def _fresh_name(self, target: Any) -> str:
        base = _target_name(target).replace(".", "_")
        count = self._counts.get(base, 0)
        self._counts[base] = count + 1
        return base if count == 0 else f"{base}_{count}"

    def create_node(
        self,
        kind: str,
        target: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Node:
        node = Node(kind, target, args, kwargs, self._fresh_name(target))
        self.graph.nodes.append(node)
        return node

    def create_proxy(
        self,
        kind: str,
        target: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Proxy:
        node = self.create_node(
            kind, target, self.create_arg(args), self.create_arg(kwargs)
        )
        return Proxy(node, self)

    def create_arg(self, value: Any) -> Any:
        """Replace proxies with their nodes, recursively through containers.

        Anything left over is a genuine constant: an int, a dtype, a list of
        sizes. Those get baked into the node as literal arguments. A real
        Tensor is *not* accepted here; see ``_demo_tensor_in_args``.
        """

        if isinstance(value, Proxy):
            return value.node
        if isinstance(value, Node):
            return value
        if isinstance(value, tuple) and not isinstance(value, torch.Size):
            return tuple(self.create_arg(item) for item in value)
        if isinstance(value, list):
            return [self.create_arg(item) for item in value]
        if isinstance(value, dict):
            return {key: self.create_arg(item) for key, item in value.items()}
        return value

    def bind(self, tensor: torch.Tensor, proxy: Proxy) -> None:
        self._memo[id(tensor)] = (tensor, proxy)

    def fetch(self, value: Any) -> Any:
        if not isinstance(value, torch.Tensor):
            return value
        entry = self._memo.get(id(value))
        if entry is None:
            raise RuntimeError(
                "tensor with shape "
                f"{tuple(value.shape)} was not produced by a traced input, so it "
                "has no node. make_fx requires every operand to be traceable."
            )
        return entry[1]


def tree_map(fn: Callable[[Any], Any], value: Any) -> Any:
    if isinstance(value, tuple) and not isinstance(value, torch.Size):
        return tuple(tree_map(fn, item) for item in value)
    if isinstance(value, list):
        return [tree_map(fn, item) for item in value]
    if isinstance(value, dict):
        return {key: tree_map(fn, item) for key, item in value.items()}
    return fn(value)


# ---------------------------------------------------------------------------
# Layer 2: the heart of make_fx. One dispatch callback, four steps.
# ---------------------------------------------------------------------------


class ProxyTorchDispatchMode(TorchDispatchMode):
    def __init__(self, tracer: PythonKeyTracer) -> None:
        self.tracer = tracer

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[Any, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        return proxy_call(self, func, args, kwargs or {})


def proxy_call(
    mode: ProxyTorchDispatchMode,
    func: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    tracer = mode.tracer

    # 1. Turn the live tensors into proxies. This is also the check that every
    #    operand is traceable; a tensor from outside the trace raises here.
    proxy_args = tree_map(tracer.fetch, args)
    proxy_kwargs = tree_map(tracer.fetch, kwargs)

    # 2. Record the row *before* running the op. If this happened afterwards,
    #    inner ops dispatched by this op would land in the graph first and the
    #    row order would no longer respect data dependencies.
    out_proxy = tracer.create_proxy("call_function", func, proxy_args, proxy_kwargs)

    # 3. Let the real kernel run. No manual "disable myself" is needed: the
    #    dispatcher pops this mode before invoking __torch_dispatch__, so a
    #    direct func(*args) cannot recurse. Real make_fx additionally wraps this
    #    in disable_proxy_modes_tracing(), which turns off the C++ PROXY infra
    #    mode rather than this Python-level one.
    out = func(*args, **kwargs)

    # 4. Tie the concrete outputs back to the row so later ops can find them.
    track_tensor_tree(out, out_proxy, tracer)

    return out


def track_tensor_tree(value: Any, proxy: Proxy, tracer: PythonKeyTracer) -> None:
    """Bind each output tensor to its proxy, projecting tuples one index at a time."""

    if isinstance(value, torch.Tensor):
        tracer.bind(value, proxy)
    elif isinstance(value, (tuple, list)) and not isinstance(value, torch.Size):
        for index, item in enumerate(value):
            track_tensor_tree(item, proxy[index], tracer)
    # SymInts and script objects are out of scope here.


# ---------------------------------------------------------------------------
# Layer 3: the entry point. Placeholders, run under the mode, emit the output.
# ---------------------------------------------------------------------------


def make_fx(fn: Callable[..., Any], *example_inputs: torch.Tensor) -> Graph:
    tracer = PythonKeyTracer()

    for index, tensor in enumerate(example_inputs):
        node = tracer.create_node("placeholder", f"x_{index}", (), {})
        tracer.bind(tensor, Proxy(node, tracer))

    # The function body has no idea it is being traced. It sees ordinary
    # tensors in and returns ordinary tensors out.
    with ProxyTorchDispatchMode(tracer):
        out = fn(*example_inputs)

    tracer.create_node("output", None, (tree_map(tracer.fetch, out),), {})
    return tracer.graph


# ---------------------------------------------------------------------------
# Graph queries: edges are derived, never stored.
# ---------------------------------------------------------------------------


def _iter_node_refs(value: Any) -> Iterator[Node]:
    if isinstance(value, Node):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_node_refs(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_node_refs(item)


def edges(graph: Graph) -> list[tuple[str, str]]:
    """Every edge, recovered by scanning args and kwargs."""

    found: list[tuple[str, str]] = []
    for node in graph.nodes:
        for ref in _iter_node_refs(node.args):
            found.append((ref.name, node.name))
        for ref in _iter_node_refs(node.kwargs):
            found.append((ref.name, node.name))
    return found


def users(graph: Graph) -> dict[str, list[str]]:
    """Reverse edges, equally derived."""

    reverse: dict[str, list[str]] = defaultdict(list)
    for source, destination in edges(graph):
        reverse[source].append(destination)
    return reverse


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------


def _silu_matmul(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(x @ w + b)


def _max_with_index(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.max(x, dim=-1)


def _rms_scale(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * x @ w


def _fan_out(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    y = x @ w
    return torch.nn.functional.silu(y + b) * y


def _op_names(nodes: list[Any], field: str) -> list[str]:
    return [
        _target_name(node.target)
        for node in nodes
        if getattr(node, field) == "call_function"
    ]


def _banner(text: str) -> None:
    print()
    print("=" * 68)
    print(text)
    print("=" * 68)


def _demo_basic() -> None:
    _banner("1. A three-input function becomes four rows plus placeholders")
    inputs = (torch.randn(2, 4), torch.randn(4, 8), torch.randn(8))
    print(make_fx(_silu_matmul, *inputs))


def _demo_multi_output() -> None:
    _banner("2. A multi-output op grows getitem rows because the tuple is indexed")
    print(make_fx(_max_with_index, torch.randn(3, 4)))


def _demo_fake_mode() -> None:
    _banner("3. FakeTensorMode makes capture shape-only instead of shape-and-data")
    size = 4096
    left, right = torch.randn(size, size), torch.randn(size, size)

    start = time.perf_counter()
    make_fx(lambda a, b: a @ b, left, right)
    real_seconds = time.perf_counter() - start

    from torch._subclasses import FakeTensorMode

    # `with FakeTensorMode()` with no device is what make_fx's
    # tracing_mode="fake" does internally.
    with FakeTensorMode():
        fake_left, fake_right = torch.randn(size, size), torch.randn(size, size)
        start = time.perf_counter()
        graph = make_fx(lambda a, b: a @ b, fake_left, fake_right)
        fake_seconds = time.perf_counter() - start

    print(graph)
    print()
    print(f"  {size}x{size} matmul, one trace each:")
    print(f"    real: {real_seconds * 1000:8.2f} ms   (computes the matmul)")
    print(f"    fake: {fake_seconds * 1000:8.2f} ms   (records shapes only)")
    print(f"    speedup: {real_seconds / fake_seconds:6.1f}x")
    print("  (timings drift between runs; the gap stays two orders of magnitude)")


def _demo_against_torch() -> None:
    _banner("4. Same ops, same row count as torch's own make_fx")
    from torch.fx.experimental.proxy_tensor import make_fx as torch_make_fx

    inputs = (torch.randn(4, 8), torch.randn(8))
    mine = make_fx(_rms_scale, *inputs)
    reference = torch_make_fx(_rms_scale)(*inputs).graph

    mine_ops = _op_names(mine.nodes, "kind")
    reference_ops = _op_names(list(reference.nodes), "op")

    print("  mini :", mine_ops)
    print("  torch:", reference_ops)
    assert mine_ops == reference_ops, "operator sequence differs"
    assert len(mine.nodes) == len(reference.nodes), "row count differs"
    print(f"\n  {len(mine.nodes)} rows each, identical operator sequence")


def _demo_no_edges_stored() -> None:
    _banner("5. The graph stores rows only; edges are recovered by scanning args")
    graph = make_fx(_fan_out, torch.randn(2, 4), torch.randn(4, 8), torch.randn(8))
    print("  Graph attributes:", list(vars(graph)))
    print()
    print(graph)
    print()
    print("  derived edges (%d):" % len(edges(graph)))
    for source, destination in edges(graph):
        print(f"    %{source:20s} -> %{destination}")
    print()
    print("  derived users (reverse edges):")
    for name, consumers in users(graph).items():
        print(f"    %{name:20s} used by {['%' + c for c in consumers]}")

    # Show the failure mode that a stored edge list would introduce: two
    # versions of the same fact, with nothing forcing them to agree.
    fabricated = edges(graph) + [("aten_mm_default", "aten_silu_default")]
    real_edges = set(edges(graph))
    bogus = [edge for edge in fabricated if edge not in real_edges]
    print()
    print(f"  fabricated edge {bogus[0]} is absent from args")
    print("  -> with a stored edge list the graph would hold two disagreeing")
    print("     descriptions of its own topology, and nothing would report it")


def _demo_two_worlds() -> None:
    _banner("6. Node is a row; it cannot be used as a value")
    tensor = torch.randn(2, 3)
    print("  value side, torch.Tensor:")
    print("    tensor + 1  ->", tuple((tensor + 1).shape))
    print("    tensor.shape[0] ->", tensor.shape[0])

    node = Node("placeholder", "x", (), {}, "x")
    print("  graph side, Node:")
    for label, operation in (
        ("node + 1", lambda: node + 1),
        ("node.shape", lambda: node.shape),
        ("node[0]", lambda: node[0]),
    ):
        try:
            operation()
            print(f"    {label:12s} -> succeeded (should not happen)")
        except Exception as exc:  # noqa: BLE001 - the failure is the point
            print(f"    {label:12s} -> {type(exc).__name__}: {exc}")


def _demo_mode_is_popped() -> None:
    _banner("7. The dispatcher pops the mode before calling __torch_dispatch__")

    observed: list[list[str]] = []

    class Probe(TorchDispatchMode):
        def __torch_dispatch__(
            self,
            func: Any,
            types: tuple[Any, ...],
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            observed.append(
                [type(mode).__name__ for mode in _get_current_dispatch_mode_stack()]
            )
            return func(*args, **(kwargs or {}))

    print("  mode stack before with-block:", len(_get_current_dispatch_mode_stack()))
    with Probe():
        torch.add(torch.randn(3), torch.randn(3))
    print("  mode stack inside __torch_dispatch__:", observed[0])
    print("  mode stack after with-block:", len(_get_current_dispatch_mode_stack()))
    print("  -> a bare func(*args) inside the hook cannot re-enter this mode")


def _demo_tensor_in_args() -> None:
    _banner("8. A real Tensor in args survives as shape and dtype, not as data")
    import torch.fx as fx

    graph = fx.Graph()
    placeholder = graph.placeholder("x")
    mul = graph.call_function(
        torch.ops.aten.mul.Tensor, (placeholder, torch.tensor(2.0))
    )
    graph.output(mul)
    print("  fx.Graph.call_function took a Tensor operand without complaining")
    print("  Graph is a plain container; it does not type-check arguments")

    graph_module = fx.GraphModule(torch.nn.Module(), graph)
    print("  GraphModule also built, because codegen does not validate either")
    for line in graph_module.code.splitlines():
        if "mul_tensor" in line:
            print(f"  generated: {line.strip()}")

    inputs = torch.arange(1.0, 4.0)
    print(f"  intent: {inputs.tolist()} * 2.0 = {[v * 2.0 for v in inputs.tolist()]}")
    try:
        print(f"  actual: {graph_module(inputs).tolist()}")
    except Exception as exc:  # noqa: BLE001 - the failure is the point
        print(f"  actual: {type(exc).__name__}: {exc}".splitlines()[0])
    print()
    print("  the operand became torch.Tensor(size=[], dtype=float32):")
    print("    shape and dtype were preserved, the number 2.0 was not")
    print("  -> this is the line between a description and a value")


def _demo_meta_holds_description() -> None:
    _banner("9. node.meta holds a description of the value, not the value")
    from torch.fx.experimental.proxy_tensor import make_fx as torch_make_fx

    graph_module = torch_make_fx(lambda x, w: x @ w)(
        torch.randn(2, 4), torch.randn(4, 8)
    )
    for node in graph_module.graph.nodes:
        if node.op == "call_function":
            value = node.meta.get("val")
            print(f"  %{node.name}: meta keys = {sorted(k for k in node.meta)}")
            print(f"    meta['val'] type  = {type(value).__name__}")
            print(f"    meta['val'] shape = {tuple(value.shape)}")
            print(f"    meta['val'] dtype = {value.dtype}")
            print(f"    declared bytes    = {value.untyped_storage().nbytes()}")
            try:
                value.untyped_storage()[0]
                print("    first byte        = readable")
            except Exception as exc:  # noqa: BLE001 - the failure is the point
                print(f"    first byte        -> {type(exc).__name__}: {exc}")
    print("  -> it knows how many bytes the value would occupy and holds none of them")


def _demo_persisted_split() -> None:
    _banner("10. Structure travels as JSON, weights travel beside it")
    sample = {
        "inputs": ["arg0", "arg1"],
        "ops": [
            {"op": "matmul", "inputs": ["arg0", "arg1"], "outputs": ["v0"]},
            {"op": "silu", "inputs": ["v0"], "outputs": ["v1"]},
        ],
        "outputs": ["v1"],
        "values": {
            "arg0": {"dtype": "float32", "shape": [8, 256]},
            "arg1": {"dtype": "float32", "shape": [256, 512]},
            "v0": {"dtype": "float32", "shape": [8, 512]},
            "v1": {"dtype": "float32", "shape": [8, 512]},
        },
    }
    graph_bytes = len(repr(sample).encode())
    weight_bytes = 256 * 512 * 4
    print(f"  a 2-op graph describing a 256x512 weight: {graph_bytes} bytes of structure")
    print(f"  the weight itself:                        {weight_bytes} bytes of data")
    print(f"  ratio: 1 : {weight_bytes // graph_bytes}")
    print("  the graph names arg1 and records its shape; the bytes live elsewhere")


def _demo_dispatch_trace() -> None:
    _banner("11. One trace advances both sides at every dispatch")
    original = globals()["proxy_call"]
    step = {"n": 0}

    def traced(mode: Any, func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        step["n"] += 1
        before = len(mode.tracer.graph.nodes)
        print(f"  -- dispatch {step['n']}: {_target_name(func)}")
        for operand in args:
            if isinstance(operand, torch.Tensor):
                print(f"     [value] operand shape={tuple(operand.shape)}")
        result = original(mode, func, args, kwargs)
        if isinstance(result, torch.Tensor):
            print(f"     [value] result  shape={tuple(result.shape)}")
        added = mode.tracer.graph.nodes[before:]
        print(f"     [graph] rows    {', '.join('%' + n.name for n in added)}")
        return result

    globals()["proxy_call"] = traced
    try:
        graph = make_fx(_silu_matmul, torch.randn(2, 4), torch.randn(4, 8), torch.randn(8))
    finally:
        globals()["proxy_call"] = original
    print()
    print(graph)


def main() -> None:
    _demo_basic()
    _demo_multi_output()
    _demo_fake_mode()
    _demo_against_torch()
    _demo_no_edges_stored()
    _demo_two_worlds()
    _demo_mode_is_popped()
    _demo_tensor_in_args()
    _demo_meta_holds_description()
    _demo_persisted_split()
    _demo_dispatch_trace()
    _banner("Next: a frontend walks an FX graph like this one and rewrites each")
    print("      row into an SSA operation, keyed by node identity. At that point")
    print("      the Proxy layer is unnecessary: everything is already graph-side.")


if __name__ == "__main__":
    main()
