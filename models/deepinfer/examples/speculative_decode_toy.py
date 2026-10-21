from collections.abc import Sequence


def draw(probs: Sequence[float], u: float) -> int:
    """Draw an index using a uniform u in [0, 1)."""
    if not 0.0 <= u < 1.0:
        raise ValueError("u must be in [0, 1)")
    total = sum(probs)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"probabilities must sum to 1, got {total}")
    cumulative = 0.0
    for token_id, probability in enumerate(probs):
        cumulative += probability
        if u < cumulative:
            return token_id
    raise RuntimeError("rounding error: no token selected")


def speculative_step(
    target_dists: Sequence[Sequence[float]],
    draft_dists: Sequence[Sequence[float]],
    draft_tokens: Sequence[int],
    accept_uniforms: Sequence[float],
    residual_uniforms: Sequence[float],
    bonus_uniform: float,
) -> tuple[list[int], str]:
    """Verify a draft prefix and return emitted token IDs plus a short trace."""
    count = len(draft_tokens)
    if len(target_dists) != count + 1:
        raise ValueError("target_dists needs one extra row for the bonus token")
    if not (len(draft_dists) == len(accept_uniforms) == len(residual_uniforms) == count):
        raise ValueError("draft inputs must have one item per proposed token")

    emitted: list[int] = []
    for step, token_id in enumerate(draft_tokens):
        target = target_dists[step]
        draft = draft_dists[step]
        if len(target) != len(draft) or not 0 <= token_id < len(target):
            raise ValueError("distribution sizes or draft token ID are invalid")
        if abs(sum(target) - 1.0) > 1e-9 or abs(sum(draft) - 1.0) > 1e-9:
            raise ValueError("target and draft distributions must sum to 1")
        draft_probability = draft[token_id]
        if draft_probability <= 0.0:
            raise ValueError("draft token must have nonzero probability under Q")

        accept_probability = min(1.0, target[token_id] / draft_probability)
        if accept_uniforms[step] < accept_probability:
            emitted.append(token_id)
            continue

        residual = [max(p - q, 0.0) for p, q in zip(target, draft)]
        residual_mass = sum(residual)
        if residual_mass <= 0.0:
            raise ValueError("residual is empty; standard rejection sampling cannot correct")
        corrected = [weight / residual_mass for weight in residual]
        emitted.append(draw(corrected, residual_uniforms[step]))
        return emitted, f"rejected draft position {step}"

    emitted.append(draw(target_dists[-1], bonus_uniform))
    return emitted, "all drafts accepted; appended bonus token"


def main() -> None:
    vocab = ["A", "B", "C", "D"]
    target = [
        [0.1, 0.6, 0.2, 0.1],
        [0.1, 0.2, 0.6, 0.1],
        [0.1, 0.2, 0.3, 0.4],
    ]
    draft = [
        [0.1, 0.7, 0.1, 0.1],
        [0.7, 0.1, 0.1, 0.1],
    ]
    rejected_tokens, rejected_trace = speculative_step(
        target, draft, [1, 0], [0.2, 0.8], [0.5, 0.5], 0.85
    )
    print("rejection:", [vocab[token] for token in rejected_tokens], "|", rejected_trace)

    all_accept_target = [[0.1, 0.6, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]]
    all_accept_draft = [[0.1, 0.7, 0.1, 0.1]]
    accepted_tokens, accepted_trace = speculative_step(
        all_accept_target, all_accept_draft, [1], [0.2], [0.5], 0.85
    )
    print("all accepted:", [vocab[token] for token in accepted_tokens], "|", accepted_trace)


if __name__ == "__main__":
    main()
