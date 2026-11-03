#!/usr/bin/env python3
"""Standalone demo of SGLang's Aaronson-Gumbel watermark contract.

This is a reference walkthrough, not the production SGLang kernel. It implements
the same single-key hash contract and selector on the CPU so that the sampling
distribution and detection statistic can be inspected without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, lgamma, log, sqrt
from random import Random
from typing import Sequence

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
KEY = 0x0123456789ABCDEF
PROMPT = (101, 102, 103, 104, 105, 106, 107, 108)
PROBABILITIES = (0.40, 0.25, 0.20, 0.10, 0.05)


def rotl32(value: int, shift: int) -> int:
    value &= MASK32
    return ((value << shift) | (value >> (32 - shift))) & MASK32


def mix(state: int, value: int) -> int:
    """MurmurHash3 x86 32-bit body mix."""
    value = (value * 0xCC9E2D51) & MASK32
    value = rotl32(value, 15)
    value = (value * 0x1B873593) & MASK32
    state = rotl32(state ^ value, 13)
    return (state * 5 + 0xE6546B64) & MASK32


def fmix32(value: int) -> int:
    """MurmurHash3 32-bit finalizer."""
    value &= MASK32
    value ^= value >> 16
    value = (value * 0x85EBCA6B) & MASK32
    value ^= value >> 13
    value = (value * 0xC2B2AE35) & MASK32
    value ^= value >> 16
    return value & MASK32


def context_hash(context: Sequence[int]) -> int:
    state = 0
    for token_id in context:
        state = mix(state, int(token_id))
    return fmix32(state ^ ((4 * len(context)) & MASK32))


def token_u(key: int, ctx_hash: int, token_id: int) -> float:
    key &= MASK64
    key_lo = key & MASK32
    key_hi = (key >> 32) & MASK32
    state = mix(0, key_lo)
    state = mix(state, key_hi)
    state = mix(state, ctx_hash)
    token_hash = fmix32(mix(state, int(token_id)) ^ 16)
    return (token_hash + 0.5) / (1 << 32)


def _validate_probabilities(probabilities: Sequence[float]) -> None:
    if not probabilities:
        raise ValueError("probabilities must not be empty")
    if any(p < 0.0 for p in probabilities):
        raise ValueError("probabilities must be non-negative")
    if abs(sum(probabilities) - 1.0) > 1e-9:
        raise ValueError("probabilities must sum to 1")


def watermark_select(
    probabilities: Sequence[float], context: Sequence[int], key: int = KEY
) -> int:
    """Select a token with argmax(log(u_i) / p_i)."""
    _validate_probabilities(probabilities)
    ctx_hash = context_hash(context)
    best_token = -1
    best_score = float("-inf")
    # Iterating in token-ID order makes the first candidate win an exact tie.
    for token_id, probability in enumerate(probabilities):
        if probability <= 0.0:
            continue
        score = log(token_u(key, ctx_hash, token_id)) / probability
        if best_token < 0 or score > best_score:
            best_token = token_id
            best_score = score
    if best_token < 0:
        raise ValueError("at least one probability must be positive")
    return best_token


def ordinary_sample(probabilities: Sequence[float], rng: Random) -> int:
    _validate_probabilities(probabilities)
    draw = rng.random()
    cumulative = 0.0
    for token_id, probability in enumerate(probabilities):
        cumulative += probability
        if draw < cumulative:
            return token_id
    return len(probabilities) - 1


@dataclass
class WatermarkState:
    """Per-request eligibility and distinct-context history for the demo."""

    key: int
    context_window: int
    max_contexts: int = 4096
    max_probability: float = 1.0
    enabled: bool = True
    seen_hashes: set[int] = field(default_factory=set)
    forced_contexts: int = 0

    def __post_init__(self) -> None:
        if not 1 <= self.context_window <= 64:
            raise ValueError("context_window must be in [1, 64]")
        if not 0.0 < self.max_probability <= 1.0:
            raise ValueError("max_probability must be in (0, 1]")

    def choose(
        self, probabilities: Sequence[float], context: Sequence[int], rng: Random
    ) -> tuple[int, bool]:
        _validate_probabilities(probabilities)
        context = tuple(context[-self.context_window :])
        positive = sum(p > 0.0 for p in probabilities)
        if (
            not self.enabled
            or not context
            or positive <= 1
            or max(probabilities) > self.max_probability
            or self.forced_contexts >= self.max_contexts
        ):
            return ordinary_sample(probabilities, rng), False

        ctx_hash = context_hash(context)
        if ctx_hash in self.seen_hashes:
            return ordinary_sample(probabilities, rng), False

        token_id = watermark_select(probabilities, context, self.key)
        self.seen_hashes.add(ctx_hash)
        self.forced_contexts += 1
        return token_id, True


def _regularized_gamma_q(shape: float, x: float) -> float:
    """Upper regularized incomplete gamma Q(shape, x), stdlib-only."""
    if shape <= 0.0 or x < 0.0:
        raise ValueError("shape must be positive and x non-negative")
    if x == 0.0:
        return 1.0
    log_prefactor = shape * log(x) - x - lgamma(shape)
    epsilon = 3e-14
    tiny = 1e-300

    if x < shape + 1.0:
        term = 1.0 / shape
        total = term
        ap = shape
        for _ in range(10_000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * epsilon:
                break
        lower = total * exp(log_prefactor)
        return max(0.0, min(1.0, 1.0 - lower))

    b = x + 1.0 - shape
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - shape)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break
    upper = exp(log_prefactor) * h
    return max(0.0, min(1.0, upper))


@dataclass(frozen=True)
class DetectionResult:
    num_contexts: int
    score: float
    z_score: float
    p_value: float
    tokens_examined: int


def detect(
    completion: Sequence[int],
    *,
    prompt: Sequence[int] = (),
    key: int = KEY,
    context_window: int = 4,
    max_contexts: int = 4096,
) -> DetectionResult:
    """Score a completion using exact context tuples, as the detector does."""
    if context_window <= 0:
        raise ValueError("context_window must be positive")
    if max_contexts <= 0:
        raise ValueError("max_contexts must be positive")

    prompt = tuple(prompt)
    text = prompt + tuple(completion)
    seen_contexts: set[tuple[int, ...]] = set()
    score = 0.0
    num_contexts = 0
    tokens_examined = 0
    # Without prompt IDs, the first h completion tokens have unknown contexts.
    start_position = len(prompt) if prompt else context_window

    for position in range(start_position, len(text)):
        tokens_examined += 1
        context = tuple(text[max(0, position - context_window) : position])
        if not context or context in seen_contexts:
            continue
        seen_contexts.add(context)
        u = token_u(key, context_hash(context), text[position])
        score += -log(1.0 - u)
        num_contexts += 1
        if num_contexts >= max_contexts:
            break

    if num_contexts == 0:
        raise ValueError("no context was eligible for detection")
    z_score = (score - num_contexts) / sqrt(num_contexts)
    return DetectionResult(
        num_contexts=num_contexts,
        score=score,
        z_score=z_score,
        p_value=_regularized_gamma_q(float(num_contexts), score),
        tokens_examined=tokens_examined,
    )


def generate(
    *,
    watermarked: bool,
    seed: int,
    num_tokens: int = 256,
    context_window: int = 4,
) -> tuple[tuple[int, ...], int]:
    rng = Random(seed)
    text = list(PROMPT)
    state = WatermarkState(
        key=KEY,
        context_window=context_window,
        max_contexts=num_tokens,
        max_probability=1.0,
    )
    forced = 0
    for _ in range(num_tokens):
        context = tuple(text[-context_window:])
        if watermarked:
            token_id, was_forced = state.choose(PROBABILITIES, context, rng)
            forced += int(was_forced)
        else:
            token_id = ordinary_sample(PROBABILITIES, rng)
        text.append(token_id)
    return tuple(text[len(PROMPT) :]), forced


def main() -> None:
    example_context = (11, 12, 13, 14)
    example_probabilities = (0.7, 0.2, 0.1)
    ctx_hash = context_hash(example_context)
    us = [token_u(KEY, ctx_hash, token_id) for token_id in range(3)]
    scores = [log(u) / p for u, p in zip(us, example_probabilities)]
    selected = watermark_select(example_probabilities, example_context)

    print("1) One watermarked selection")
    print(f"context_hash = 0x{ctx_hash:08x}")
    for token_id in range(3):
        print(
            f"token={token_id} p={example_probabilities[token_id]:.1f} "
            f"u={us[token_id]:.6f} score={scores[token_id]:.6f}"
        )
    print(f"selected token = {selected}\n")

    distribution_draws = 50_000
    counts = [0] * len(PROBABILITIES)
    for i in range(distribution_draws):
        context = (9000 + i, 31, 32, 33)
        counts[watermark_select(PROBABILITIES, context)] += 1
    print(f"2) Marginal distribution over {distribution_draws:,} contexts")
    for token_id, expected in enumerate(PROBABILITIES):
        observed = counts[token_id] / distribution_draws
        print(
            f"token={token_id} expected={expected:.3f} "
            f"observed={observed:.3f} delta={observed - expected:+.3f}"
        )
    print()

    trials = 100
    print(f"3) Detector over {trials} independent sequences")
    for label, watermarked in (("ordinary", False), ("watermarked", True)):
        p_values = []
        forced_counts = []
        for seed in range(trials):
            completion, forced = generate(watermarked=watermarked, seed=seed)
            result = detect(completion, prompt=PROMPT, key=KEY, context_window=4)
            p_values.append(result.p_value)
            forced_counts.append(forced)
        flagged = sum(p < 0.01 for p in p_values)
        print(
            f"{label:11s}: flagged_at_p<0.01={flagged}/{trials} "
            f"mean_forced={sum(forced_counts) / trials:.1f} "
            f"min_p={min(p_values):.3e}"
        )


if __name__ == "__main__":
    main()
