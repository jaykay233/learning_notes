#!/usr/bin/env python3
"""CPU model of ragged token packing and paged KV address mapping."""

from __future__ import annotations


def cumulative(lengths: list[int]) -> list[int]:
    result = [0]
    for length in lengths:
        result.append(result[-1] + length)
    return result


def unpack_ragged(values: list[int], lengths: list[int]) -> list[list[int]]:
    offsets = cumulative(lengths)
    return [
        values[offsets[request] : offsets[request + 1]]
        for request in range(len(lengths))
    ]


def paged_lookup(
    logical_values: list[int],
    page_size: int,
    block_table: list[int],
) -> list[int]:
    result: list[int] = []
    for position, expected in enumerate(logical_values):
        logical_page = position // page_size
        page_offset = position % page_size
        physical_page = block_table[logical_page]
        value = page_pool[physical_page][page_offset]
        assert value == expected
        result.append(value)
    return result


q_lens = [3, 1, 4]
kv_lens = [5, 2, 6]
page_size = 4

q_flat = [100 + i for i in range(sum(q_lens))]
kv_flat = [200 + i for i in range(sum(kv_lens))]

cu_q = cumulative(q_lens)
cu_kv = cumulative(kv_lens)

request_q = unpack_ragged(q_flat, q_lens)
request_kv = unpack_ragged(kv_flat, kv_lens)

page_pool = {
    0: [200, 201, 202, 203],
    1: [204, -1, -1, -1],
    3: [205, 206, -1, -1],
    5: [207, 208, 209, 210],
    2: [211, 212, -1, -1],
}
request_2_pages = [5, 2]

print(f"q_lens={q_lens}, cu_seqlens_q={cu_q}")
print(f"kv_lens={kv_lens}, cu_seqlens_k={cu_kv}")
for request in range(len(q_lens)):
    print(
        f"request {request}: q={request_q[request]}, "
        f"kv={request_kv[request]}"
    )

reconstructed_kv_2 = paged_lookup(request_kv[2], page_size, request_2_pages)
print(f"request 2 logical KV={request_kv[2]}")
print(f"request 2 physical pages={request_2_pages}")
print(f"request 2 reconstructed KV={reconstructed_kv_2}")

assert cu_q == [0, 3, 4, 8]
assert cu_kv == [0, 5, 7, 13]
assert request_q == [[100, 101, 102], [103], [104, 105, 106, 107]]
assert request_kv == [
    [200, 201, 202, 203, 204],
    [205, 206],
    [207, 208, 209, 210, 211, 212],
]
assert reconstructed_kv_2 == request_kv[2]
print("PASS: ragged boundaries and page-table mapping are consistent")
