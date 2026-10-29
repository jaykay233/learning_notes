from typing import Iterable, List, Sequence

BLOCK_SIZE = 4
SEQUENCE_LENGTH = 12
QUERY_POSITION = 10
WINDOW_SIZE = 6
TOP_K_BLOCKS = (0, 2)


def gather_blocks(
    block_ids: Iterable[int],
    block_size: int,
    sequence_length: int,
) -> List[int]:
    """将逻辑 block id 展开成 token 位置，并裁掉最后一个短 block。"""
    result: List[int] = []
    for block_id in block_ids:
        start = block_id * block_size
        end = min(start + block_size, sequence_length)
        result.extend(range(start, end))
    return result


def sliding_window(
    query_position: int,
    window_size: int,
) -> List[int]:
    """返回包含当前 token 的因果 SWA 窗口。"""
    start = max(0, query_position - window_size + 1)
    return list(range(start, query_position + 1))


def apply_causal_mask(
    positions: Sequence[int],
    query_position: int,
) -> List[int]:
    return [position for position in positions if position <= query_position]


def main() -> None:
    gathered = gather_blocks(
        TOP_K_BLOCKS, BLOCK_SIZE, SEQUENCE_LENGTH
    )
    sparse_visible = apply_causal_mask(gathered, QUERY_POSITION)
    swa_visible = sliding_window(QUERY_POSITION, WINDOW_SIZE)

    sparse_set = set(sparse_visible)
    swa_set = set(swa_visible)

    print(f"gathered KV       = {gathered}")
    print(f"sparse after mask = {sparse_visible}")
    print(f"SWA visible       = {swa_visible}")
    print(f"overlap           = {sorted(sparse_set & swa_set)}")
    print(f"sparse only       = {sorted(sparse_set - swa_set)}")
    print(f"SWA only          = {sorted(swa_set - sparse_set)}")

    assert sparse_visible == [0, 1, 2, 3, 8, 9, 10]
    assert swa_visible == [5, 6, 7, 8, 9, 10]
    assert sorted(sparse_set & swa_set) == [8, 9, 10]


if __name__ == "__main__":
    main()
