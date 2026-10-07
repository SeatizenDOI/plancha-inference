"""One-batch-ahead CPU preprocess overlap with GPU consumption.

While the caller runs inference on batch N, a single worker thread prepares
batch N+1. Disable with enabled=False for a sequential baseline.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterator, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def iter_with_preprocess_overlap(
    data_stream: Iterator[T],
    preprocess_fn: Callable[[T], R],
    enabled: bool = True,
) -> Iterator[tuple[T, R]]:
    """Yield (data, preprocessed) pairs; optionally overlap next preprocess."""
    if not enabled:
        for data in data_stream:
            yield data, preprocess_fn(data)
        return

    stream = iter(data_stream)
    try:
        first = next(stream)
    except StopIteration:
        return

    # One worker = exactly one batch ahead of the GPU consumer.
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(preprocess_fn, first)
        pending_data = first
        for data in stream:
            result = pending.result()
            next_fut = pool.submit(preprocess_fn, data)
            yield pending_data, result
            pending = next_fut
            pending_data = data
        yield pending_data, pending.result()
