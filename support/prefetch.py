"""Bounded CPU-side prefetching for batch sources.

Written for zmocap. The vendored inference code wants a batch iterator whose
next item is prepared on a worker thread while the GPU is busy; this module is
our own implementation of that contract. It is copied into the vendor tree by
``install.py`` so the third-party code imports *our* helper instead of anything
from the reference application.

Public entry point: :func:`batch_source`, a context manager yielding an
iterable (the source itself when prefetching is not worthwhile).
"""
from __future__ import annotations

import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

__all__ = ["batch_source", "prefetch_depth"]

# Never hold more than this much prepared CPU data, whatever the batch size is.
MAX_PREFETCH_BYTES = 256 * 1024 ** 2
# Leave this much RAM to the GPU runtime and the rest of ComfyUI.
RAM_HEADROOM = 6 * 1024 ** 3
MIN_CPUS = 8
MAX_DEPTH = 2


def prefetch_depth(batch_bytes: int = 0) -> int:
    """How many batches to prepare ahead of the consumer.

    ``batch_bytes`` is the size of one prepared batch in memory; when known it
    caps the depth so the queued data stays under ``MAX_PREFETCH_BYTES``.
    Returns 0 when the machine has no spare RAM or too few cores, which makes
    ``batch_source`` a no-op.
    """
    try:
        import psutil

        memory = psutil.virtual_memory()
        available = int(memory.available)
    except Exception:
        return 0
    if (os.cpu_count() or 1) < MIN_CPUS or available - RAM_HEADROOM <= 0:
        return 0
    depth = MAX_DEPTH
    if batch_bytes > 0:
        depth = min(depth, MAX_PREFETCH_BYTES // int(batch_bytes))
    return max(0, depth)


class _Prefetched:
    """Iterate ``source`` while preparing up to ``depth`` items ahead."""

    def __init__(self, source, depth: int) -> None:
        self._source = iter(source)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="zmocap-prep")
        self._pending: deque = deque()
        self._closed = False
        try:
            for _ in range(depth):
                self._pending.append(self._pool.submit(next, self._source))
        except BaseException:
            self.close()
            raise

    def __iter__(self):
        try:
            while self._pending:
                future = self._pending.popleft()
                try:
                    item = future.result()
                except StopIteration:
                    return
                self._pending.append(self._pool.submit(next, self._source))
                yield item
                del item
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        pending, self._pending = self._pending, deque()
        for future in pending:
            future.cancel()
        self._pool.shutdown(wait=True, cancel_futures=True)
        close = getattr(self._source, "close", None)
        if callable(close):
            close()


@contextmanager
def batch_source(source, *, batch_bytes: int = 0):
    """Yield ``source`` with bounded look-ahead preparation, or unchanged."""
    depth = prefetch_depth(batch_bytes)
    if not depth:
        yield source
        return
    prepared = _Prefetched(source, depth)
    try:
        yield prepared
    finally:
        prepared.close()
