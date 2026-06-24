"""Parallel execution helper consolidating the ``ProcessPoolExecutor + tqdm``
scaffolding that was duplicated across legacy scripts 02 and 03.

The ``worker_fn`` must be a *top-level* (module-level) callable so that it
is picklable by the worker pool. Each domain module therefore exposes its
own ``_process_one_case`` at module top level.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from tqdm import tqdm


def _resolve_workers(n_workers: Optional[int],
                     parallel: bool = True) -> int:
    """Return worker count: 1 = serial, >1 = parallel."""
    if not parallel:
        return 1
    if n_workers is None or n_workers <= 0:
        return max(1, (os.cpu_count() or 1))
    return int(n_workers)


def run_parallel(tasks: List[Any],
                 worker_fn: Callable[[Any], Tuple[bool, str, Dict[str, Any]]],
                 n_workers: Optional[int] = None,
                 parallel: bool = True,
                 desc: str = "Processing",
                 unit: str = "file") -> List[Tuple[Any, bool, str, Dict[str, Any]]]:
    """Execute ``worker_fn(task)`` for every task in ``tasks``.

    Args:
        tasks:          list of arbitrary task payloads (tuples, dicts, ...)
        worker_fn:      top-level callable returning ``(ok: bool, msg: str, result: dict)``
        n_workers:      process count. None → os.cpu_count()
        parallel:       if False, run serially in the current process
        desc, unit:     tqdm progress-bar label / unit

    Returns:
        list of ``(task, ok, msg, result)`` in *completion* order.
    """
    n_workers = _resolve_workers(n_workers, parallel)
    results: List[Tuple[Any, bool, str, Dict[str, Any]]] = []

    if n_workers == 1 or len(tasks) <= 1:
        for task in tqdm(tasks, desc=desc, unit=unit, disable=not tasks):
            try:
                ok, msg, result = worker_fn(task)
            except Exception as exc:  # noqa: BLE001
                ok, msg, result = False, f"exception: {exc!r}", {}
            results.append((task, ok, msg, result))
        return results

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        future_map = {pool.submit(worker_fn, task): task for task in tasks}
        for fut in tqdm(as_completed(future_map),
                        total=len(future_map), desc=desc, unit=unit):
            task = future_map[fut]
            try:
                ok, msg, result = fut.result()
            except Exception as exc:  # noqa: BLE001
                ok, msg, result = False, f"exception: {exc!r}", {}
            results.append((task, ok, msg, result))
    return results
