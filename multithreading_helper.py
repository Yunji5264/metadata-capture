from concurrent.futures import ThreadPoolExecutor, as_completed
import os

def collect_metadata_parallel(paths, worker, max_workers: int | None = None):
    """
    Execute `worker(path)` concurrently for all paths using a ThreadPoolExecutor.
    - `paths`: iterable of file paths (str or Path-like)
    - `worker`: callable that accepts a single path and returns a result dict
    - `max_workers`: optional override; if None a sensible default is chosen
    Returns: list of results aligned to the original `paths` order.
    """
    paths = [str(p) for p in paths]
    if max_workers is None:
        # For I/O-bound workloads, more threads are beneficial up to a limit
        max_workers = min(32, max(4, (os.cpu_count() or 1) * 5))

    index_of = {p: i for i, p in enumerate(paths)}
    results = [None] * len(paths)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(worker, p): p for p in paths}
        for fut in as_completed(future_map):
            p = future_map[fut]
            i = index_of[p]
            try:
                results[i] = fut.result()
            except Exception as e:
                # c
                results[i] = {"path": p, "error": f"{type(e).__name__}: {e}"}

    return results
