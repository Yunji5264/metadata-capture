from general_function import *

def get_granularity(granularities, hierarchies) -> str:

    # Step 1: pick most-specific match per path
    picks: List[str] = []
    for path in hierarchies:
        top = get_most_specific_in_path(granularities, path)
        if top:
            return top

    if not picks:
        return None

    # Step 2: check if all picks lie within a single path
    pickset: Set[str] = set(picks)
    for path in hierarchies:
        path_names = {level_name(lvl) for lvl in path}
        if pickset.issubset(path_names):
            # Collapse to that single path's most-general match
            collapsed = get_most_specific_in_path(granularities, path)
            return collapsed if collapsed else None

    # Step 3: return per-path picks (deduplicated, order-preserving)
    seen: Set[str] = set()
    out: List[str] = []
    for p in picks:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out[0]