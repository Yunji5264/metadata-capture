import os
import json
import glob
import pandas as pd
from typing import List, Dict, Any, Optional
from reference import PERF_DIR

def _safe_get(d: Dict[str, Any], key: str, default=None):
    """Return d[key] if exists else default."""
    return d.get(key, default) if isinstance(d, dict) else default

def _flatten_perf_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one perf.json entry into a single-row dict."""
    timings = _safe_get(entry, "timings", {}) or {}
    return {
        # identifiers
        "datasetId": _safe_get(entry, "datasetId"),
        "sourcePath": _safe_get(entry, "sourcePath"),
        "executedAt": _safe_get(entry, "executedAt"),
        # size on disk
        "fileSizeBytes": _safe_get(entry, "fileSizeBytes"),
        # timings (seconds)
        "read_df_sec": timings.get("read_df_sec"),
        "semantic_helper_sec": timings.get("semantic_helper_sec"),
        "classify_attributes_sec": timings.get("classify_attributes_sec"),
        "scopes_granularities_sec": timings.get("scopes_granularities_sec"),
        "find_common_theme_sec": timings.get("find_common_theme_sec"),
        "transform_result_sec": timings.get("transform_result_sec"),
        "total_sec": timings.get("total_sec"),
    }

def load_perf_entries(perf_dir: str = PERF_DIR) -> List[Dict[str, Any]]:
    """Read all *.perf.json files and return flattened rows."""
    rows: List[Dict[str, Any]] = []
    pattern = os.path.join(perf_dir, "*.perf.json")
    for path in glob.glob(pattern):
        try:
            with open(path, "r", encoding="utf-8") as f:
                entry = json.load(f)
            rows.append(_flatten_perf_entry(entry))
        except Exception as ex:
            print(f"[WARN] Failed to parse {path}: {ex}")
    return rows

def build_perf_table(
    perf_dir: str = PERF_DIR,
    out_csv: Optional[str] = os.path.join("metadata", "perf_summary.csv"),
    out_parquet: Optional[str] = None,
) -> pd.DataFrame:
    """
    Aggregate all perf logs into a single table and save to disk.
    Returns the DataFrame for further analysis.
    """
    rows = load_perf_entries(perf_dir)
    if not rows:
        print(f"No perf files found under {perf_dir}.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Normalise dtypes and sort by executedAt (if present)
    if "executedAt" in df.columns:
        # If executedAt is ISO string, convert to datetime for sorting
        df["executedAt"] = pd.to_datetime(df["executedAt"], errors="coerce")
        df = df.sort_values(["executedAt", "datasetId"], ascending=[False, True])

    # Save outputs
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"[OK] Wrote CSV: {out_csv}")
    if out_parquet:
        os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
        df.to_parquet(out_parquet, index=False)
        print(f"[OK] Wrote Parquet: {out_parquet}")

    return df

if __name__ == "__main__":
    # Example usage
    df = build_perf_table(
        perf_dir = PERF_DIR,
        out_csv = "../perf_summary.csv",
        out_parquet = "../perf_summary.parquet",
    )
    print(df)
