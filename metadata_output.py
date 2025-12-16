import inspect
from collections.abc import Generator, Iterable
import hashlib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from general_function import *  # your shared helpers
from uml_class import Dataset
from metadata_selector import construct_dataset, construct_dataset_new  # your pipeline entry
from reference import DATA_DIR, METADATA_DIR, CATALOG_PATH, PERF_DIR, DEFAULT_EXTS


# ----------------------------- Small utilities -----------------------------

def make_json_safe(obj):
    """
    Recursively convert sets and other non-serialisable types to JSON-safe ones.
    - set → list
    - any nested structure handled
    """
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, set):
        return sorted(make_json_safe(v) for v in obj)  # keep deterministic order
    else:
        return obj

def file_checksum(path: str, algo: str = "sha256", chunk_size: int = 1 << 20) -> str:
    """Compute a reproducible checksum for provenance and change detection."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def dataset_id_from_path(path: str) -> str:
    """Derive a simple stable ID from file path (customise if you prefer UUIDs)."""
    return hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()[:16]


def iter_data_files(root: str, allow_exts: Optional[Iterable[str]] = None) -> Iterable[str]:
    """
    Yield normalised file paths under root filtered by extensions (case-insensitive).
    Hidden files and .ini files are ignored.
    """
    allow = {e.lower() for e in (allow_exts or DEFAULT_EXTS)}
    for r, _, files in os.walk(root):
        for fn in files:
            if fn.startswith(".") or fn.endswith(".ini"):
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext in allow:
                yield os.path.normpath(os.path.join(r, fn))


def _file_mtime_int(p: str) -> int:
    """Return the file mtime as an int (seconds)."""
    try:
        return int(os.path.getmtime(p))
    except Exception:
        return -1


def _stem(p: str) -> str:
    """Return filename without extension."""
    return os.path.splitext(os.path.basename(p))[0]


def _outputs_exist_for(p: str) -> bool:
    """
    Fallback existence check: both metadata and perf outputs must exist
    according to your naming/layout conventions.
    """
    s = _stem(p)
    meta_path = os.path.join(METADATA_DIR, f"{s}.json")  # adjust if you use another pattern
    perf_path = os.path.join(PERF_DIR, f"{s}.perf.json")
    return os.path.exists(meta_path) and os.path.exists(perf_path)


def was_processed_successfully(p: str, catalog: Dict[str, Any]):
    """
    Decide whether to skip processing for file `p`.

    Strategy (fast and robust):
      1) Prefer an exact match in the catalog by:
         - dataset id or sourcePath
         - matching sourceMtime
         - status == "ok"
      2) If the catalog is missing/incomplete, fallback to checking output files.

    This avoids recomputing checksums on every run.
    """


    mtime = _file_mtime_int(p)
    pid = dataset_id_from_path(p)
    abs_p = os.path.abspath(p)

    try:
        datasets = catalog.get("datasets", []) if isinstance(catalog, dict) else []
        for e in datasets:
            same_id = (e.get("id") == pid)
            same_path = os.path.abspath(e.get("sourcePath", "")) == abs_p
            if not (same_id or same_path):
                continue

            # mtime & status must match
            if int(e.get("sourceMtime", -2)) == mtime and e.get("status", "").lower() == "ok":
                # Verify metadata file exists
                meta_path = e.get("metadataPath")
                if meta_path and os.path.exists(meta_path):
                    return True
                # metadata record exists but file missing → force reprocess
                return False
    except Exception:
        pass

    # Fallback: if both outputs exist, treat as processed
    return _outputs_exist_for(p)


# ------------------------- Catalog / entry construction -------------------------


def _extract_theme_names(meta_theme) -> Optional[List[str]]:
    """
    Normalize various theme shapes into a list of theme names (list[str]).
    Accepts: None / dict / list[dict] / str / list[str] / Theme / generator / iterable.
    Returns None if nothing valid is found.
    """
    if meta_theme is None:
        return None

    # Materialize generators to avoid one-time consumption surprises
    if inspect.isgenerator(meta_theme) or isinstance(meta_theme, Generator):
        meta_theme = list(meta_theme)

    names: List[str] = []
    seen = set()

    def _add_name(val):
        if not val:
            return
        name = str(val).strip()
        if name and name not in seen:
            names.append(name)
            seen.add(name)

    def _extract_from_dict(d: Dict[str, Any]):
        # Try several common keys for theme name
        return d.get("themeName") or d.get("name") or d.get("title") or d.get("theme_name")

    # Dict: single theme object
    if isinstance(meta_theme, dict):
        _add_name(_extract_from_dict(meta_theme))
        return names or None

    # String: single theme name
    if isinstance(meta_theme, str):
        _add_name(meta_theme)
        return names or None

    # Iterable (list/tuple/set etc.)
    if isinstance(meta_theme, Iterable) and not isinstance(meta_theme, (bytes, bytearray, str)):
        for it in meta_theme:
            if isinstance(it, dict):
                _add_name(_extract_from_dict(it))
            elif isinstance(it, str):
                _add_name(it)
            else:
                # Theme-like object with .themeName
                cand = getattr(it, "themeName", None)
                _add_name(cand)
        return names or None

    # Fallback: Theme-like single object
    cand = getattr(meta_theme, "themeName", None)
    _add_name(cand)
    return names or None


def build_metadata_entry(data_path: str, ds: "Dataset") -> Dict[str, Any]:
    """
    Build catalog entry from Dataset (data-centric).
    - Matches the updated Dataset where `theme` is Set[Theme] and serializes to list[dict].
    - Stores themes as list[str] (theme names) for the catalog entry.
    - Adds `status="ok"` and `sourceMtime` so we can skip unchanged files on next runs.
    """
    meta_dict = ds.to_dict()

    filename = os.path.basename(data_path)
    # If you prefer "<stem>.json" adjust accordingly in save_outputs().
    meta_filename = f"{filename}.metadata.json"
    meta_path = os.path.abspath(os.path.join(METADATA_DIR, meta_filename))

    # If it's a ZIP (e.g., shapefile), compute uncompressed size if not already set
    unzipped_size = meta_dict.get("uncompressedSizeBytes")
    if (unzipped_size is None) and data_path.lower().endswith(".zip"):
        unzipped_size = uncompressed_zip_size(data_path)

    # Normalize theme(s) to a list of names; compatible with Dataset.to_dict() new shape (list[dict])
    theme_names = _extract_theme_names(meta_dict.get("theme"))

    entry = {
        # Identity / provenance
        "id": dataset_id_from_path(data_path),
        "title": meta_dict.get("title"),
        "sourcePath": os.path.abspath(data_path),
        "sourceMtime": _file_mtime_int(data_path),            # used for skip comparison
        "metadataPath": meta_path,
        "status": "ok",                                       # marks last run success

        # Core descriptors
        "fileType": meta_dict.get("fileType"),
        "dataFormat": meta_dict.get("dataFormat"),
        "updateFrequency": meta_dict.get("updateFrequency"),
        # Store as list[str]; if you must keep a single string, use: theme_names[0] if theme_names else None
        "theme": theme_names,
        "spatialGranularity": meta_dict.get("spatialGranularity"),
        "temporalGranularity": meta_dict.get("temporalGranularity"),
        "spatialScope": meta_dict.get("spatialScope"),
        "temporalScope": meta_dict.get("temporalScope"),

        # File size / counts
        "fileSizeBytes": meta_dict.get("fileSizeBytes"),
        "fileSizeHuman": meta_dict.get("fileSizeHuman"),
        "nRows": meta_dict.get("nRows"),
        "nCols": meta_dict.get("nCols"),
        "nRecords": meta_dict.get("nRecords"),
        "nFeatures": meta_dict.get("nFeatures"),
        "uncompressedSizeBytes": unzipped_size,

        # System info
        "checksum": file_checksum(data_path),
        # Note: This uses local time with a 'Z' suffix; switch to datetime.utcnow() if true UTC is required.
        "generatedAt": datetime.now().isoformat(timespec="seconds") + "Z",
        "version": 1,
    }
    return entry


def build_perf_entry(data_path: str, timings: Dict[str, float], ds: Dataset) -> Dict[str, Any]:
    """Build performance log entry (process-centric)."""
    return {
        "datasetId": dataset_id_from_path(data_path),
        "sourcePath": os.path.abspath(data_path),
        "fileSizeBytes": ds.fileSizeBytes,
        "timings": timings,
        "executedAt": datetime.now().isoformat(timespec="seconds") + "Z"
    }


def load_catalog() -> Dict[str, Any]:
    """Load existing catalog or return an empty skeleton."""
    if os.path.exists(CATALOG_PATH):
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"datasets": []}


def upsert_catalog_entry(catalog: Dict[str, Any], new_entry: Dict[str, Any]) -> None:
    """Insert or update a dataset entry in catalog, keyed by id (in-place)."""
    datasets = catalog.setdefault("datasets", [])
    for i, e in enumerate(datasets):
        if e.get("id") == new_entry["id"]:
            datasets[i] = new_entry
            break
    else:
        datasets.append(new_entry)


# ------------------------------- Processing I/O -------------------------------

def process_file(path: str):
    """
    Execute the dataset construction pipeline for a single file.

    Returns:
        meta_dict: full metadata dict (serialisable)
        meta_entry: catalog entry to be merged into the global catalog
        perf_entry: performance entry (or None if timing was skipped)
    """
    ds, timings = construct_dataset(path, measure=True)
    meta_dict = ds.to_dict()
    meta_entry = build_metadata_entry(path, ds)

    # If timing was skipped due to semantic cache hit, do not produce perf entry
    perf_entry = None
    if not timings.get("timing_skipped"):
        perf_entry = build_perf_entry(path, timings, ds)

    return meta_dict, meta_entry, perf_entry


def save_outputs(meta_dict, entry, perf_entry):
    # metadata.json
    meta_path = entry["metadataPath"]
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)

    safe_meta = make_json_safe(meta_dict)   # <--- convert before dump
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(safe_meta, f, ensure_ascii=False, indent=2)

    # perf.json (only if present)
    if perf_entry:
        os.makedirs(PERF_DIR, exist_ok=True)
        perf_filename = f"{os.path.basename(entry['sourcePath'])}.perf.json"
        perf_path = os.path.join(PERF_DIR, perf_filename)
        safe_perf = make_json_safe(perf_entry)
        with open(perf_path, "w", encoding="utf-8") as f:
            json.dump(safe_perf, f, ensure_ascii=False, indent=2)



# ----------------------------- Parallel catalogue -----------------------------

def build_catalog_for_dir_parallel(
    data_dir: str = DATA_DIR,
    max_workers: int = os.cpu_count() or 4,
    allow_exts: Optional[Iterable[str]] = None,
    *,
    force: bool = False,
) -> None:
    """
    Parallel metadata + performance extraction for all files under `data_dir`.
    If force=True, reprocess ALL files regardless of previous results.
    Otherwise, skip files already processed successfully with up-to-date source mtime.

    NOTE: Set max_workers=1 to run in strict sequential mode (no thread pool).
    """
    # Ensure output dirs exist
    os.makedirs(METADATA_DIR, exist_ok=True)
    os.makedirs(PERF_DIR, exist_ok=True)

    # Discover candidate files
    paths = list(iter_data_files(data_dir, allow_exts=allow_exts or DEFAULT_EXTS))
    if not paths:
        print(f"No files found under '{data_dir}'.")
        return

    # Load current catalog and decide pending jobs
    catalog = load_catalog()

    if force:
        pending = paths
        skipped = 0
        print("[INFO] force=True → will reprocess all files (no cache skipping).")
    else:
        pending = [p for p in paths if not was_processed_successfully(p, catalog)]
        skipped = len(paths) - len(pending)

    if skipped:
        print(f"Skipped {skipped}/{len(paths)} file(s) already processed and up-to-date.")
    if not pending:
        print("All files are up-to-date. Nothing to do.")
        return

    print(f"Discovered {len(paths)} file(s); {len(pending)} to process; max_workers={max_workers}.")

    # -------- Sequential fast path when max_workers <= 1 --------
    if max_workers <= 1:
        for p in pending:
            fn = os.path.basename(p)
            try:
                meta_dict, meta_entry, perf_entry = process_file(p)
                # Persist per-file outputs
                save_outputs(meta_dict, meta_entry, perf_entry)
                # Update in-memory catalog
                upsert_catalog_entry(catalog, meta_entry)
                print(f"[OK] {fn}")
            except Exception as ex:
                print(f"[ERR] {fn}: {ex}")
        # Finalise catalog
        catalog.setdefault("datasets", []).sort(
            key=lambda x: ((x.get("title") or "").lower(), x.get("id", ""))
        )
        with open(CATALOG_PATH, "w", encoding="utf-8") as f:
            safe_catalog = make_json_safe(catalog)
            json.dump(safe_catalog, f, ensure_ascii=False, indent=4)
        print(f"Catalog written to {CATALOG_PATH}")
        return

    # -------- Original threaded branch --------
    futures: Dict[Any, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for p in pending:
            futures[ex.submit(process_file, p)] = p

        for fut in as_completed(futures):
            p = futures[fut]
            fn = os.path.basename(p)
            try:
                meta_dict, meta_entry, perf_entry = fut.result()
                save_outputs(meta_dict, meta_entry, perf_entry)
                upsert_catalog_entry(catalog, meta_entry)
                print(f"[OK] {fn}")
            except Exception as ex:
                print(f"[ERR] {fn}: {ex}")

    catalog.setdefault("datasets", []).sort(
        key=lambda x: ((x.get("title") or "").lower(), x.get("id", ""))
    )
    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        safe_catalog = make_json_safe(catalog)
        json.dump(safe_catalog, f, ensure_ascii=False, indent=4)
    print(f"Catalog written to {CATALOG_PATH}")



# ---------------------------------- CLI entry ---------------------------------

if __name__ == "__main__":
    # Example: only process certain extensions and use 8 workers
    build_catalog_for_dir_parallel(DATA_DIR, max_workers=8, allow_exts=DEFAULT_EXTS, force=True)
