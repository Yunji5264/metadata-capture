import json
import hashlib
from datetime import datetime
from general_function import *
from uml_class import Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from reference import DATA_DIR, METADATA_DIR, CATALOG_PATH, PERF_DIR, DEFAULT_EXTS

# --- your imports (reuse your existing pipeline) ---
from metadata_selector import construct_dataset  # or from current file if same module


def ensure_dirs():
    """Create output directories if they do not exist."""
    os.makedirs(METADATA_DIR, exist_ok=True)


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
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:16]


def iter_data_files(root: str, allow_exts: Optional[Iterable[str]] = None) -> Iterable[str]:
    """Yield normalised file paths under root filtered by extensions (case-insensitive)."""
    allow = {e.lower() for e in (allow_exts or DEFAULT_EXTS)}
    for r, _, files in os.walk(root):
        for fn in files:
            if fn.startswith(".") or fn.endswith(".ini"):
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext in allow:
                yield os.path.normpath(os.path.join(r, fn))


def build_metadata_entry(data_path: str, ds: Dataset) -> Dict[str, Any]:
    """
    Build catalog entry from Dataset (data-centric).
    """
    meta_dict = ds.to_dict()

    filename = os.path.basename(data_path)
    meta_filename = f"{filename}.metadata.json"
    meta_path = os.path.join(METADATA_DIR, meta_filename)

    # If it's a ZIP (likely shapefile), compute uncompressed size if not already set
    unzipped_size = meta_dict.get("uncompressedSizeBytes")
    if (unzipped_size is None) and data_path.lower().endswith(".zip"):
        unzipped_size = uncompressed_zip_size(data_path)

    entry = {
        "id": dataset_id_from_path(data_path),
        "title": meta_dict.get("title"),
        "sourcePath": data_path,
        "metadataPath": meta_path,
        "fileType": meta_dict.get("fileType"),
        "dataFormat": meta_dict.get("dataFormat"),
        "updateFrequency": meta_dict.get("updateFrequency"),
        "theme": (meta_dict.get("theme") or {}).get("themeName") if meta_dict.get("theme") else None,
        "spatialGranularity": meta_dict.get("spatialGranularity"),
        "temporalGranularity": meta_dict.get("temporalGranularity"),
        "spatialScope": meta_dict.get("spatialScope"),
        "temporalScope": meta_dict.get("temporalScope"),
        # File size metrics
        "fileSizeBytes": meta_dict.get("fileSizeBytes"),
        "fileSizeHuman": meta_dict.get("fileSizeHuman"),
        "nRows": meta_dict.get("nRows"),
        "nCols": meta_dict.get("nCols"),
        "nRecords": meta_dict.get("nRecords"),
        "nFeatures": meta_dict.get("nFeatures"),
        "uncompressedSizeBytes": unzipped_size,
        # System info
        "checksum": file_checksum(data_path),
        "generatedAt": datetime.now().isoformat(timespec="seconds") + "Z",
        "version": 1,
    }
    return entry

def build_perf_entry(data_path: str, timings: Dict[str, float], ds: Dataset) -> Dict[str, Any]:
    """
    Build performance log entry (process-centric).
    """
    return {
        "datasetId": dataset_id_from_path(data_path),
        "sourcePath": data_path,
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


def process_file(path: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Worker wrapper:
    - build Dataset with timings
    - build metadata catalog entry
    - build performance entry
    Returns (meta_dict, meta_entry, perf_entry).
    """
    ds, timings = construct_dataset(path, measure=True)
    meta_dict = ds.to_dict()
    meta_entry = build_metadata_entry(path, ds)
    perf_entry = build_perf_entry(path, timings, ds)
    return meta_dict, meta_entry, perf_entry


def save_outputs(meta_dict: Dict[str, Any], entry: Dict[str, Any], perf_entry: Dict[str, Any]) -> None:
    """Persist per-file outputs: metadata JSON and perf JSON."""
    # 1) Save metadata json (path provided inside entry)
    meta_path = entry["metadataPath"]
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, ensure_ascii=False, indent=2)

    # 2) Save perf json (under METADATA_DIR/perf/<filename>.perf.json)
    os.makedirs(PERF_DIR, exist_ok=True)
    perf_filename = f"{os.path.basename(entry['sourcePath'])}.perf.json"
    perf_path = os.path.join(PERF_DIR, perf_filename)
    with open(perf_path, "w", encoding="utf-8") as f:
        json.dump(perf_entry, f, ensure_ascii=False, indent=2)


def build_catalog_for_dir_parallel(
    data_dir: str = DATA_DIR,
    max_workers: int = os.cpu_count() or 4,
    allow_exts: Optional[Iterable[str]] = None,
) -> None:
    """
    Parallel metadata + performance extraction for all files under `data_dir`.
    Uses a thread pool by default (switch to processes if CPU-bound).
    """
    # Ensure output dirs exist
    os.makedirs(METADATA_DIR, exist_ok=True)
    os.makedirs(PERF_DIR, exist_ok=True)

    # Discover files
    paths = list(iter_data_files(data_dir, allow_exts=allow_exts or DEFAULT_EXTS))
    if not paths:
        print(f"No files found under '{data_dir}'.")
        return

    print(f"Discovered {len(paths)} file(s); running with max_workers={max_workers}.")

    futures = {}
    catalog = load_catalog()

    # Submit tasks
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for p in paths:
            futures[ex.submit(process_file, p)] = p

        # Collect results as they complete
        for fut in as_completed(futures):
            p = futures[fut]
            fn = os.path.basename(p)
            try:
                meta_dict, meta_entry, perf_entry = fut.result()
                # Persist per-file outputs
                save_outputs(meta_dict, meta_entry, perf_entry)
                # Update in-memory catalog (single-threaded here)
                upsert_catalog_entry(catalog, meta_entry)
                print(f"[OK] {fn}")
            except Exception as ex:
                print(f"[ERR] {fn}: {ex}")

    # Finalise catalog (deterministic ordering)
    catalog.setdefault("datasets", []).sort(key=lambda x: ((x.get("title") or "").lower(), x.get("id", "")))
    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=4)
    print(f"Catalog written to {CATALOG_PATH}")

if __name__ == "__main__":
    # Example: only process certain extensions and use 8 workers
    build_catalog_for_dir_parallel(DATA_DIR, max_workers=8, allow_exts=DEFAULT_EXTS)

