import hashlib
import pandas as pd
import geopandas as gpd
import json
import openpyxl
import numpy as np
import unicodedata
import re
import os
import zipfile
from typing import Dict, List, Any, Tuple, Set, Optional, Iterable, Union
from pathlib import Path
from reference import CACHE_DIR

ColT = Union[str, int]

def _sample_bytes(path: str, size: int = 256_000) -> bytes:
    with open(path, "rb") as f:
        return f.read(size)

def _guess_encoding(sample: bytes) -> str:
    if sample.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            sample.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin1"

def _guess_sep(sample_text: str):
    cands = [",", ";", "\t", "|"]
    counts = {sep: sample_text.count(sep) for sep in cands}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else None

def _cache_path(src_path: str | os.PathLike, ext: str = ".parquet") -> str:
    """
    Return a stable cache file path OUTSIDE the data_dir so it won't be scanned as a dataset.
    Uses (abs path + mtime) hash to invalidate when source changes.
    """
    src_path = Path(src_path).resolve()
    os.makedirs(CACHE_DIR, exist_ok=True)
    mtime = src_path.stat().st_mtime if src_path.exists() else 0
    key = f"{src_path.as_posix()}|{mtime}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{h}{ext}")

def _is_cache_fresh(src: Path, cache: Path) -> bool:
    try:
        return cache.exists() and cache.stat().st_mtime >= src.stat().st_mtime
    except Exception:
        return False


# Normalisation helpers
def strip_accents(s: str):
    """Remove accents; normalise spaces/hyphens/apostrophes; fix common encoding issues."""
    if not isinstance(s, str):
        return s

    # --- Step 1: fix common mis-encodings (©, ¨e, etc.) ---
    replacements = {
        "Ã©": "é",   # common CSV bug: ma©tropole → métropole
        "Æ": "AE",
        "æ": "ae",
        "œ": "oe",  # œufs → oeufs
    }
    for bad, good in replacements.items():
        s = s.replace(bad, good)

    # --- Step 2: remove accents properly ---
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # --- Step 3: unify apostrophes and separators ---
    s = s.replace("’", "'").replace("`", "'").replace("´", "'")
    s = re.sub(r"[-_\s]+", " ", s)

    return s.strip()


def norm_name(v):
    """Normalise a display name for robust comparison."""
    if pd.isna(v):
        return ""
    return strip_accents(str(v)).lower()

def norm_code(v):
    """Normalise a code for robust comparison (uppercase, trimmed)."""
    if pd.isna(v):
        return ""
    return str(v).strip().upper()

def is_numeric_series(s):
    return pd.api.types.is_numeric_dtype(s)

def is_string_series(s):
    return pd.api.types.is_string_dtype(s) or s.dtype == object

def not_null_ratio(s):
    n = len(s)
    return float(s.notna().sum())/n if n else 0.0

def normalise_colname(name: str) -> str:
    """Lightweight colname normaliser for pattern matching."""
    # convert to string and lowercase
    s = str(name).lower()

    # remove accents using unicodedata
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # replace whitespace and hyphens with underscore
    s = re.sub(r"[\s\-]+", "_", s)

    return s

# ---------- helpers for zip + /vsizip/ ----------

def _first_member_with_suffix(zf: zipfile.ZipFile, *suffixes: str) -> str | None:
    """Return the first entry inside the zip matching any of the given suffixes (case-insensitive)."""
    suf = tuple(s.lower() for s in suffixes)
    for n in zf.namelist():
        ln = n.lower()
        if ln.startswith("__macosx/"):
            continue
        if ln.endswith(suf):
            return n
    return None

def _vsizip_path(zip_path: Path, inner_path: str) -> str:
    """Build a GDAL /vsizip/ path with forward slashes (works on Windows too)."""
    return f"/vsizip/{zip_path.as_posix()}/{inner_path}"

def _maybe_drop_geometry(gdf: gpd.GeoDataFrame, keep_geometry: bool = True):
    """Optionally drop active geometry to return a plain DataFrame."""
    if keep_geometry:
        return gdf
    return gdf.drop(columns=[gdf.geometry.name])


# ---------- your existing readers (unchanged logic) ----------

# -------- engine chooser --------
def _engine_for_excel(path: str) -> tuple[str|None, str|None]:
    ext = Path(path).suffix.lower()
    if ext in {".xlsx", ".xlsm"}:
        return "openpyxl", "pip install openpyxl"
    if ext == ".xls":
        return "xlrd", "pip install xlrd"
    if ext == ".xlsb":
        return "pyxlsb", "pip install pyxlsb"
    if ext == ".ods":
        return "odf", "pip install odfpy"
    return None, None

# -------- helpers --------
def _peek_block(path: str, sheet, engine: str|None, nrows: int = 1000) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet, header=None, nrows=nrows, engine=engine)

def _detect_header_row(block: pd.DataFrame) -> int|None:
    # pick the row with max non-nulls as header line
    rc = block.count(axis=1)
    if rc.empty or rc.max() == 0:
        return None
    return int(rc.idxmax())

def _norm(s: str) -> str:
    s = str(s) if s is not None else ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _is_description_sheet(sheet_name: str, header_row_values: list) -> bool:
    # 1) name-based signal
    name_pat = re.compile(
        r"(description\s+(des\s+)?(données|variables)|"
        r"data\s+dictionary|dictionary|liste\s+des\s+variables|variables?)",
        re.IGNORECASE
    )
    if name_pat.search(str(sheet_name) or ""):
        return True

    # 2) header-based signal
    headers = [_norm(x) for x in header_row_values]
    # typical dictionary headers
    key_tokens = {"description", "détails", "définition", "definition", "explication"}
    var_tokens = {"variable", "champ", "field", "colonne", "column", "attribut"}
    if any(h in key_tokens for h in headers) and any(h in var_tokens for h in headers):
        return True
    # 2-column pattern: ["variable", "description"] or similar
    if len(headers) in (2, 3) and any(h in var_tokens for h in headers) and any(h in key_tokens for h in headers):
        return True

    return False

# -------- main --------
def read_excel_workbook_with_desc_detection(file_path: str):
    """
    Read all sheets; for each, detect header row, read a proper DataFrame,
    and flag whether the sheet looks like a 'description' (data dictionary).
    Returns: list of dicts with keys:
      - sheet_index, sheet_name, df, header_row, is_description
    """
    engine, hint = _engine_for_excel(file_path)
    try:
        with pd.ExcelFile(file_path, engine=engine) as xls:
            results = []
            for i, sheet_name in enumerate(xls.sheet_names):
                # 1) peek to find header row
                block = _peek_block(file_path, sheet_name, engine=engine, nrows=1000).dropna(how="all")
                if block.empty:
                    continue
                hdr = _detect_header_row(block)
                # guard: if none, skip this sheet
                if hdr is None:
                    continue

                # 2) final read with header
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=hdr, engine=engine)

                # 3) description detection using sheet name and header row
                header_values = block.iloc[hdr].tolist()
                is_desc = _is_description_sheet(sheet_name, header_values)

                results.append({
                    "sheet_index": i,
                    "sheet_name": sheet_name,
                    "df": df,
                    "header_row": hdr,
                    "is_description": is_desc,
                })
            return results
    except ImportError as e:
        raise ImportError(f"Missing Excel engine for {Path(file_path).suffix.lower()}. Try: {hint}") from e

def count_str(df, num_row):
    row = df.iloc[num_row]  # 取第6行（Series）
    str_count = row.apply(lambda x: isinstance(x, str)).sum()
    return str_count

def _is_empty(x):
    if pd.isna(x):
        return True
    if isinstance(x, str):
        return x.strip() == ""
    return False

def test_title_with_null(df, num_row, max_num):
    nb_null = max_num - count_str(df, num_row)
    row_first = df.iloc[num_row,:nb_null]
    empty_mask = row_first.apply(_is_empty)
    return empty_mask.all()

def excel_EL(file_path, sheet_name=0, usecols=None, cache_parquet: bool = True):
    """
    Excel loader with automatic header-row detection.
    - Always previews the *first sheet* to detect a header row when no explicit preview target is provided.
    - Supports sheet selection by index or name.
    - If sheet_name is None / "first", it loads the first sheet.
    - Caches a Parquet copy (single-sheet mode) when enabled and cache is fresh.

    Parameters
    ----------
    file_path : str or Path
        Path to the Excel workbook.
    sheet_name : int | str | None, default 0
        Sheet index or name. If None or "first", the first sheet is used.
    usecols : list[str] | str | None
        Columns to load (passed through to pandas).
    cache_parquet : bool, default True
        Whether to read/write a Parquet cache alongside the source.

    Returns
    -------
    pandas.DataFrame
        Loaded sheet as a DataFrame with a detected header row.
    """
    src = Path(file_path)
    pq = _cache_path(file_path, ".parquet")

    # Use cache only for single-sheet read
    if cache_parquet and _is_cache_fresh(src, pq):
        try:
            return pd.read_parquet(pq, columns=usecols)
        except Exception:
            pass  # fall back to reading Excel

    engine, hint = _engine_for_excel(file_path)
    try:
        with pd.ExcelFile(file_path, engine=engine) as xls:
            # Resolve the target sheet to load
            if sheet_name in (None, "first"):
                target_sheet = xls.sheet_names[0] if xls.sheet_names else 0
            elif isinstance(sheet_name, int):
                # Clamp index to valid range
                if not xls.sheet_names:
                    raise ValueError("Workbook has no sheets.")
                idx = max(0, min(sheet_name, len(xls.sheet_names) - 1))
                target_sheet = xls.sheet_names[idx]
            else:
                # Assume it's a sheet name (string)
                target_sheet = sheet_name

            # --- Preview FIRST sheet to detect header row ---
            # We always use the first sheet for header-row detection to avoid dict return.
            first_sheet = xls.sheet_names[0] if xls.sheet_names else target_sheet
            block = pd.read_excel(file_path, sheet_name=first_sheet,
                                  header=None, nrows=1000, engine=engine)
    except ImportError as e:
        raise ImportError(f"Missing Excel engine for {src.suffix.lower()}. Try: {hint}") from e

    # Detect header row from the preview block
    block = block.dropna(how="all")
    if block.empty:
        header_row = 0
    else:
        row_counts = block.count(axis=1)
        header_row = int(row_counts.idxmax()) if not row_counts.empty else 0

    # Final read for the *target* sheet, using the detected header row
    df = pd.read_excel(
        file_path,
        sheet_name=target_sheet,
        header=header_row,
        engine=engine,
        usecols=usecols
    )

    # Cache to Parquet (best-effort)
    if cache_parquet:
        try:
            df.to_parquet(pq, index=False)
        except Exception:
            pass

    return df

def csv_EL(file_path):
    encs = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    seps = [None, ";", ",", "\t", "|"]
    for enc in encs:
        for sep in seps:
            try:
                df = pd.read_csv(file_path, encoding=enc, sep=sep, low_memory=False, on_bad_lines="skip")
                if df.shape[1] > 1: return df
            except Exception:
                pass
    raise ValueError(f"Could not parse {file_path} as CSV/TSV.")

def geojson_EL(file_path):
    """Read GeoJSON data"""
    gdf = gpd.read_file(file_path)
    return gdf

def json_EL(file_path):
    """Read JSON or NDJSON file into DataFrame"""
    try:
        return pd.read_json(file_path, lines=True)
    except ValueError:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        return pd.json_normalize(data)

def parquet_EL(file_path):
    """Read Parquet data"""
    df = pd.read_parquet(file_path, engine="pyarrow")
    return df


def detect_json_type(filepath):
    """Detect GeoJSON vs generic JSON by inspecting the 'type' field."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "type" in data:
        geo_types = {"FeatureCollection", "Feature", "Point", "Polygon", "MultiPolygon", "LineString"}
        if data["type"] in geo_types:
            return geojson_EL(filepath)
    return json_EL(filepath)


# ---------- enhanced Shapefile reader (zip without extraction) ----------

def shapefile_EL(zip_file_path: str):
    """
    Read a Shapefile directly from a .zip (no extraction) using GDAL /vsizip/.
    If given a .shp path, read directly.
    """
    p = Path(zip_file_path).resolve()
    if p.suffix.lower() == ".shp":
        return gpd.read_file(p.as_posix())

    if p.suffix.lower() != ".zip":
        raise ValueError("shapefile_EL expects a .zip (or .shp) path.")

    with zipfile.ZipFile(p, "r") as zf:
        shp_inside = _first_member_with_suffix(zf, ".shp")
        if not shp_inside:
            raise FileNotFoundError("No .shp file found inside the zip.")
    vsip = _vsizip_path(p, shp_inside)
    return gpd.read_file(vsip)


# ---------- NEW: FlatGeobuf (.fgb) ----------

def fgb_EL(path: str):
    """
    Read a FlatGeobuf (.fgb). If a .zip is provided, find the first .fgb inside and use /vsizip/.
    """
    p = Path(path).resolve()
    if p.suffix.lower() == ".fgb":
        return gpd.read_file(p.as_posix())

    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p, "r") as zf:
            fgb_inside = _first_member_with_suffix(zf, ".fgb")
            if not fgb_inside:
                raise FileNotFoundError("No .fgb file found inside the zip.")
        vsip = _vsizip_path(p, fgb_inside)
        return gpd.read_file(vsip)

    raise ValueError("fgb_EL expects a .fgb or .zip path.")


# ---------- NEW: KML / KMZ (multi-layer support) ----------

def _list_layers(pathlike: str) -> list[str]:
    """List layers for a datasource using pyogrio first, then Fiona as fallback."""
    try:
        from pyogrio import list_layers  # preferred when available
        return [l[0] for l in list_layers(pathlike)]
    except Exception:
        try:
            import fiona
            return fiona.listlayers(pathlike)
        except Exception as e:
            raise RuntimeError(f"Cannot list layers from: {pathlike}. Error: {e}")

def _concat_layers(path_like: str, layers: list[str]) -> gpd.GeoDataFrame:
    """Read multiple layers and concatenate, tagging the source layer into 'src_layer'."""
    frames = []
    for lyr in layers:
        sub = gpd.read_file(path_like, layer=lyr)
        sub["src_layer"] = lyr
        frames.append(sub)
    if not frames:
        raise ValueError("No readable layers found.")
    return pd.concat(frames, ignore_index=True)

def kml_EL(path: str, layer: str | None = None):
    """
    Read a KML or KMZ. For KMZ, the first .kml inside is used via /vsizip/.
    If 'layer' is None, read all available layers and concatenate.
    """
    p = Path(path).resolve()

    # .kml straight
    if p.suffix.lower() == ".kml":
        layers = _list_layers(p.as_posix())
        if layer is None:
            return _concat_layers(p.as_posix(), layers)
        if layer not in layers:
            raise ValueError(f"Layer '{layer}' not found. Available: {layers}")
        return gpd.read_file(p.as_posix(), layer=layer)

    # .kmz: find inner .kml
    if p.suffix.lower() == ".kmz":
        with zipfile.ZipFile(p, "r") as zf:
            kml_inside = _first_member_with_suffix(zf, ".kml")
            if not kml_inside:
                raise FileNotFoundError("No .kml file found inside the KMZ.")
        vsip = _vsizip_path(p, kml_inside)
        layers = _list_layers(vsip)
        if layer is None:
            return _concat_layers(vsip, layers)
        if layer not in layers:
            raise ValueError(f"Layer '{layer}' not found. Available: {layers}")
        return gpd.read_file(vsip, layer=layer)

    raise ValueError("kml_EL expects a .kml or .kmz path.")


# ---------- NEW: GPX (multi-layer support) ----------

def gpx_EL(path: str, layer: str | None = None):
    """
    Read a GPX file. Common layers: 'waypoints', 'routes', 'tracks',
    'route_points', 'track_points'. If 'layer' is None, read a practical subset
    (tracks/routes/waypoints if present) or all layers as a fallback.
    Supports .zip containing a single .gpx via /vsizip/.
    """
    p = Path(path).resolve()

    def _read_from_pathlike(pathlike: str) -> gpd.GeoDataFrame:
        layers = _list_layers(pathlike)
        if layer is None:
            wanted = [l for l in ("tracks", "routes", "waypoints") if l in layers]
            if not wanted:
                wanted = layers
            return _concat_layers(pathlike, wanted)
        if layer not in layers:
            raise ValueError(f"Layer '{layer}' not found. Available: {layers}")
        return gpd.read_file(pathlike, layer=layer)

    if p.suffix.lower() == ".gpx":
        return _read_from_pathlike(p.as_posix())

    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p, "r") as zf:
            gpx_inside = _first_member_with_suffix(zf, ".gpx")
            if not gpx_inside:
                raise FileNotFoundError("No .gpx file found inside the zip.")
        vsip = _vsizip_path(p, gpx_inside)
        return _read_from_pathlike(vsip)

    raise ValueError("gpx_EL expects a .gpx or .zip path.")


# ---------- OPTIONAL: a unified zip geodata reader ----------

def zip_geodata_EL(zip_path: str):
    """
    Read geodata from a .zip by detecting the inner file type.
    Priority: Shapefile (.shp) > FlatGeobuf (.fgb) > KML (.kml) > GPX (.gpx).
    """
    p = Path(zip_path).resolve()
    if p.suffix.lower() != ".zip":
        raise ValueError("zip_geodata_EL expects a .zip path.")

    with zipfile.ZipFile(p, "r") as zf:
        # Try SHP
        inner = _first_member_with_suffix(zf, ".shp")
        if inner:
            return gpd.read_file(_vsizip_path(p, inner))
        # Try FGB
        inner = _first_member_with_suffix(zf, ".fgb")
        if inner:
            return gpd.read_file(_vsizip_path(p, inner))
        # Try KML (KMZ scenario)
        inner = _first_member_with_suffix(zf, ".kml")
        if inner:
            kml_vsip = _vsizip_path(p, inner)
            layers = _list_layers(kml_vsip)
            return _concat_layers(kml_vsip, layers)
        # Try GPX
        inner = _first_member_with_suffix(zf, ".gpx")
        if inner:
            gpx_vsip = _vsizip_path(p, inner)
            layers = _list_layers(gpx_vsip)
            # default subset for GPX
            wanted = [l for l in ("tracks", "routes", "waypoints") if l in layers] or layers
            return _concat_layers(gpx_vsip, wanted)

    raise FileNotFoundError("No supported geodata found inside the zip (.shp/.fgb/.kml/.gpx).")

# mapping: extension → handler function
dict_EL = {
    '.xlsx':    excel_EL,
    '.xls':     excel_EL,
    '.csv':     csv_EL,
    '.tsv':     csv_EL,
    '.parquet': parquet_EL,
    '.geojson': geojson_EL,
    '.json':    detect_json_type,
    '.shp':     shapefile_EL,      # direct shapefile path
    '.zip':     zip_geodata_EL,    # unified zip reader (SHP/FGB/KML/GPX)
    '.kml':     kml_EL,
    '.kmz':     kml_EL,
    '.fgb':     fgb_EL,
    '.gpx':     gpx_EL,
}

# mapping: extension → data category
dict_category = {
    '.xlsx':    "structured",
    '.xls':     "structured",
    '.csv':     "structured",
    '.tsv':     "structured",
    '.parquet': "structured",
    '.geojson': "semi-structured",
    '.json':    "semi-structured",
    '.shp':     "semi-structured",
    '.zip':     "semi-structured",
    '.kml':     "semi-structured",
    '.kmz':     "semi-structured",
    '.fgb':     "semi-structured",
    '.gpx':     "semi-structured",
}

def get_df(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in dict_EL:
        df = dict_EL[ext](path)
        return df, ext, dict_category[ext]
    else:
        return None, "unknown", "unknown"

def _resolve_excel_target_sheet(file_path: str, sheet) -> tuple[object, str]:
    """Return (target_sheet, engine) with header detection to be done later."""
    engine, _ = _engine_for_excel(file_path)
    with pd.ExcelFile(file_path, engine=engine) as xls:
        if sheet in (None, "first", "auto"):
            target = xls.sheet_names[0] if xls.sheet_names else 0
        elif isinstance(sheet, int):
            if not xls.sheet_names:
                raise ValueError("Workbook has no sheets.")
            idx = max(0, min(sheet, len(xls.sheet_names) - 1))
            target = xls.sheet_names[idx]
        else:
            target = sheet
    return target, engine

def _detect_excel_header(file_path: str, target_sheet, engine: str, preview_n: int = 300) -> int:
    """Detect header row on the target sheet using a small preview."""
    block = pd.read_excel(file_path, sheet_name=target_sheet, header=None, nrows=preview_n, engine=engine)
    block = block.dropna(how="all")
    if block.empty:
        return 0
    rc = block.count(axis=1)
    return int(rc.idxmax()) if not rc.empty else 0

def read_head_any(
        path: str,
        *,
        sheet: int | str | None = "auto",
        nrows: int = 20,
        columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Read first `nrows` rows for ANY supported format.
    If `columns` is provided, also limit to these columns.
    Excel is header-aware on the *target sheet*.
    Geo formats return attributes only (geometry dropped).
    """
    ext = Path(path).suffix.lower()

    # --- Excel family ---
    if ext in {".xlsx", ".xls", ".xlsm", ".xlsb", ".ods"}:
        target, engine = _resolve_excel_target_sheet(path, sheet)
        header = _detect_excel_header(path, target, engine, preview_n=max(50, nrows))
        return pd.read_excel(
            path, sheet_name=target, header=header, nrows=nrows, usecols=columns, engine=engine
        )

    # --- CSV/TSV with encoding & separator guess (using your helpers) ---
    if ext in {".csv", ".tsv"}:
        # if separator not obvious, guess from sample bytes
        encs = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
        seps = [None, ",", ";", "\t", "|"]
        for enc in encs:
            for sep in seps:
                try:
                    df = pd.read_csv(
                        path,
                        usecols=columns,
                        sep=sep,
                        encoding=enc,
                        low_memory=False,
                        on_bad_lines="skip"
                    )
                    if df.shape[1] > 1: return df
                except Exception:
                    pass
        raise ValueError(f"Could not parse {path} as CSV/TSV.")
        # pandas CSV usecols 同样支持名字或位置

    # --- Parquet ---
    if ext == ".parquet":
        # column pushdown is native; then head for rows
        df = pd.read_parquet(path, engine="pyarrow", columns=columns)
        return df.head(nrows)

    # --- JSON / NDJSON ---
    if ext == ".json":
        try:
            df = pd.read_json(path, lines=True, nrows=nrows)
        except ValueError:
            with open(path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            df = pd.json_normalize(data[:nrows])
        if columns:
            keep = [c for c in columns if c in df.columns]
            df = df[keep]
        return df

    # --- Geo formats (attributes only, fast) ---
    if ext in {".geojson", ".shp", ".kml", ".kmz", ".gpx", ".fgb", ".zip"}:
        try:
            from pyogrio import read_dataframe
            gdf = read_dataframe(path, read_geometry=False, columns=columns, max_features=nrows)
            return pd.DataFrame(gdf)
        except Exception:
            gdf = gpd.read_file(path)
            df = pd.DataFrame(gdf.drop(columns=[gdf.geometry.name], errors="ignore"))
            if columns:
                keep = [c for c in columns if c in df.columns]
                df = df[keep]
            return df.head(nrows)

    # --- Fallback: use existing get_df then trim ---
    df, _, _ = get_df(path)
    if columns:
        keep = [c for c in columns if c in df.columns]
        df = df[keep]
    return df.head(nrows)

def _map_pos_to_names(names: List[str], cols: List[ColT]) -> List[str]:
    """Map positional indices to column names; keep name strings as-is; de-dup in order."""
    out: List[str] = []
    for c in cols:
        if isinstance(c, int):
            if 0 <= c < len(names):
                out.append(names[c])
        else:
            out.append(str(c))
    seen = set()
    uniq = []
    for c in out:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def read_cols_full(
        path: str,
        *,
        sheet: int | str | None = "auto",
        columns: Optional[List[ColT]] = None,
) -> pd.DataFrame:
    """
    Read FULL number of rows but ONLY the given `columns` for ANY supported format.
    `columns` can be column NAMES and/or POSITIONAL indices (0-based).
    Excel: header auto-detected on the target sheet.
    CSV/TSV: encoding & separator guessed from a small byte sample.
    Parquet: positional indices mapped to schema names via pyarrow.
    JSON: load then select (no true column pushdown).
    Geo: attributes only (geometry dropped); use pyogrio when available.
    """
    if columns is not None and len(columns) == 0:
        return pd.DataFrame()

    ext = Path(path).suffix.lower()

    # --- Excel family ---
    if ext in {".xlsx", ".xls", ".xlsm", ".xlsb", ".ods"}:
        target, engine = _resolve_excel_target_sheet(path, sheet)
        header = _detect_excel_header(path, target, engine, preview_n=300)
        # pandas Excel usecols 支持名字或位置，混用也可以
        return pd.read_excel(
            path,
            sheet_name=target,
            header=header,
            usecols=columns,
            engine=engine
        )

    # --- CSV / TSV ---
    if ext in {".csv", ".tsv"}:
        encs = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
        seps = [None, ",", ";", "\t", "|"]
        for enc in encs:
            for sep in seps:
                try:
                    return pd.read_csv(
                        path,
                        usecols=columns,
                        sep=sep,
                        encoding=enc,
                        low_memory=False,
                        on_bad_lines="skip"
                    )
                except Exception:
                    pass
        raise ValueError(f"Could not parse {path} as CSV/TSV.")
        # pandas CSV usecols 同样支持名字或位置

    # --- Parquet ---
    if ext == ".parquet":
        if columns is None:
            return pd.read_parquet(path, engine="pyarrow")
        if any(isinstance(c, int) for c in columns):
            import pyarrow.parquet as pq
            schema_names = pq.ParquetFile(path).schema_arrow.names
            colnames = _map_pos_to_names(schema_names, list(columns))
            return pd.read_parquet(path, engine="pyarrow", columns=colnames)
        else:
            return pd.read_parquet(path, engine="pyarrow", columns=list(columns))

    # --- JSON / NDJSON ---
    if ext == ".json":
        try:
            df = pd.read_json(path, lines=True)
        except ValueError:
            with open(path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            df = pd.json_normalize(data)
        if columns is None:
            return df
        names = list(df.columns)
        keep = _map_pos_to_names(names, list(columns))
        keep = [c for c in keep if c in df.columns]
        return df[keep]

    # --- Geo formats (attributes only) ---
    if ext in {".geojson", ".shp", ".kml", ".kmz", ".gpx", ".fgb", ".zip"}:
        try:
            from pyogrio import read_dataframe
            if columns is None:
                gdf = read_dataframe(path, read_geometry=False)
                return pd.DataFrame(gdf)
            if any(isinstance(c, int) for c in columns):
                probe = read_dataframe(path, read_geometry=False, max_features=1)
                names = list(pd.DataFrame(probe).columns)
                sel = _map_pos_to_names(names, list(columns))
                gdf = read_dataframe(path, read_geometry=False, columns=sel)
                return pd.DataFrame(gdf)
            else:
                gdf = read_dataframe(path, read_geometry=False, columns=list(columns))
                return pd.DataFrame(gdf)
        except Exception:
            gdf = gpd.read_file(path)
            df = pd.DataFrame(gdf.drop(columns=[gdf.geometry.name], errors="ignore"))
            if columns is None:
                return df
            names = list(df.columns)
            sel = _map_pos_to_names(names, list(columns))
            keep = [c for c in sel if c in df.columns]
            return df[keep]

    # --- Fallback: use existing get_df then select ---
    df, _, _ = get_df(path)
    if columns is None:
        return df
    names = list(df.columns)
    sel = _map_pos_to_names(names, list(columns))
    keep = [c for c in sel if c in df.columns]
    return df[keep]
# =================== end unified two-entry readers (add-on) ===================

def read_head_with_meta(
    path: str,
    *,
    sheet: int | str | None = "auto",
    nrows: int = 5,
    columns: list[str] | None = None,
):
    """
    Read first nrows (optionally limited to columns) AND return ext/category.
    Returns: (df, ext, category)
    """
    df = read_head_any(path, sheet=sheet, nrows=nrows, columns=columns)
    ext = os.path.splitext(path)[1].lower()
    cat = dict_category.get(ext, "unknown")
    return df, ext, cat


def level_matches(level, granularities) -> bool:
    """Return True if `level` is present in `granularities`.
    If `level` is a tuple of aliases (e.g. ("geometry", "wkt_geojson")),
    match if ANY alias appears in `granularities`.
    """
    if isinstance(level, tuple):
        return any(alias in granularities for alias in level)
    return level in granularities


def level_name(level) -> str:
    """Normalise a level to its canonical name.
    For tuples, use the first alias as the canonical name.
    """
    return level[0] if isinstance(level, tuple) else level


def get_most_general_in_path(granularities, path) -> Union[str, None]:
    """Scan a single hierarchy path from most general to most specific.
    Return the first (i.e., most general) level that appears in `granularities`.
    If none match, return None.
    """
    for level in path:
        if level_matches(level, granularities):
            return level_name(level)
    return None

def get_most_specific_in_path(granularities, path) -> Union[str, None]:
    """Scan a single hierarchy path from most specific to most general.
    Return the first (i.e., most specific) level that appears in `granularities`.
    If none match, return None.
    """
    # Iterate in reverse so we check the most specific level first
    for level in reversed(path):
        if level_matches(level, granularities):
            return level_name(level)
    return None

def human_readable_size(num_bytes: Optional[int]) -> Optional[str]:
    """Convert bytes to a human-readable string. Return None if input is None."""
    if num_bytes is None:
        return None
    size = float(num_bytes)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"

def uncompressed_zip_size(path: str) -> Optional[int]:
    """Return total uncompressed size of a ZIP or None on failure/not a zip."""
    if not (os.path.exists(path) and path.lower().endswith(".zip")):
        return None
    try:
        with zipfile.ZipFile(path, 'r') as zf:
            return sum(info.file_size for info in zf.infolist())
    except Exception:
        return None