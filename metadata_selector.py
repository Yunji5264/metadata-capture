
import time
from uml_class import *
from scope_detector import *
from granularity_detector import *          # should provide extract_label_ranges
from theme_detector import find_min_common_theme, collect_all_themes_set
from semantic_helper import *
from attribute_classifier import classify_attributes_with_semantic_helper, find_geometry_columns
from reference import HIER
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

def _collect_geoms_from_geometry_column(df: pd.DataFrame, col: str) -> List[BaseGeometry]:
    """Collect shapely geometries from a 'geometry-like' column."""
    s = df[col]
    # If df is a GeoDataFrame and `col` is the active geometry, values are shapely
    # If it's a plain DataFrame, values may already be shapely or None
    return [g for g in s if g is not None]

def _collect_points_from_latlon(df: pd.DataFrame, lat_col: str, lon_col: str) -> List[Point]:
    """Build shapely Points from numeric lat/lon columns."""
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")
    pts = []
    for x, y in zip(lon, lat):
        if pd.notna(x) and pd.notna(y):
            pts.append(Point(float(x), float(y)))
    return pts

def _aggregate_geometry(
    geoms: Iterable[BaseGeometry],
    *,
    method: str = "envelope",
    buffer_m: float = 0.0
) -> Optional[BaseGeometry]:
    """
    Aggregate many geometries into a single extent geometry.
    method:
      - 'envelope': bounding rectangle of the union
      - 'convex_hull': convex hull of the union
      - 'union': full union (can be expensive/complex)
    buffer_m: optional buffer (in CRS units) after aggregation
    """
    geoms = [g for g in geoms if g is not None]
    if not geoms:
        return None

    merged = unary_union(geoms)
    if method == "envelope":
        agg = merged.envelope
    elif method == "convex_hull":
        agg = merged.convex_hull
    elif method == "union":
        agg = merged
    else:
        agg = merged.envelope

    if buffer_m and buffer_m > 0:
        try:
            agg = agg.buffer(buffer_m)
        except Exception:
            pass
    return agg

def _geometry_values_as_wkt(geom: Optional[BaseGeometry]) -> List[str]:
    """Return the aggregated geometry as a WKT list (DS_Spatial_Scope expects 'values')."""
    return [geom.wkt] if geom is not None else []


def transform_result(dataset: Dataset, atts_spatial, atts_temporal, atts_indicator, atts_other) -> Dataset:
    """
    Map classifier results into Dataset attributes.

    Now: themeName keeps the FULL path, normalised with ' > ' between levels.
    Previous behaviour (leaf-only) is replaced.
    Other rules remain:
      - Preserve multi-column features as a list in 'dataName' (do NOT join).
      - Single-column features remain a string.
      - If no theme description is provided, keep the original (raw) name/path in themeDescription.
    """

    def _name_preserve_list(cols: Union[str, List[str], None]) -> Union[str, List[str]]:
        """Return 'columns' as-is: list stays list, string stays string; None -> empty string."""
        if cols is None:
            return ""
        if isinstance(cols, list):
            return [str(c) for c in cols]
        return str(cols)

    def _dtype_from_type(tp: Any) -> Union[str, List[str]]:
        """Extract a dtype string/list from 'type' which may be a list or a single value."""
        if isinstance(tp, list) and tp:
            return [str(t) for t in tp]
        return str(tp) if tp is not None else "object"

    def _text(val: Any, default: str = "") -> str:
        """Coerce any value to string, with default for None."""
        return str(val) if val is not None else default

    def _granularity(entry: Dict[str, Any], default: Optional[str] = None) -> Optional[str]:
        """Read 'granularity' from entry with an optional default."""
        g = entry.get("granularity", default)
        return str(g) if g is not None else None

    def _indicator_type(entry: Dict[str, Any]) -> Optional[str]:
        """Accept both 'indicatorType' and 'indicator_type'."""
        it = entry.get("indicatorType", entry.get("indicator_type"))
        return str(it) if it is not None else None

    def _normalise_path(name: str) -> str:
        """
        Normalise a hierarchical path string to 'A > B > C'.
        - Supports common separators: '>', '/', '\\', '→', '»'
        - Trims whitespace around tokens
        - Removes empty tokens
        """
        parts = re.split(r'>|/|\\|→|»', str(name))
        tokens = [p.strip() for p in parts if p and p.strip()]
        return " > ".join(tokens) if tokens else str(name).strip()

    def _theme(entry: Dict[str, Any]) -> Optional[Theme]:
        """
        Build a Theme with FULL normalised path as themeName.
        - If 'theme' is a Theme: copy with normalised full path (keep existing description).
        - If dict: use themeName/themeDescription keys when present; otherwise best-effort.
        - If string: normalised full path becomes themeName; the raw string goes to description.
        """
        th = entry.get("theme")
        if th is None:
            return None

        # Already a Theme instance
        if isinstance(th, Theme):
            normalised = _normalise_path(th.themeName)
            return Theme(normalised, th.themeDescription)

        # Dict payload
        if isinstance(th, dict):
            name = th.get("themeName") or th.get("name") or th.get("theme_name") or th.get("title")
            desc = th.get("themeDescription") or th.get("description") or th.get("desc") or th.get("theme_description")
            if isinstance(name, str):
                normalised = _normalise_path(name)
                final_desc = desc if desc is not None else str(name)  # keep raw if no desc
                return Theme(normalised, str(final_desc))
            return Theme("Theme", str(desc) if desc is not None else "")

        # Plain string path
        if isinstance(th, str):
            normalised = _normalise_path(th)
            return Theme(normalised, th)  # keep raw in description

        # Unsupported type
        return None

    # Spatial parameters
    for att in atts_spatial:
        name = _name_preserve_list(att.get("columns"))
        desc = _text(att.get("description"))
        dtype = _dtype_from_type(att.get("type"))
        level = _granularity(att, default="geocode")
        dataset.add_spatial_parameter(name, desc, dtype, level)

    # Temporal parameters
    for att in atts_temporal:
        name = _name_preserve_list(att.get("columns"))
        desc = _text(att.get("description"))
        dtype = _dtype_from_type(att.get("type"))
        level = _granularity(att, default="unknown")
        dataset.add_temporal_parameter(name, desc, dtype, level)

    # Existing indicators
    for att in atts_indicator:
        name = _name_preserve_list(att.get("columns"))
        desc = _text(att.get("description"))
        dtype = _dtype_from_type(att.get("type"))
        ind_type = _indicator_type(att)
        theme = _theme(att)
        dataset.add_existing_indicator(name, desc, dtype, ind_type, theme)

    # Other → Complementary_Information
    for att in atts_other:
        name = _name_preserve_list(att.get("columns"))
        desc = _text(att.get("description"))
        dtype = _dtype_from_type(att.get("type"))
        gran = _granularity(att)  # may be None
        theme = _theme(att)
        dataset.add_complementary_information(name, desc, dtype, gran, theme)

    return dataset


def _collect_values(df, cols):
    """Return a set of values for a column spec.
    - If cols is a string: return unique scalar values as strings.
    - If cols is a list/tuple:
        - len == 1: same as single column
        - len > 1: return unique row-wise tuples (converted to strings)
    """
    import pandas as pd

    def normalize_val(v):
        try:
            f = float(v)
            if f.is_integer():
                return str(int(f))
            return str(f)
        except Exception:
            return str(v)

    if isinstance(cols, str):
        return set(df[cols].dropna().map(normalize_val))
    elif isinstance(cols, (list, tuple)):
        if len(cols) == 0:
            return set()
        if len(cols) == 1:
            return set(df[cols[0]].dropna().map(normalize_val))
        sub = df[list(cols)].dropna()
        return set(
            map(
                tuple,
                sub.applymap(normalize_val).itertuples(index=False, name=None)
            )
        )
    else:
        return set()


from typing import List
from shapely.geometry.base import BaseGeometry

def get_dataset_scopes_gras(df, atts_spatial, atts_temporal):
    """
    Compute dataset scopes and granularities for spatial and temporal dimensions.
    """

    # Gather granularities present
    spatial_gras = [att["granularity"] for att in atts_spatial if att.get("granularity")]
    temporal_gras = [att["granularity"] for att in atts_temporal if att.get("granularity")]

    # Determine scope levels and final granularity per hierarchy
    spatial_scope_level = get_scope(spatial_gras, HIER["spatial"])
    temporal_scope_level = get_scope(temporal_gras, HIER["temporal"])
    spatial_granularity = get_granularity(spatial_gras, HIER["spatial"])
    temporal_granularity = get_granularity(temporal_gras, HIER["temporal"])

    spatial_scope = []
    temporal_scope = []

    # Spatial parameters → DS_Spatial_Scope(level, values)
    for att in atts_spatial:
        gra = att.get("granularity")

        if gra in spatial_scope_level:
            # Case A: normal label-like scopes (e.g., country/region/city names)
            if gra not in {"geometry", "geopoint", "latlon_pair"}:
                values = list(_collect_values(df, att["columns"]))
                spatial_scope.append(DS_Spatial_Scope(gra, values))
                continue

            # Case B: geometry or geopoint → build an aggregated geometry scope
            cols = att["columns"]
            geoms: List[BaseGeometry] = []

            # 1) If an explicit geometry column is present
            # Example: columns == ["geometry"] or ["geom"]
            try:
                # Ensure we only pick columns present in df
                present_cols = [
                    c for c in (cols if isinstance(cols, list) else [cols])
                    if c in df.columns
                ]
                # Prefer geometry-like column if available
                geom_col = None
                for c in present_cols:
                    sample = df[c].dropna().head(1)
                    if not sample.empty and hasattr(sample.iloc[0], "geom_type"):
                        geom_col = c
                        break
                if geom_col:
                    geoms = _collect_geoms_from_geometry_column(df, geom_col)
            except Exception:
                pass

            # 2) If it's a lat/lon pair (e.g., columns == ["lat","lon"])
            if not geoms:
                if isinstance(cols, list) and len(cols) == 2:
                    la, lo = cols[0], cols[1]
                    if la in df.columns and lo in df.columns:
                        # Heuristic: swap if names look inverted
                        la_l, lo_l = la.lower(), lo.lower()
                        if any(k in la_l for k in ("lon", "lng", "x")) and \
                           any(k in lo_l for k in ("lat", "y")):
                            la, lo = lo, la
                        try:
                            geoms = _collect_points_from_latlon(df, la, lo)
                            gra = "geopoint"  # normalise naming
                        except Exception:
                            pass

            # 3) Aggregate into a single extent geometry
            agg = _aggregate_geometry(geoms, method="union", buffer_m=0.0)
            values = _geometry_values_as_wkt(agg)

            # Emit a new scope entry that represents the geometry extent
            spatial_scope.append(DS_Spatial_Scope("geometry_extent", values))

    # Temporal parameters → DS_Temporal_Scope(level, ranges)
    for att in atts_temporal:
        gra = att.get("granularity")
        if gra in temporal_scope_level:
            tokens = _collect_values(df, att["columns"])
            # extract_label_ranges expects (tokens, granularity)
            ranges = extract_label_ranges(gra, tokens)
            temporal_scope.append(DS_Temporal_Scope(gra, ranges))

    return spatial_scope, temporal_scope, spatial_granularity, temporal_granularity


def construct_dataset(path: str, measure: bool = True):
    """
    End-to-end dataset construction from a file path.
    If measure=True:
      - When semantic cache MISS: return (dataset, timings) with detailed timings.
      - When semantic cache HIT : return (dataset, timings) with minimal flags and no timings.
    If measure=False: return dataset only.
    """
    title = os.path.basename(path)
    semantic_cache_path = os.path.normpath(os.path.join("..", "ref", "ref_semantic", f"{title}.json"))
    cache_hit = os.path.exists(semantic_cache_path)

    perf_cache_path = None
    sec_timing = None

    if cache_hit:
        perf_cache_path = os.path.normpath(os.path.join("..", "metadata", "perf", f"{title}.perf.json"))

    if perf_cache_path and os.path.exists(perf_cache_path):
        with open(perf_cache_path, "r", encoding="utf-8") as f:
            perf_obj = json.load(f)
        timings_obj = perf_obj.get("timings") or {}
        sec_timing = timings_obj.get("semantic_helper_sec")

    timings = {}

    t0_total = time.perf_counter()

    # --- read data ---

    t0 = time.perf_counter()
    df, ext, data_format = get_df(path)
    timings["read_df_sec"] = time.perf_counter() - t0

    # --- semantic helper on sample (with cache) ---
    if cache_hit and sec_timing:
        # Use cached semantic and recorded semantic helper timing
        semantic_res = pd.read_json(semantic_cache_path, orient="records")
        timings["semantic_helper_sec"] = sec_timing
    else:
        samples = df.head(5)
        geom_cols = find_geometry_columns(samples)[0]

        if geom_cols:
            samples_for_sem = samples.drop(columns=geom_cols)
        else:
            samples_for_sem = samples

        t0 = time.perf_counter()

        semantic_res = semantic_helper(samples_for_sem)
        timings["semantic_helper_sec"] = time.perf_counter() - t0
        # Save semantic cache for next runs
        os.makedirs(os.path.dirname(semantic_cache_path), exist_ok=True)
        semantic_res.to_json(semantic_cache_path, orient="records", force_ascii=False, indent=2)

    # --- classify attributes ---
    t0 = time.perf_counter()
    results = classify_attributes_with_semantic_helper(df, semantic_res)
    timings["classify_attributes_sec"] = time.perf_counter() - t0

    # --- buckets ---
    atts_spatial  = results.get("spatial", []) or []
    atts_temporal = results.get("temporal", []) or []
    atts_indicator = results.get("indicators", results.get("indicator", [])) or []
    atts_other    = results.get("other", []) or []
    atts_theme = atts_indicator + atts_other


    # --- scopes & granularities ---
    t0 = time.perf_counter()
    ss, ts, sg, tg = get_dataset_scopes_gras(df, atts_spatial, atts_temporal)
    timings["scopes_granularities_sec"] = time.perf_counter() - t0

    # --- common theme ---
    t0 = time.perf_counter()
    # theme_path = find_min_common_theme(atts_theme)
    all_themes = collect_all_themes_set(atts_theme)
    themes = (Theme(t,t) for t in all_themes)
    # theme_obj = theme_path if theme_path else None
    theme_obj = themes if themes else None
    timings["find_common_theme_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    # --- file size / dimensions ---
    file_size_bytes = os.path.getsize(path) if os.path.exists(path) else None
    file_size_human = human_readable_size(file_size_bytes)
    n_rows, n_cols = (df.shape if df is not None else (None, None))
    n_records = int(len(df)) if df is not None else None
    n_features = n_records if ext in {".geojson", ".zip"} else None
    unzipped_size = uncompressed_zip_size(path) if ext == ".zip" else None

    # --- build dataset skeleton ---
    dataset = Dataset(
        title=title,
        description="",
        dataFormat=data_format,
        fileType=ext,
        updateFrequency="",
        sourceName=path,
        sourceType="local",
        sourceAddress=path,
        spatialGranularity=sg,
        spatialScope=ss,
        temporalGranularity=tg,
        temporalScope=ts,
        theme=theme_obj,
        attributes=[],
        fileSizeBytes=file_size_bytes,
        fileSizeHuman=file_size_human,
        nRows=n_rows,
        nCols=n_cols,
        nRecords=n_records,
        nFeatures=n_features,
        uncompressedSizeBytes=unzipped_size
    )

    # --- fill attributes ---
    dataset = transform_result(dataset, atts_spatial, atts_temporal, atts_indicator, atts_other)
    timings["transform_result_sec"] = time.perf_counter() - t0
    timings["total_sec"] = time.perf_counter() - t0_total

    return (dataset, timings) if measure else dataset


def construct_dataset_new(
    path: str,
    *,
    sheet: int | str | None = "auto",
    sample_rows: int = 10,
    measure: bool = True,
    semantic_cache_dir: str = os.path.join("..", "ref", "ref_semantic"),
    perf_cache_dir: str = os.path.join("..", "metadata", "perf"),
):
    """
    Two-phase constructor:
      1) Read a small head sample to compute semantic + attribute classification.
      2) Read only scope-related columns (full rows) to compute scopes & granularities.

    Returns
    -------
    dataset : dict
        Final dataset metadata dict (same as construct_dataset 原版).
    timings : dict
        Only returned if measure=True.
    """
    title = os.path.basename(path)

    semantic_cache_path = os.path.normpath(
        os.path.join(semantic_cache_dir, f"{title}.json")
    )
    cache_hit = os.path.exists(semantic_cache_path)

    perf_cache_path = os.path.normpath(
        os.path.join(perf_cache_dir, f"{title}.perf.json")
    )
    sec_timing: Optional[float] = None
    if cache_hit and os.path.exists(perf_cache_path):
        try:
            with open(perf_cache_path, "r", encoding="utf-8") as f:
                prev_perf = json.load(f)
            sec_timing = (prev_perf.get("timings") or {}).get("semantic_helper_sec")
        except Exception:
            sec_timing = None

    timings: Dict[str, float] = {}
    t0_total = time.perf_counter()

    # --- Phase 1: read head sample ---
    t0 = time.perf_counter()
    df_head = read_head_any(path, sheet=sheet, nrows=sample_rows)
    timings["read_head_sec"] = time.perf_counter() - t0

    # --- Semantic helper (with cache) ---
    if cache_hit and sec_timing:
        semantic_res = pd.read_json(semantic_cache_path, orient="records")
        timings["semantic_helper_sec"] = float(sec_timing)
    else:
        geom_cols, _ = find_geometry_columns(df_head)
        sample_for_sem = df_head.drop(columns=geom_cols, errors="ignore") if geom_cols else df_head
        t0 = time.perf_counter()
        semantic_res = semantic_helper(sample_for_sem)
        timings["semantic_helper_sec"] = time.perf_counter() - t0
        try:
            os.makedirs(os.path.dirname(semantic_cache_path), exist_ok=True)
            semantic_res.to_json(
                semantic_cache_path, orient="records", force_ascii=False, indent=2
            )
        except Exception:
            pass

    # --- Attribute classification ---
    t0 = time.perf_counter()
    att_results = classify_attributes_with_semantic_helper(
        df_head,
        semantic_res,
        filename=title,
        sheet_names=None,
    )
    timings["classify_attributes_sec"] = time.perf_counter() - t0

    atts_spatial = att_results.get("spatial", []) or []
    atts_temporal = att_results.get("temporal", []) or []

    # --- Phase 2: scopes & granularities (full rows, needed cols only) ---
    t0 = time.perf_counter()
    ss, ts, sg, tg, n_rows = get_dataset_scopes_gras_new(
        path, atts_spatial, atts_temporal, sheet=sheet
    )
    timings["scopes_granularities_sec"] = time.perf_counter() - t0

    timings["total_sec"] = time.perf_counter() - t0_total

    # --- Build dataset dict (与原版 construct_dataset 保持一致结构) ---
    dataset = {
        "title": title,
        "path": path,
        "attributes": att_results,
        "scopes": {
            "spatial": [s.to_dict() if hasattr(s, "to_dict") else s for s in ss],
            "temporal": [t.to_dict() if hasattr(t, "to_dict") else t for t in ts],
        },
        "granularity": {"spatial": sg, "temporal": tg},
        "n_rows": n_rows,
    }

    return (dataset, timings) if measure else dataset


def get_dataset_scopes_gras_new(
    path: str,
    atts_spatial: List[Dict[str, Any]],
    atts_temporal: List[Dict[str, Any]],
    *,
    sheet: int | str | None = "auto",
) -> Tuple[List[Any], List[Any], Optional[str], Optional[str], Optional[int]]:
    """
    Compute dataset scopes and granularities using FULL rows but ONLY the needed columns.

    Strategy
    --------
    1) Collect the union of columns used by spatial/temporal attributes.
    2) Read those columns fully via `read_cols_full`.
    3) Build spatial & temporal scopes and infer final granularities.

    Returns
    -------
    spatial_scopes : list[DS_Spatial_Scope]
    temporal_scopes : list[DS_Temporal_Scope]
    spatial_granularity : str | None
    temporal_granularity : str | None
    n_rows : int | None
    """
    def _as_list(x) -> List[str]:
        if x is None:
            return []
        if isinstance(x, list):
            return [str(v) for v in x]
        return [str(x)]

    # 0) Gather needed columns
    needed_cols: List[str] = []
    for att in (atts_spatial or []):
        needed_cols.extend(_as_list(att.get("columns")))
    for att in (atts_temporal or []):
        needed_cols.extend(_as_list(att.get("columns")))
    needed_cols = list(dict.fromkeys(needed_cols))  # de-dup, keep order

    # 1) Read only needed columns (full rows)
    if needed_cols:
        df_scope = read_cols_full(path, sheet=sheet, columns=needed_cols)
    else:
        df_scope = pd.DataFrame()
    n_rows = int(len(df_scope)) if df_scope is not None else None

    # 2) Determine presence of granularities from attribute outputs
    spatial_gras = [att.get("granularity") for att in (atts_spatial or []) if att.get("granularity")]
    temporal_gras = [att.get("granularity") for att in (atts_temporal or []) if att.get("granularity")]

    spatial_scope_level = get_scope(spatial_gras, HIER["spatial"])
    temporal_scope_level = get_scope(temporal_gras, HIER["temporal"])
    spatial_granularity = get_granularity(spatial_gras, HIER["spatial"])
    temporal_granularity = get_granularity(temporal_gras, HIER["temporal"])

    spatial_scopes: List[Any] = []
    temporal_scopes: List[Any] = []

    # 3) Build spatial scopes
    for att in (atts_spatial or []):
        gra = att.get("granularity")
        cols = _as_list(att.get("columns"))
        if not gra or gra not in spatial_scope_level:
            continue

        # A) Label-like scopes
        if gra not in {"geometry", "geopoint", "latlon_pair"}:
            values = _collect_values(df_scope, cols)
            spatial_scopes.append(DS_Spatial_Scope(gra, values))
            continue

        # B) Geometry/geopoint scopes
        present_cols = [c for c in cols if c in df_scope.columns]
        geoms = []

        # 1) Explicit geometry-like column
        try:
            geom_col = None
            for c in present_cols:
                sample = df_scope[c].dropna().head(1)
                if not sample.empty and hasattr(sample.iloc[0], "geom_type"):
                    geom_col = c
                    break
            if geom_col:
                geoms = _collect_geoms_from_geometry_column(df_scope, geom_col)
        except Exception:
            pass

        # 2) Lat/lon pair
        if not geoms and len(present_cols) == 2:
            la, lo = present_cols[0], present_cols[1]
            la_l, lo_l = la.lower(), lo.lower()
            # Heuristic swap if the first looks like lon/X and second like lat/Y
            if any(k in la_l for k in ("lon", "lng", "x")) and any(k in lo_l for k in ("lat", "y")):
                la, lo = lo, la
            try:
                geoms = _collect_points_from_latlon(df_scope, la, lo)
                gra = "geopoint"  # normalize
            except Exception:
                pass

        # 3) Aggregate to extent and emit as scope
        agg = _aggregate_geometry(geoms, method="union", buffer_m=0.0)
        values = _geometry_values_as_wkt(agg)
        spatial_scopes.append(DS_Spatial_Scope("geometry_extent", values))

    # 4) Build temporal scopes
    for att in (atts_temporal or []):
        gra = att.get("granularity")
        cols = _as_list(att.get("columns"))
        if not gra or gra not in temporal_scope_level:
            continue
        tokens = _collect_values(df_scope, cols)
        ranges = extract_label_ranges(gra, tokens)
        temporal_scopes.append(DS_Temporal_Scope(gra, ranges))

    return spatial_scopes, temporal_scopes, spatial_granularity, temporal_granularity, n_rows

