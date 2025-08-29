import time
from uml_class import *
from scope_detector import *
from granularity_detector import *          # should provide extract_label_ranges
from theme_detector import find_min_common_theme
from semantic_helper import *
from attribute_classifier import classify_attributes_with_semantic_helper
from reference import HIER


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
    if isinstance(cols, str):
        return set(df[cols].dropna().astype(str))
    elif isinstance(cols, (list, tuple)):
        if len(cols) == 0:
            return set()
        if len(cols) == 1:
            return set(df[cols[0]].dropna().astype(str))
        sub = df[list(cols)].dropna()
        return set(map(tuple, sub.astype(str).itertuples(index=False, name=None)))
    else:
        return set()


def get_dataset_scopes_gras(df, atts_spatial, atts_temporal):
    """Compute dataset scopes and granularities for spatial and temporal dimensions."""
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
            values = _collect_values(df, att["columns"])
            spatial_scope.append(DS_Spatial_Scope(gra, values))

    # Temporal parameters → DS_Temporal_Scope(level, ranges)
    for att in atts_temporal:
        gra = att.get("granularity")
        if gra in temporal_scope_level:
            tokens = _collect_values(df, att["columns"])
            # IMPORTANT: extract_label_ranges expects (tokens, granularity)
            ranges = extract_label_ranges(gra, tokens)
            temporal_scope.append(DS_Temporal_Scope(gra, ranges))

    return spatial_scope, temporal_scope, spatial_granularity, temporal_granularity


def construct_dataset(path: str, measure: bool = False):
    """
    End-to-end dataset construction from a file path.
    If measure=True, returns (dataset, timings).
    Otherwise returns dataset.
    """
    timings = {}
    t0_total = time.perf_counter()

    title = os.path.basename(path)

    # --- read data ---
    t0 = time.perf_counter()
    df, ext, data_format = get_df(path)
    timings["read_df_sec"] = time.perf_counter() - t0

    # --- semantic helper on sample ---
    samples = df.head(10)
    t0 = time.perf_counter()
    semantic_res = semantic_helper(samples)
    timings["semantic_helper_sec"] = time.perf_counter() - t0

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
    theme_path = find_min_common_theme(atts_theme)
    theme_obj = Theme(theme_path, theme_path) if theme_path else None
    timings["find_common_theme_sec"] = time.perf_counter() - t0

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
    t0 = time.perf_counter()
    dataset = transform_result(dataset, atts_spatial, atts_temporal, atts_indicator, atts_other)
    timings["transform_result_sec"] = time.perf_counter() - t0

    timings["total_sec"] = time.perf_counter() - t0_total

    return (dataset, timings) if measure else dataset
