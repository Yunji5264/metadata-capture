from spatial_detector import *
from temporal_detector import *
from indicator_detector import *
from reference import *

def classify_attributes(
    df: pd.DataFrame,
    *,
    min_ratio: float = 0.7,
    sample_size: int = 300,
    include_city_zip_in_address: bool = False,
    address_colname: str = "__address__",
    filename: str | None = None,
    sheet_names: List[str] | None = None,
    require_name_hint_for_geoformats: bool = True,
) -> Dict[str, Any]:
    """
    Heuristic + reference-based classifier (no semantic_res required).
    All entries consistently use "columns": [...] (never "column").
    """

    def _to_columns(entry: Dict[str, Any], name: str) -> List[str]:
        """Ensure a list-of-string for the 'columns' field."""
        if "columns" in entry and entry["columns"] is not None:
            cols = entry["columns"]
            if isinstance(cols, list):
                return [str(c) for c in cols]
            return [str(cols)]
        if "column" in entry and entry["column"] is not None:
            return [str(entry["column"])]
        return [str(name)]

    results = {
        "spatial": [],
        "temporal": [],
        "unknown": [],
        "indicators": [],
        "other": [],
        "meta": {}
    }

    # 0) Address aggregation
    df2, addr_cols = add_combined_address_column(
        df, colname=address_colname, include_city_zip=include_city_zip_in_address
    )
    results["meta"]["address_columns_used"] = addr_cols
    results["meta"]["config"] = {
        "min_ratio": min_ratio,
        "sample_size": sample_size,
        "include_city_zip_in_address": include_city_zip_in_address,
        "address_colname": address_colname,
        "filename": filename,
        "sheet_names": sheet_names or [],
        "require_name_hint_for_geoformats": require_name_hint_for_geoformats,
    }

    # 1) Lat/Lon pair
    consumed: set = set()
    latlon = detect_latlon_pair(df2)
    if latlon:
        la, lo = latlon
        consumed.update([la, lo])
        try:
            df2[la] = pd.to_numeric(df2[la], errors="coerce")
            df2[lo] = pd.to_numeric(df2[lo], errors="coerce")
        except Exception:
            pass

        results["spatial"].append({
            "columns": [la, lo],
            "description": "latlon pair geopoint",
            "type": [str(df2[la].dtype), str(df2[lo].dtype)],
            "granularity": "latlon_pair",
            "confidence": 0.98,
            "evidence": "Both columns numeric and within valid lat/lon ranges."
        })

    # 2) Reference sets
    ref_sets = build_ref_sets(ref_dict) if ref_dict else {}

    # 3) Column-wise classification
    for col in df2.columns:
        if col in consumed:
            continue

        s = df2[col]
        fmt_hints = geoformat_hints_from_colname(col) or {}

        # Geometry objects
        if detect_geometry_object(s):
            conf = 0.99 if fmt_hints.get("geometry") else 0.98
            evd = "Objects/dtype appear to be geometry." + (" Column name suggests geometry." if fmt_hints.get("geometry") else "")
            results["spatial"].append({
                "columns": col,
                "description": "geometry",
                "type": str(s.dtype),
                "granularity": "geometry",
                "confidence": conf,
                "evidence": evd
            })
            continue

        # WKT / GeoJSON strings
        if detect_wkt_geojson_string(s):
            name_gate_ok = fmt_hints.get("wkt") or fmt_hints.get("geojson") or fmt_hints.get("geometry")
            if (not require_name_hint_for_geoformats) or name_gate_ok:
                conf = 0.96 if name_gate_ok else 0.93
                evd = "Values match WKT/GeoJSON textual patterns." + (" Column name suggests WKT/GeoJSON." if name_gate_ok else "")
                results["spatial"].append({
                    "columns": col,
                    "description": "geometry",
                    "type": str(s.dtype),
                    "granularity": "geometry",
                    "confidence": conf,
                    "evidence": evd
                })
                continue

        # Address
        if col == address_colname and detect_address(s):
            results["spatial"].append({
                "columns": addr_cols,
                "description": "complete address",
                "type": str(s.dtype),
                "granularity": "address",
                "confidence": 0.8,
                "evidence": "Aggregated address lines with street tokens and numbers."
            })
            continue
        if not addr_cols and detect_address(s):
            results["spatial"].append({
                "columns": col,
                "description": "complete address",
                "type": str(s.dtype),
                "granularity": "address",
                "confidence": 0.75,
                "evidence": "Address-like strings detected."
            })
            continue

        # Reference matching (gated)
        gate = spatial_gate_from_colname(col) or {"has_gate": False, "levels": [], "generic": False}
        if ref_sets and not_null_ratio(s) > 0.2 and gate.get("has_gate", False):
            if gate.get("levels"):
                filtered = {lvl: ref_sets[lvl] for lvl in gate["levels"] if lvl in ref_sets}
                if filtered:
                    best = match_series_to_ref_levels(s, filtered, sample_size=sample_size)
                    if best and best["ratio"] >= min_ratio:
                        results["spatial"].append({
                            "columns": col,
                            "description": f"geographic attribute hinted by name ({', '.join(gate['levels'])})",
                            "type": str(s.dtype),
                            "granularity": best["level"],
                            "matched_by": best["by"],
                            "confidence": min(0.99, 0.7 + 0.3 * best["ratio"]),
                            "evidence": f"Name gate [{', '.join(gate['levels'])}] → ref match by {best['by']} ({round(best['ratio'] * 100)}%)."
                        })
                        continue
                    elif best:
                        results["other"].append({
                            "columns": col,
                            "description": f"geographic attribute hinted by name ({', '.join(gate['levels'])})",
                            "type": str(s.dtype),
                            "granularity": best["level"],
                            "matched_by": best["by"]
                        })
                        continue

            elif gate.get("generic", False):
                best = match_series_to_ref_levels(s, ref_sets, sample_size=sample_size) if ref_sets else None
                generic_min = max(min_ratio, 0.80)
                if best and best["ratio"] >= generic_min:
                    results["spatial"].append({
                        "columns": col,
                        "description": "generic geographic attribute (name-based gate)",
                        "type": str(s.dtype),
                        "granularity": best["level"],
                        "matched_by": best["by"],
                        "confidence": min(0.99, 0.68 + 0.32 * best["ratio"]),
                        "evidence": f"Generic geo gate → ref match by {best['by']} ({round(best['ratio'] * 100)}%)."
                    })
                    continue
                elif best:
                    results["other"].append({
                        "columns": col,
                        "description": "generic geographic attribute (name-based gate)",
                        "type": str(s.dtype),
                        "granularity": best["level"],
                        "matched_by": best["by"]
                    })
                    continue

        # Temporal granularity
        gran, conf = detect_temporal_granularity(s)
        if gran:
            results["temporal"].append({
                "columns": col,
                "description": "temporal attribute",
                "type": str(s.dtype),
                "granularity": gran,
                "confidence": conf
            })
            continue

        # Generic geocode fallback
        nunique_ratio = s.nunique(dropna=True) / max(1, len(s))
        if nunique_ratio > 0.9 and (is_string_series(s) or is_numeric_series(s)):
            results["spatial"].append({
                "columns": col,
                "description": "high-uniqueness identifier (treated as geocode)",
                "type": str(s.dtype),
                "granularity": "geocode",
                "confidence": 0.6,
                "evidence": "High-uniqueness identifier; treated as generic geocode."
            })
            continue

        # Indicators (quantitative / qualitative) — normalised to "columns"
        is_quant, payload_q = detect_quantitative_indicator(s, col)
        if is_quant:
            payload_q = dict(payload_q)  # copy
            payload_q["columns"] = _to_columns(payload_q, col)
            payload_q.pop("column", None)
            results["indicators"].append({
                "columns": payload_q["columns"],
                "type": str(s.dtype),
                **{k: v for k, v in payload_q.items() if k not in {"columns"}}
            })
            continue

        is_qual, payload_ql = detect_qualitative_indicator(s)
        if is_qual:
            payload_ql = dict(payload_ql)
            payload_ql["columns"] = _to_columns(payload_ql, col)
            payload_ql.pop("column", None)
            results["indicators"].append({
                "columns": payload_ql["columns"],
                "type": str(s.dtype),
                **{k: v for k, v in payload_ql.items() if k not in {"columns"}}
            })
            continue

        # Other
        avg_len = None
        try:
            avg_len = float(s.dropna().astype(str).str.len().mean())
        except Exception:
            pass
        results["other"].append({
            "columns": col,
            "description": "unclassified attribute",
            "type": str(s.dtype),
            "reason": "does not fit spatial/temporal or indicator heuristics",
            "avg_text_len": avg_len
        })

    return results




def classify_attributes_with_semantic_helper(
    df: pd.DataFrame,
    semantic_res: pd.DataFrame,
    *,
    min_ratio: float = 0.7,
    sample_size: int = 300,
    include_city_zip_in_address: bool = False,
    address_colname: str = "__address__",
    filename: str | None = None,
    sheet_names: List[str] | None = None,
    require_name_hint_for_geoformats: bool = True,
) -> Dict[str, Any]:
    """
    Hybrid classifier with semantic guidance.
    All entries consistently use "columns": [...] (never "column").
    """

    def _ensure_columns_name_list(name: str | List[str]) -> List[str]:
        if isinstance(name, list):
            return [str(x) for x in name]
        return [str(name)]

    def _col_list_from_sem(sem_df: pd.DataFrame, mask) -> List[str]:
        cols = sem_df.loc[mask, "column_name"].tolist()
        return [c for c in cols if c in df.columns]

    def _desc(sem_df: pd.DataFrame, col: str) -> str:
        vals = sem_df.loc[sem_df["column_name"] == col, "meaning"].tolist()
        return vals[0] if vals else ""

    results = {
        "spatial": [],
        "temporal": [],
        "unknown": [],
        "indicators": [],
        "other": [],
        "meta": {}
    }

    # 0) Resolve columns
    spatial_cols = _col_list_from_sem(semantic_res, semantic_res["is_spatial"] == True)
    temporal_cols = _col_list_from_sem(semantic_res, semantic_res["is_temporal"] == True)
    indicator_qual_cols = _col_list_from_sem(
        semantic_res, (semantic_res["is_indicator"] == True) & (semantic_res["indicator_type"] == "Qualitative")
    )
    indicator_quant_cols = _col_list_from_sem(
        semantic_res, (semantic_res["is_indicator"] == True) & (semantic_res["indicator_type"] == "Quantitative")
    )

    used_cols = set(spatial_cols + temporal_cols + indicator_qual_cols + indicator_quant_cols)
    other_cols = [c for c in df.columns if c not in used_cols]

    df_spatial = df[spatial_cols] if spatial_cols else pd.DataFrame(index=df.index)
    df_temporal = df[temporal_cols] if temporal_cols else pd.DataFrame(index=df.index)
    df_indicator_qual = df[indicator_qual_cols] if indicator_qual_cols else pd.DataFrame(index=df.index)
    df_indicator_quant = df[indicator_quant_cols] if indicator_quant_cols else pd.DataFrame(index=df.index)
    df_other = df[other_cols] if other_cols else pd.DataFrame(index=df.index)

    # 1) Spatial refinement
    df2, addr_cols = add_combined_address_column(
        df_spatial, colname=address_colname, include_city_zip=include_city_zip_in_address
    )
    results["meta"]["address_columns_used"] = addr_cols
    results["meta"]["config"] = {
        "min_ratio": min_ratio,
        "sample_size": sample_size,
        "include_city_zip_in_address": include_city_zip_in_address,
        "address_colname": address_colname,
        "filename": filename,
        "sheet_names": sheet_names or [],
        "require_name_hint_for_geoformats": require_name_hint_for_geoformats,
    }

    consumed: set = set()
    latlon = detect_latlon_pair(df2)
    if latlon:
        la, lo = latlon
        consumed.update([lo, la])
        try:
            df2[lo] = pd.to_numeric(df2[lo], errors="coerce")
            df2[la] = pd.to_numeric(df2[la], errors="coerce")
        except Exception:
            pass
        results["spatial"].append({
            "columns": [la, lo],  # keep [lat, lon]
            "description": "latlon pair geopoint",
            "granularity": "latlon_pair",
            "type": [str(df2[la].dtype), str(df2[lo].dtype)],
            "confidence": 0.98,
            "evidence": "Both columns numeric and within valid lat/lon ranges."
        })

    ref_sets = build_ref_sets(ref_dict) if ref_dict else {}

    for col in df2.columns:
        if col in consumed:
            continue
        s = df2[col]
        fmt_hints = geoformat_hints_from_colname(col) or {}

        if detect_geometry_object(s):
            conf = 0.99 if fmt_hints.get("geometry") else 0.98
            evd = "Objects/dtype appear to be geometry." + (" Column name suggests geometry." if fmt_hints.get("geometry") else "")
            results["spatial"].append({
                "columns": col,
                "description": "geometry",
                "type": str(s.dtype),
                "granularity": "geometry",
                "confidence": conf,
                "evidence": evd
            })
            continue

        if detect_wkt_geojson_string(s):
            name_gate_ok = fmt_hints.get("wkt") or fmt_hints.get("geojson") or fmt_hints.get("geometry")
            if (not require_name_hint_for_geoformats) or name_gate_ok:
                conf = 0.96 if name_gate_ok else 0.93
                evd = "Values match WKT/GeoJSON textual patterns." + (" Column name suggests WKT/GeoJSON." if name_gate_ok else "")
                results["spatial"].append({
                    "columns": col,
                    "description": "geometry",
                    "type": str(s.dtype),
                    "granularity": "geometry",
                    "confidence": conf,
                    "evidence": evd
                })
                continue

        if col == address_colname and detect_address(s):
            results["spatial"].append({
                "columns": addr_cols,
                "description": "complete address",
                "type": str(s.dtype),
                "granularity": "address",
                "confidence": 0.8,
                "evidence": "Aggregated address lines with street tokens and numbers."
            })
            continue
        if not addr_cols and detect_address(s):
            results["spatial"].append({
                "columns": col,
                "description": "complete address",
                "type": str(s.dtype),
                "granularity": "address",
                "confidence": 0.75,
                "evidence": "Address-like strings detected."
            })
            continue

        gate = spatial_gate_from_colname(col) or {"has_gate": False, "levels": [], "generic": False}
        des = _desc(semantic_res, col)

        if ref_sets and not_null_ratio(s) > 0.2 and gate.get("has_gate", False):
            if gate.get("levels"):
                filtered = {lvl: ref_sets[lvl] for lvl in gate["levels"] if lvl in ref_sets}
                if filtered:
                    best = match_series_to_ref_levels(s, filtered, sample_size=sample_size)
                    if best and best["ratio"] >= min_ratio:
                        results["spatial"].append({
                            "columns": col,
                            "description": des,
                            "type": str(s.dtype),
                            "granularity": best["level"],
                            "matched_by": best["by"],
                            "confidence": min(0.99, 0.7 + 0.3 * best["ratio"]),
                            "evidence": f"Name gate [{', '.join(gate['levels'])}] → ref match by {best['by']} ({round(best['ratio'] * 100)}%)."
                        })
                        continue
                    elif best:
                        results["other"].append({
                            "columns": col,
                            "description": des,
                            "type": str(s.dtype),
                            "granularity": best["level"],
                            "matched_by": best["by"]
                        })
                        continue

            elif gate.get("generic", False):
                best = match_series_to_ref_levels(s, ref_sets, sample_size=sample_size) if ref_sets else None
                generic_min = max(min_ratio, 0.80)
                if best and best["ratio"] >= generic_min:
                    results["spatial"].append({
                        "columns": col,
                        "description": des,
                        "type": str(s.dtype),
                        "granularity": best["level"],
                        "matched_by": best["by"],
                        "confidence": min(0.99, 0.68 + 0.32 * best["ratio"]),
                        "evidence": f"Generic geo gate → ref match by {best['by']} ({round(best['ratio'] * 100)}%)."
                    })
                    continue
                elif best:
                    results["other"].append({
                        "columns": col,
                        "description": des,
                        "type": str(s.dtype),
                        "granularity": best["level"],
                        "matched_by": best["by"]
                    })
                    continue

    # 2) Temporal (always "columns":[col])
    for col in df_temporal.columns:
        des = _desc(semantic_res, col)
        s = df_temporal[col]
        gran, conf = detect_temporal_granularity(s)
        if gran:
            results["temporal"].append({
                "columns": col,
                "description": des,
                "type": str(s.dtype),
                "granularity": gran,
                "confidence": conf
            })
        else:
            results["other"].append({
                "columns": col,
                "description": des,
                "type": str(s.dtype),
                "granularity": None,
                "confidence": None
            })

    # 3) Indicators — builders may return 'column'; normalise to 'columns'
    for col in df_indicator_quant.columns:
        s = df_indicator_quant[col]
        entry = build_quantitative_entry(col, s, semantic_res)
        if entry:
            entry = dict(entry)
            entry["columns"] = _ensure_columns_name_list(entry.get("columns", col))
            entry.pop("column", None)
            results["indicators"].append(entry)

    for col in df_indicator_qual.columns:
        s = df_indicator_qual[col]
        entry = build_qualitative_entry(col, s, semantic_res)
        if entry:
            entry = dict(entry)
            entry["columns"] = _ensure_columns_name_list(entry.get("columns", col))
            entry.pop("column", None)
            results["indicators"].append(entry)

    # 4) Other (pure semantic 'other')
    for col in df_other.columns:
        s = df_other[col]
        des = _desc(semantic_res, col)
        avg_len = None
        try:
            avg_len = float(s.dropna().astype(str).str.len().mean())
        except Exception:
            pass
        results["other"].append({
            "columns": col,
            "description": des,
            "type": str(s.dtype),
            "reason": "Semantically detected as other information",
            "avg_text_len": avg_len
        })

    # 5) Meta
    results["meta"]["config"] = {
        "min_ratio": min_ratio,
        "sample_size": sample_size,
        "include_city_zip_in_address": include_city_zip_in_address,
        "address_colname": address_colname,
        "filename": filename,
        "sheet_names": sheet_names or [],
        "require_name_hint_for_geoformats": require_name_hint_for_geoformats,
    }

    return results
