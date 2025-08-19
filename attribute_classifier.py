from spaital_detector import *
from temporal_detector import *
from indicator_detector import *

def classify_attributes(
    df: pd.DataFrame,
    ref_dict: Dict[str, pd.DataFrame] | None = None,
    *,
    min_ratio: float = 0.7,
    sample_size: int = 300,
    include_city_zip_in_address: bool = False,
    address_colname: str = "__address__",
    filename: str | None = None,
    sheet_names: List[str] | None = None,
    require_name_hint_for_geoformats: bool = True,   # <-- NEW
) -> Dict[str, Any]:
    """
    Unified classifier combining reference matching and heuristics for spatial,
    plus temporal granularity detection.

    Returns:
    {
      "spatial": [
        {"columns": [...], "type": "latlon_pair", "granularity": "latlon_pair", "confidence": 0.98, "evidence": "..."},
        {"columns": ["dep_code"], "type": "departement", "granularity": "departement", "confidence": 0.93, "evidence": "Matched by ref code (93%)."},
        ...
      ],
      "temporal": [
        {"column": "date", "granularity": "day", "confidence": 0.95}
      ],
      "unknown": [...],
      "meta": {"address_columns_used": [...], "config": {...}}
    }
    """
    results = {
        "spatial": [],
        "temporal": [],
        "unknown": [],
        "indicators": [],  # <-- NEW
        "other": [],  # <-- NEW
        "meta": {}
    }

    # 0) Address aggregation (if any)
    df2, addr_cols = add_combined_address_column(
        df, colname=address_colname, include_city_zip=include_city_zip_in_address
    )
    results["meta"]["address_columns_used"] = addr_cols
    results["meta"]["config"] = {
        "min_ratio": min_ratio,
        "sample_size": sample_size,
        "include_city_zip_in_address": include_city_zip_in_address,
        "address_colname": address_colname
    }

    # 1) Lat/Lon pair detection (consumes two columns)
    consumed: set = set()
    latlon = detect_latlon_pair(df2)
    if latlon:
        la, lo = latlon
        consumed.update([la, lo])
        results["spatial"].append({
            "columns": [la, lo],
            "type": "latlon_pair",
            "granularity": "latlon_pair",
            "confidence": 0.98,
            "evidence": "Both columns numeric and within valid lat/lon ranges."
        })

    # 2) Pre-build reference sets (if given)
    ref_sets = build_ref_sets(ref_dict) if ref_dict else None

    # 3) Per-column classification
    for col in df2.columns:
        if col in consumed:
            continue

        # ALWAYS define s at the top of the loop
        s = df2[col]

        # ---- Geometry / WKT-GeoJSON / Geohash with optional name gating ----
        fmt_hints = geoformat_hints_from_colname(col)

        # Geometry objects (dtype=geometry or objects with __geo_interface__)
        if detect_geometry_object(s):
            conf = 0.98 if not fmt_hints.get("geometry") else 0.99
            evd = "Objects/dtype appear to be geometry." + (
                " Column name suggests geometry." if fmt_hints.get("geometry") else "")
            results["spatial"].append({
                "columns": [col], "type": "geometry", "granularity": "geometry",
                "confidence": conf, "evidence": evd
            })
            continue

        # WKT / GeoJSON stored as text
        if detect_wkt_geojson_string(s):
            if (not require_name_hint_for_geoformats) or (
                    fmt_hints.get("wkt") or fmt_hints.get("geojson") or fmt_hints.get("geometry")):
                conf = 0.93 if not (
                            fmt_hints.get("wkt") or fmt_hints.get("geojson") or fmt_hints.get("geometry")) else 0.96
                evd = "Values match WKT/GeoJSON textual patterns." + (" Column name suggests WKT/GeoJSON." if (
                            fmt_hints.get("wkt") or fmt_hints.get("geojson") or fmt_hints.get("geometry")) else "")
                results["spatial"].append({
                    "columns": [col], "type": "wkt_geojson", "granularity": "wkt_geojson",
                    "confidence": conf, "evidence": evd
                })
                continue

        # Geohash (base32 without a,i,l,o) — with optional name gating
        if detect_geohash(s):
            if (not require_name_hint_for_geoformats) or fmt_hints.get("geohash"):
                conf = 0.90 if not fmt_hints.get("geohash") else 0.94
                evd = "Values match geohash charset/length." + (
                    " Column name mentions geohash." if fmt_hints.get("geohash") else "")
                results["spatial"].append({
                    "columns": [col], "type": "geohash", "granularity": "geohash",
                    "confidence": conf, "evidence": evd
                })
                continue

        # Address (especially the combined one)
        if col == address_colname and detect_address(s):
            results["spatial"].append({
                "columns": [col], "type": "address", "granularity": "address",
                "confidence": 0.8, "evidence": "Aggregated address lines with street tokens and numbers."
            })
            continue
        # Also allow single-column address detection if no aggregation occurred
        if not addr_cols and detect_address(s):
            results["spatial"].append({
                "columns": [col], "type": "address", "granularity": "address",
                "confidence": 0.75, "evidence": "Address-like strings detected."
            })
            continue

        # 3.2 Spatial: reference matching (GATED by column name)
        gate = spatial_gate_from_colname(col)
        if not_null_ratio(s) > 0.2 and gate["has_gate"]:
            if gate["levels"]:
                # Strict: only hinted levels
                filtered = {lvl: ref_sets[lvl] for lvl in gate["levels"] if lvl in ref_sets}
                if filtered:
                    best = match_series_to_ref_levels(s, filtered, sample_size=sample_size)
                    if best and best["ratio"] >= min_ratio:
                        results["spatial"].append({
                            "columns": [col],
                            "granularity": best["level"],
                            "matched_by": best["by"],
                            "confidence": min(0.99, 0.7 + 0.3 * best["ratio"]),
                            "evidence": (
                                f"Name gate [{', '.join(gate['levels'])}] → ref match by {best['by']} ({round(best['ratio'] * 100)}%).")
                        })
                        continue
                    elif best:
                        results["other"].append({
                            "columns": [col],
                            "granularity": best["level"],
                            "matched_by": best["by"]
                        })
                        continue
                # If hinted but below threshold, do not try other levels (strict by design).

            elif gate["generic"]:
                # Generic gate (e.g., 'zone', 'geo', 'codgeo'): allow all levels but raise bar a bit
                best = match_series_to_ref_levels(s, ref_sets, sample_size=sample_size)
                generic_min = max(min_ratio, 0.80)
                if best and best["ratio"] >= generic_min:
                    results["spatial"].append({
                        "columns": [col],
                        "type": best["level"],
                        "granularity": best["level"],
                        "matched_by": best["by"],
                        "confidence": min(0.99, 0.68 + 0.32 * best["ratio"]),
                        "evidence": (f"Generic geo gate → ref match by {best['by']} ({round(best['ratio'] * 100)}%).")
                    })
                    continue
                elif best:
                    results["other"].append({
                        "columns": [col],
                        "granularity": best["level"],
                        "matched_by": best["by"]
                    })
                    continue

        # 3.3 Temporal
        gran, conf = detect_temporal_granularity(s)
        if gran:
            results["temporal"].append({
                "column": col, "granularity": gran, "confidence": conf
            })
            continue

        # 3.4 Generic geocode fallback (ID-like high uniqueness)
        nunique_ratio = s.nunique(dropna=True) / max(1, len(s))
        if nunique_ratio > 0.9 and (is_string_series(s) or is_numeric_series(s)):
            results["spatial"].append({
                "columns": [col], "type": "geocode", "granularity": "geocode",
                "confidence": 0.6, "evidence": "High-uniqueness identifier; treated as generic geocode."
            })
            continue

        # 3.5 Indicators (quantitative / qualitative) before "other"
        is_quant, payload_q = detect_quantitative_indicator(s, col)
        if is_quant:
            results["indicators"].append({"column": col, **payload_q})
            continue

        is_qual, payload_ql = detect_qualitative_indicator(s)
        if is_qual:
            results["indicators"].append({"column": col, **payload_ql})
            continue

        # 3.6 Other
        avg_len = None
        try:
            avg_len = float(s.dropna().astype(str).str.len().mean())
        except Exception:
            pass
        results["other"].append({
            "column": col,
            "reason": "does not fit spatial/temporal or indicator heuristics",
            "avg_text_len": avg_len
        })

    results["meta"]["config"] = {
        "min_ratio": min_ratio,
        "sample_size": sample_size,
        "include_city_zip_in_address": include_city_zip_in_address,
        "address_colname": address_colname,
        "filename": filename,
        "sheet_names": sheet_names or [],
        "require_name_hint_for_geoformats": require_name_hint_for_geoformats,  # <-- add
    }

    return results

