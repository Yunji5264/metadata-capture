from general_function import *


# --- Tunable thresholds (centralised) ---------------------------------------
_INDICATOR_THRESHOLDS = {
    # For numeric detection after coercion
    "numeric_ratio_min": 0.80,     # at least 80% of non-null values must coerce to numeric
    "nonzero_ratio_min": 0.10,     # at least 10% values non-zero to avoid all-zeros columns
    # For excluding IDs (too unique)
    "id_like_nunique_ratio_min": 0.90,  # >=90% unique -> likely an ID, not quantitative
    # For qualitative category detection
    "qual_max_nunique_ratio": 0.20,     # unique/rows <= 20%
    "qual_max_abs_unique": 50,          # and absolute unique <= 50 (hard cap)
    "qual_min_abs_unique": 2,           # at least 2 distinct categories
}

# --- Utilities ---------------------------------------------------------------
_CURRENCY_RE = re.compile(r"[€$£¥]|(eur|usd|gbp|cny|rmb)", re.I)
_THOUSANDS_SEP_RE = re.compile(r"[ \u00A0\u2009\u202F]", re.UNICODE)  # space, nbsp, thin space, narrow nbspace
_PERCENT_RE = re.compile(r"%")
_SIGN_RE = re.compile(r"^[\+\-]")
_NON_NUM_KEEP_DOT_COMMA_RE = re.compile(r"[^0-9\-,\.]")

def _coerce_numeric_like(s: pd.Series) -> Tuple[pd.Series, float]:
    """
    Try to coerce a possibly string-like numeric series into float.
    Handles:
      - currency symbols (€, $, etc.)
      - thousands separators (space, nbsp, thin space)
      - percent values (keeps as raw number; detection done elsewhere)
      - French decimal comma -> dot
    Returns (numeric_series, numeric_ratio among non-null).
    """
    if pd.api.types.is_numeric_dtype(s):
        s_num = pd.to_numeric(s, errors="coerce")
    else:
        x = s.astype(str).str.strip()
        # Remove currency words/symbols
        x = x.str.replace(_CURRENCY_RE, "", regex=True)
        # Remove thousands separators
        x = x.str.replace(_THOUSANDS_SEP_RE, "", regex=True)
        # Keep only digits, signs, dot/comma, hyphen in-between
        x = x.str.replace(_NON_NUM_KEEP_DOT_COMMA_RE, "", regex=True)
        # Replace French decimal comma with dot (if comma present and dot absent)
        # Heuristic: if both comma and dot appear, prefer removing thousands via previous step and treat dot as decimal.
        x = x.str.replace(",", ".", regex=False)
        # Remove leading '+' signs
        x = x.str.replace(_SIGN_RE, "", regex=True)
        s_num = pd.to_numeric(x, errors="coerce")

    non_null = s.notna()
    numeric_ok = s_num.notna() & non_null
    ratio = (numeric_ok.sum() / max(1, non_null.sum())) if non_null.any() else 0.0
    return s_num, ratio

def _detect_boolean_series(s: pd.Series) -> Tuple[bool, dict]:
    """
    Detect boolean-like series: yes/no, true/false, 0/1, oui/non, y/n.
    Returns (is_boolean, payload)
    """
    if s.empty:
        return False, {}
    if pd.api.types.is_bool_dtype(s):
        vals = s.dropna().unique().tolist()
        return True, {"true_values": [True], "false_values": [False], "unique": len(vals)}
    # Normalize text for common boolean forms
    mapping = {
        "true": True, "false": False,
        "yes": True, "no": False,
        "y": True, "n": False,
        "1": True, "0": False,
        "oui": True, "non": False,
        "vrai": True, "faux": False,
        "t": True, "f": False,
    }
    t = s.dropna().astype(str).str.strip().str.lower()
    m = t.map(mapping).dropna()
    if not t.empty and (len(m) / len(t) >= 0.95):
        uniq = set(m.unique().tolist())
        if uniq.issubset({True, False}) and len(uniq) <= 2:
            return True, {"mapping": "common_boolean_strings", "unique": len(uniq)}
    return False, {}

def _detect_likert_series(s: pd.Series) -> Tuple[bool, dict]:
    """
    Detect Likert-like ordinal categories: e.g., 'strongly agree' ... 'strongly disagree',
    or small ordinal sets like {1..5} in text or ints.
    """
    if s.empty:
        return False, {}
    t = s.dropna().astype(str).str.strip().str.lower()

    # Numeric 1..5 or 1..7 common Likert scales
    try:
        n = pd.to_numeric(t, errors="coerce")
        if n.notna().mean() >= 0.95:
            vals = set(n.dropna().astype(int).tolist())
            for k in (5, 7):
                if vals.issubset(set(range(1, k+1))) and 2 <= len(vals) <= k:
                    return True, {"pattern": f"1..{k} ordinal", "unique": len(vals)}
    except Exception:
        pass

    # Textual Likert keywords
    keys = ["strongly agree", "agree", "neutral", "disagree", "strongly disagree",
            "tout a fait d'accord", "plutot d'accord", "ni d'accord ni pas d'accord",
            "plutot pas d'accord", "pas du tout d'accord"]
    score = t.apply(lambda x: any(k in x for k in keys)).mean()
    if score >= 0.6:
        return True, {"pattern": "likert_keywords", "coverage": round(float(score), 3)}
    return False, {}

def _name_hints_rate_count(colname: str) -> dict:
    """
    Extract weak hints from column name for subtype classification.
    """
    n = normalise_colname(colname)
    hints = {
        "is_rate": bool(re.search(r"\b(rate|taux|ratio|pourcentage|pct|percent|percentage|per_100k|per_capita)\b", n)),
        "is_count": bool(re.search(r"\b(nb|nbr|count|nombre|qty|quantite|effectif|n_)\b", n)),
        "is_index": bool(re.search(r"\b(index|indice|score|note)\b", n)),
        "is_mean": bool(re.search(r"\b(mean|moyenne|avg)\b", n)),
        "is_median": bool(re.search(r"\b(median|mediane)\b", n)),
    }
    return hints

def detect_quantitative_indicator(s: pd.Series, colname: str) -> Tuple[bool, dict]:
    """
    Detect quantitative indicator candidates:
      - Majority numerically coercible
      - Not ID-like (too unique)
      - Not all zeros or constant
      - Subtype heuristics: count / rate% / index / continuous
    """
    N = len(s)
    non_null = s.notna().sum()
    if N == 0 or non_null == 0:
        return False, {}

    s_num, num_ratio = _coerce_numeric_like(s)
    if num_ratio < _INDICATOR_THRESHOLDS["numeric_ratio_min"]:
        return False, {}

    # Exclude ID-like: too unique or strictly integer sequential with near 1:1 uniqueness
    nunique = s_num.nunique(dropna=True)
    nunique_ratio = nunique / max(1, non_null)
    if nunique_ratio >= _INDICATOR_THRESHOLDS["id_like_nunique_ratio_min"]:
        return False, {}

    # Must not be almost constant or almost all zero
    non_zero_ratio = (s_num.fillna(0) != 0).mean()
    if non_zero_ratio < _INDICATOR_THRESHOLDS["nonzero_ratio_min"]:
        return False, {}

    # Subtype classification
    hints = _name_hints_rate_count(colname)
    finite = s_num.replace([np.inf, -np.inf], np.nan).dropna()
    subtype = "continuous"
    evidence = []
    conf = 0.9

    # rate / percentage heuristic
    if hints["is_rate"]:
        subtype = "rate"
        evidence.append("name hint: rate-like keyword")
    else:
        # If values mostly in [0,1] or [0,100], treat as rate/percentage
        if not finite.empty:
            in_0_1 = ((finite >= 0) & (finite <= 1)).mean()
            in_0_100 = ((finite >= 0) & (finite <= 100)).mean()
            if in_0_1 >= 0.9:
                subtype = "rate_fraction"
                evidence.append("values mostly in [0,1]")
            elif in_0_100 >= 0.9:
                subtype = "percentage_like"
                evidence.append("values mostly in [0,100]")

    # count heuristic
    if hints["is_count"]:
        as_int = (finite.round().sub(finite).abs() < 1e-9).mean()
        if as_int >= 0.95:
            subtype = "count"
            evidence.append("name hint: count-like keyword and integer values")
        else:
            evidence.append("name hint: count-like keyword (non-integer values)")

    # index/score heuristic
    if hints["is_index"]:
        subtype = "index"
        evidence.append("name hint: index/score keyword")

    # summary
    payload = {
        "suggested_dtype": "float" if subtype not in ("count",) else "int",
        "non_null": int(non_null),
        "nunique": int(nunique),
        "nunique_ratio": float(round(nunique_ratio, 3)),
        "value_range": (
            float(finite.min()) if not finite.empty else None,
            float(finite.max()) if not finite.empty else None,
        ),
        "subtype": subtype,
        "evidence": ", ".join(evidence) if evidence else "numeric coercion passed",
    }
    # Confidence tweaks
    if subtype in ("rate", "rate_fraction", "percentage_like", "index", "count"):
        conf = 0.93
    return True, {"indicator_type": "quantitative", "confidence": conf, **payload}

def detect_qualitative_indicator(s: pd.Series) -> Tuple[bool, dict]:
    """
    Detect qualitative (categorical/label) indicator candidates:
      - String-like or mixed
      - Unique ratio not too high (exclude IDs)
      - Reasonable number of distinct categories
      - Includes boolean and Likert detection
    """
    if s.empty:
        return False, {}

    # Boolean?
    is_bool, payload_bool = _detect_boolean_series(s)
    if is_bool:
        cats = s.dropna().astype(str).str.strip().str.lower().unique().tolist()
        return True, {
            "indicator_type": "qualitative",
            "subtype": "binary",
            "confidence": 0.97,
            "categories_sample": cats[:10],
            "evidence": "boolean-like values",
        }

    # Likert?
    is_likert, payload_likert = _detect_likert_series(s)
    if is_likert:
        cats = s.dropna().astype(str).str.strip().str.lower().unique().tolist()
        return True, {
            "indicator_type": "qualitative",
            "subtype": "ordinal_likert",
            "confidence": 0.94,
            "categories_sample": cats[:10],
            "evidence": f"likert pattern ({payload_likert.get('pattern','')})",
        }

    # Categorical by cardinality
    non_null = s.notna().sum()
    if non_null == 0:
        return False, {}
    nunique = s.nunique(dropna=True)
    nunique_ratio = nunique / max(1, non_null)

    if (nunique_ratio <= _INDICATOR_THRESHOLDS["qual_max_nunique_ratio"]
        and _INDICATOR_THRESHOLDS["qual_min_abs_unique"] <= nunique <= _INDICATOR_THRESHOLDS["qual_max_abs_unique"]):
        # candidate qualitative
        cats = s.dropna().astype(str).str.strip().unique().tolist()
        # lightweight text-length heuristic: avoid free-text paragraphs
        avg_len = s.dropna().astype(str).str.len().mean()
        if avg_len <= 64:  # long free text unlikely to be categorical indicator
            return True, {
                "indicator_type": "qualitative",
                "subtype": "nominal",
                "confidence": 0.9,
                "nunique": int(nunique),
                "nunique_ratio": float(round(nunique_ratio, 3)),
                "categories_sample": cats[:20],
                "evidence": "limited distinct categories and short average token length",
            }
    return False, {}
#
# # ---- Extend your existing _INDICATOR_THRESHOLDS with guard-specific defaults (idempotent) ----
# # These keys are added only if not already present.
# _GUARD_DEFAULTS = {
#     # Quantitative guard thresholds
#     "guard_name_numeric_min_ratio": 0.50,   # if name hints a measure AND >=50% numeric -> quantitative
#     "guard_numeric_min_ratio": 0.80,        # if >=80% numeric and other numeric signals -> quantitative
#     "guard_decimals_min_ratio": 0.05,       # fraction with decimals to consider "continuous"
#     "guard_nunique_numeric_min_ratio": 0.50,# high uniqueness among numeric values indicates measure
#     "guard_value_range_min": 20.0,          # wide numeric range indicates measure
#
#     # Qualitative guard thresholds
#     "qual_max_avg_len": 64.0,               # average string length threshold to exclude free text
#     "guard_likert_min_coverage": 0.60,      # textual Likert keywords coverage
#     "guard_boolean_min_coverage": 0.95,     # proportion of boolean-mappable values
# }
# for _k, _v in _GUARD_DEFAULTS.items():
#     _INDICATOR_THRESHOLDS.setdefault(_k, _v)
#
# # ---- Name hint regexes (quantitative, qualitative, and spatial “do-not-guard” hints) ---------
# _QUAN_NAME_RE = re.compile(
#     r"\b(rate|taux|pct|pourcentage|ratio|score|index|indice|moyenne|avg|median|mediane|"
#     r"sum|total|val(eur|or)|value|revenu|income|population|pop|densite|density|surface|surf(?:_?hab)?|area|"
#     r"nb|nbr|nombre|count|qty|quantite|pour_mille|per_?100k|per_?capita)\b",
#     re.I,
# )
# _QUAL_NAME_RE = re.compile(
#     r"\b(libelle|label|nom|name|categorie|category|type|genre|sexe|statut|status|"
#     r"classe|class|groupe|group|tranche|modalite|moda|description|desc|qtv)\b",
#     re.I,
# )
# # Fallback spatial hints (used only if spatial_level_hints_from_colname is not available)
# _SPATIAL_NAME_HINT_RE = re.compile(
#     r"\b(reg|region|code_?reg(ion)?|dep|dpt|departement|code_?dep|canton|code_?canton|"
#     r"epci|code_?epci|siren_?epci|academie|aca|code_?aca(demie)?|"
#     r"com|commune|code_?com(mune)?|insee_?com|iris|code_?iris)\b",
#     re.I,
# )
#
# def _has_spatial_hint(colname: str) -> bool:
#     """Return True if the column name strongly suggests a spatial level (DEP/REG/COM/...)."""
#     # Prefer your existing helper if present
#     try:
#         if 'spatial_level_hints_from_colname' in globals():
#             return bool(spatial_level_hints_from_colname(colname))
#     except Exception:
#         pass
#     # Fallback to regex if helper not available
#     try:
#         n = normalise_colname(colname)
#     except Exception:
#         n = str(colname).lower().strip()
#     return bool(_SPATIAL_NAME_HINT_RE.search(n))
#
# def _numeric_profile_for_indicator_guard(s: pd.Series) -> dict:
#     """
#     Profile numeric characteristics using your _coerce_numeric_like to stay consistent
#     with quantitative detection.
#     """
#     s_num, num_ratio = _coerce_numeric_like(s)
#     N = len(s)
#     non_null = int(s_num.notna().sum())
#     if N == 0 or non_null == 0:
#         return {"num_ratio": 0.0, "decimals_ratio": 0.0, "nunique_ratio_num": 0.0, "value_range": 0.0}
#
#     decimals_ratio = float(((s_num.dropna() % 1) != 0).mean())
#     nunique_ratio_num = float(s_num.nunique(dropna=True) / non_null)
#     finite = s_num.replace([np.inf, -np.inf], np.nan).dropna()
#     value_range = float(finite.max() - finite.min()) if not finite.empty else 0.0
#     return {
#         "num_ratio": num_ratio,
#         "decimals_ratio": decimals_ratio,
#         "nunique_ratio_num": nunique_ratio_num,
#         "value_range": value_range,
#     }
#
# def _boolean_coverage(series: pd.Series) -> float:
#     """Return coverage of boolean-mappable values based on your _detect_boolean_series mapping."""
#     # Reuse your boolean detector if available
#     try:
#         ok, _ = _detect_boolean_series(series)
#         if ok:
#             return 1.0
#     except Exception:
#         pass
#     # Fallback lightweight check
#     t = series.dropna().astype(str).str.strip().str.lower()
#     if t.empty:
#         return 0.0
#     mapping = {
#         "true": True, "false": False, "t": True, "f": False,
#         "yes": True, "no": False, "y": True, "n": False,
#         "1": True, "0": False, "oui": True, "non": False,
#         "vrai": True, "faux": False, "male": True, "female": False,
#         "m": True, "f": False, "homme": True, "femme": False, "h": True,
#     }
#     m = t.map(mapping)
#     return float(m.notna().mean())
#
# def _is_likert_like_guard(series: pd.Series) -> bool:
#     """Use your Likert detector if present; fallback to a lean check."""
#     try:
#         is_likert, _ = _detect_likert_series(series)
#         return bool(is_likert)
#     except Exception:
#         pass
#     t = series.dropna().astype(str).str.strip().str.lower()
#     if t.empty:
#         return False
#     n = pd.to_numeric(t, errors="coerce")
#     if n.notna().mean() >= 0.95:
#         vals = set(n.dropna().astype(int).tolist())
#         for k in (5, 7):
#             if vals.issubset(set(range(1, k + 1))) and 2 <= len(vals) <= k:
#                 return True
#     keys = [
#         "strongly agree", "agree", "neutral", "disagree", "strongly disagree",
#         "tout a fait d'accord", "plutot d'accord", "ni d'accord ni pas d'accord",
#         "plutot pas d'accord", "pas du tout d'accord"
#     ]
#     cov = t.apply(lambda x: any(k in x for k in keys)).mean()
#     return cov >= _INDICATOR_THRESHOLDS["guard_likert_min_coverage"]
#
# def should_skip_spatial_match_for_indicator(s: pd.Series, colname: str) -> bool:
#     """
#     Unified guard for both quantitative and qualitative indicators.
#     Returns True -> skip spatial code/name matching for this column.
#
#     Logic:
#       0) If the column name carries explicit spatial hints (DEP/REG/COM/...), do NOT skip.
#       1) Quantitative guard:
#          - name has measure keywords AND >= guard_name_numeric_min_ratio numeric; OR
#          - >= guard_numeric_min_ratio numeric AND (has decimals OR high numeric uniqueness OR wide range)
#       2) Qualitative guard (only if quantitative guard did not trigger):
#          - boolean coverage >= guard_boolean_min_coverage; OR
#          - Likert-like; OR
#          - small nominal categories with short labels, and (NOT mostly numeric OR name has qualitative hint)
#     """
#     # 0) Spatial hint in name → let spatial matcher run
#     if _has_spatial_hint(colname):
#         return False
#
#     # Normalised name
#     try:
#         nname = normalise_colname(colname)
#     except Exception:
#         nname = str(colname).lower().strip()
#
#     # 1) Quantitative guard
#     prof = _numeric_profile_for_indicator_guard(s)
#     name_measure_hit = bool(_QUAN_NAME_RE.search(nname))
#
#     quant_guard = (
#         (name_measure_hit and prof["num_ratio"] >= _INDICATOR_THRESHOLDS["guard_name_numeric_min_ratio"])
#         or (
#             prof["num_ratio"] >= _INDICATOR_THRESHOLDS["guard_numeric_min_ratio"]
#             and (
#                 prof["decimals_ratio"] >= _INDICATOR_THRESHOLDS["guard_decimals_min_ratio"]
#                 or prof["nunique_ratio_num"] >= _INDICATOR_THRESHOLDS["guard_nunique_numeric_min_ratio"]
#                 or prof["value_range"] > _INDICATOR_THRESHOLDS["guard_value_range_min"]
#             )
#         )
#     )
#     if quant_guard:
#         return True
#
#     # 2) Qualitative guard
#     if _boolean_coverage(s) >= _INDICATOR_THRESHOLDS["guard_boolean_min_coverage"]:
#         return True
#     if _is_likert_like_guard(s):
#         return True
#
#     non_null = int(s.notna().sum())
#     if non_null == 0:
#         return False
#     nunique = int(s.nunique(dropna=True))
#     nunique_ratio = nunique / non_null if non_null else 0.0
#     avg_len = float(s.dropna().astype(str).str.len().mean() or 0.0)
#
#     small_nominal = (
#         _INDICATOR_THRESHOLDS["qual_min_abs_unique"] <= nunique <= _INDICATOR_THRESHOLDS["qual_max_abs_unique"]
#         and nunique_ratio <= _INDICATOR_THRESHOLDS["qual_max_nunique_ratio"]
#         and avg_len <= _INDICATOR_THRESHOLDS["qual_max_avg_len"]
#     )
#
#     mostly_numeric = prof["num_ratio"] >= _INDICATOR_THRESHOLDS["guard_numeric_min_ratio"]
#     name_qual_hint = bool(_QUAL_NAME_RE.search(nname))
#
#     qualitative_guard = small_nominal and (not mostly_numeric or name_qual_hint)
#     return qualitative_guard