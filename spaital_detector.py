from general_function import *

# Adress multi-column handling

def find_address_cols(df: pd.DataFrame, include_city_zip: bool = False) -> List[str]:
    """Detect address sub-columns and return them in a canonical order for concatenation."""
    # --- Role-specific patterns (French admin datasets) ---------------------
    role_patterns = {
        # Thoroughfare core
        "house_number": [
            r"^(num(?:ero)?(?:_?voie)?|no_?voie|num_?voie|numvoie|n(?:o|um))$",
        ],
        "repetition_index": [
            r"^(indrep|indice(_?de)?_?repet(?:ition)?)$",
        ],
        "way_type": [
            r"^(typ(?:e)?_?voie|type_?rue|typvoie)$",
        ],
        "way_name": [
            r"^(nom_?voie|libelle_?voie|lib_?voie|nomvoie|nom_?rue)$",
        ],
        # Building / unit details
        "building": [
            r"^(bat(?:iment)?|batiment|immeu(?:ble)?|immeuble|tour|bloc)$",
        ],
        "entrance": [
            r"^(entree|ent)$",
        ],
        "stair": [
            r"^(esc(?:alier)?)$",
        ],
        "floor": [
            r"^(etage|niveau)$",
        ],
        "corridor": [
            r"^(couloir|coul)$",
        ],
        "unit_number": [
            r"^(num(app?t|apt|appartement|logt|logement)|app?t|apt|porte)$",
        ],
        "mailbox": [
            r"^(numboite|boite(_?lettres)?|bp|cs)$",
        ],
        # Complements / generic address lines
        "complement": [
            r"^(compl(ement)?(_?(adresse|ident|geo))?)$",
            r"^(lieu(_|-)?dit|residence|resid|quartier)$",
        ],
        "address_line": [
            r"^(adresse|address|addr)(_?\d+)?$",
            r"^(ligne|line)_?\d+$",
        ],
        # City/ZIP (optional)
        "postal_code": [
            r"^(cp|code(_|-)?postal|postal(_|-)?code|zip)$",
        ],
        "city": [
            r"^(ville|commune|localite|city|town)$",
        ],
        # Also allow direct thoroughfare tokens (rare single-field forms)
        "thoroughfare_token": [
            r"^(voie|rue|avenue|bd|boulevard|av|chemin|route|impasse|allee|place|quai|cours|quartier|lieu(_|-)?dit)$"
        ],
    }

    # Compile all regexes once
    role_res = {role: [re.compile(p, re.I) for p in pats] for role, pats in role_patterns.items()}

    # Canonical output order
    order = [
        "house_number", "repetition_index", "way_type", "way_name",          # line 1
        "building", "entrance", "stair", "floor", "corridor", "unit_number", "mailbox",  # line 2
        "complement", "address_line", "thoroughfare_token"                   # complements / generic lines
    ]
    if include_city_zip:
        order += ["postal_code", "city"]

    # Match roles while preserving the original column order per role
    matched: Dict[str, List[str]] = {k: [] for k in role_patterns.keys()}
    for col in df.columns:
        norm = normalise_colname(col)
        for role, regs in role_res.items():
            if any(r.match(norm) for r in regs):
                matched[role].append(col)
                break  # stop at the first role that matches (highest specificity wins)

    # Flatten in canonical order and deduplicate while preserving order
    cols: List[str] = []
    seen: set = set()
    for role in order:
        for c in matched.get(role, []):
            if c not in seen:
                cols.append(c)
                seen.add(c)

    return cols

def concat_columns_safe(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """Concatenate multiple columns into a single string series with spaces."""
    if not cols:
        return pd.Series([None]*len(df), index=df.index, dtype="object")
    s = pd.Series([""]*len(df), index=df.index, dtype="object")
    for c in cols:
        part = df[c].astype(str).where(df[c].notna(), "")
        s = s.str.cat(part, sep=" ")
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    s = s.mask(s.eq(""))
    return s

def add_combined_address_column(
    df: pd.DataFrame,
    colname: str = "__address__",
    include_city_zip: bool = False
) -> Tuple[pd.DataFrame, List[str]]:
    """Return a copy with aggregated address column; drop original address sub-columns."""
    addr_cols = find_address_cols(df, include_city_zip=include_city_zip)
    df2 = df.copy()
    if addr_cols:
        df2[colname] = concat_columns_safe(df2, addr_cols)
        df2 = df2.drop(columns=addr_cols)
    return df2, addr_cols

# --- Column-name first: spatial level hints ---------------------------------
_SPATIAL_COLNAME_HINTS = [
    # region
    ("reg", re.compile(r"\b(?:reg|region|code_?reg(?:ion)?|insee_?reg(?:ion)?)\b", re.I)),
    # departement
    ("dep", re.compile(r"\b(?:dep|dpt|departement|code_?dep(?:art(?:ement)?)?|insee_?dep(?:art(?:ement)?)?)\b", re.I)),
    # canton
    ("canton", re.compile(r"\b(?:canton|code_?canton|insee_?canton)\b", re.I)),
    # epci  (allow siren_epci; also cover insee_epci)
    ("epci", re.compile(r"\b(?:epci|code_?epci|siren_?epci|insee_?epci)\b", re.I)),
    # academie (accents removed by normalise_colname)
    ("academie", re.compile(r"\b(?:academie|aca|code_?aca(?:demie)?|insee_?aca(?:demie)?)\b", re.I)),
    # commune (use 'com' only as a whole token)
    ("com", re.compile(r"\b(?:com|commune|code_?com(?:mune)?|insee_?com(?:mune)?)\b", re.I)),
    # iris
    ("iris", re.compile(r"\b(?:iris|code_?iris|insee_?iris)\b", re.I)),
]


def spatial_level_hints_from_colname(name: str) -> list[str]:
    """Return canonical spatial levels suggested by a column name."""
    n = normalise_colname(name)
    hits_short = [key for key, pat in _SPATIAL_COLNAME_HINTS if pat.search(n)]
    # map short labels (reg/dep/com/...) to canonical keys expected in ref_sets
    # hits = [_LEVEL_ALIAS[h] for h in hits_short]
    # de-duplicate while preserving order
    seen, out = set(), []
    for h in hits_short:
        if h not in seen:
            out.append(h)
            seen.add(h)
    return out

# Spatial detector without reference
def detect_wkt_geojson_string(s: pd.Series) -> bool:
    """Detect WKT or GeoJSON stored as text."""
    if not is_string_series(s):
        return False
    sample = s.dropna().astype(str).head(200)
    wkt_pat = re.compile(r"^\s*(POINT|LINESTRING|POLYGON|MULTI\w+)\s*\(", re.I)
    geojson_pat = re.compile(r'^\s*\{.*"type"\s*:\s*"(Point|LineString|Polygon|Multi\w+)"', re.I)
    return any(wkt_pat.search(x) or geojson_pat.search(x) for x in sample)

def detect_geohash(s: pd.Series) -> bool:
    """Detect geohash strings (base32 without a,i,l,o)."""
    if not is_string_series(s):
        return False
    pat = re.compile(r"^[0123456789bcdefghjkmnpqrstuvwxyz]{5,}$")
    sample = s.dropna().astype(str).str.strip().str.lower().head(200)
    ok = sample.apply(lambda x: bool(pat.match(x)) and len(x) <= 12).mean()
    return ok >= 0.7

def detect_geometry_object(s: pd.Series) -> bool:
    """Detect geometry-like Python objects (shapely/GeoSeries)."""
    if str(getattr(s, "dtype", "")).lower() == "geometry":
        return True
    vals = s.dropna().head(20)
    for v in vals:
        name = type(v).__name__.lower()
        if hasattr(v, "__geo_interface__") or any(k in name for k in ["polygon","point","linestring","multipolygon"]):
            return True
    return False

def detect_address(s: pd.Series) -> bool:
    """Heuristic for French addresses (needs tokens + a number)."""
    if not is_string_series(s):
        return False    # aggregated address will be string
    tokens = r"(rue|avenue|av\.|bd|boulevard|chemin|route|place|impasse|allee|quai|cours)"
    sample = s.dropna().astype(str).str.lower().head(200)
    ok = sample.apply(lambda x: bool(re.search(tokens, x)) and bool(re.search(r"\d", x))).mean()
    return ok >= 0.5

def detect_latlon_pair(df: pd.DataFrame) -> Tuple[str, str] | None:
    """Detect latitude/longitude columns by names and ranges."""
    cand_lat = [c for c in df.columns if re.search(r"\b(lat|latitude|y)\b", str(c), re.I)]
    cand_lon = [c for c in df.columns if re.search(r"\b(lon|lng|longitude|x)\b", str(c), re.I)]
    for la in cand_lat:
        for lo in cand_lon:
            s_lat, s_lon = df[la], df[lo]
            if is_numeric_series(s_lat) and is_numeric_series(s_lon):
                lat_ok = (s_lat.dropna().between(-90, 90)).mean() if len(s_lat.dropna()) else 0
                lon_ok = (s_lon.dropna().between(-180, 180)).mean() if len(s_lon.dropna()) else 0
                if lat_ok >= 0.9 and lon_ok >= 0.9:
                    return (la, lo)
    return None

# Spatial detector with reference
def build_ref_sets(ref_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    """
    Expect each ref table with columns ['code','nom'].
    Keys of ref_dict are the spatial levels you want reported (e.g. 'region','departement', ...).
    """
    sets = {}
    for level, df in ref_dict.items():
        if not {"code", "nom"}.issubset(df.columns):
            raise ValueError(f"Reference '{level}' must have columns ['code','nom']")
        codes = set(norm_code(x) for x in df["code"].dropna().unique())
        names = set(norm_name(x) for x in df["nom"].dropna().unique())
        sets[level] = {"codes": codes, "names": names}
    return sets

def match_ratio(sample_values, valid_set) -> float:
    if not sample_values:
        return 0.0
    hits = sum(1 for v in sample_values if v in valid_set)
    return hits / len(sample_values)

def match_series_to_ref_levels(
    s: pd.Series,
    ref_sets: Dict[str, Dict[str, Any]],
    sample_size: int = 300,
) -> Dict[str, Any]:
    """Return best {'level','by','ratio'} or {}. Robust to int/str code forms."""
    import re
    from collections import Counter

    sample = s.dropna()
    if sample.empty:
        return {}
    # Work with strings consistently
    sample = sample.astype(str).str.strip().head(sample_size)

    # Precompute normalized name samples once
    sample_names = [norm_name(x) for x in sample]

    # For each level, learn plausible code lengths from its ref codes,
    # then zero-pad numeric samples accordingly before matching.
    level_len_candidates: Dict[str, list[int]] = {}
    for level, sets in ref_sets.items():
        codes_set = sets.get("codes", set()) or set()
        if codes_set:
            lens = [len(str(c)) for c in codes_set]
            if lens:
                cnt = Counter(lens)
                # Try up to 3 most common lengths (e.g., dep: 2 & 3; epci/iris: 9; commune: 5)
                level_len_candidates[level] = [L for L, _ in cnt.most_common(3)]
            else:
                level_len_candidates[level] = []
        else:
            level_len_candidates[level] = []

    best = {"level": None, "by": None, "ratio": 0.0}

    for level, sets in ref_sets.items():
        codes_set = sets.get("codes", set()) or set()
        names_set = sets.get("names", set()) or set()
        lens_to_try = level_len_candidates.get(level, [])

        # Build level-specific normalized code samples
        norm_codes_for_level: list[str] = []
        for x in sample:
            base = norm_code(x)  # your global canonicaliser (upper, strip, etc.)
            # If contains letters (e.g., '2A'), keep as-is
            if re.search(r"[A-Za-z]", base):
                norm_codes_for_level.append(base)
                continue
            # Purely digits: try zero-padding to the plausible lengths for this level
            cands = [base] + [base.zfill(L) for L in lens_to_try if L >= len(base)]
            # Choose the first candidate that exists in ref codes; otherwise keep the longest padded (or base)
            chosen = next((c for c in cands if c in codes_set), cands[-1])
            norm_codes_for_level.append(chosen)

        r_code = match_ratio(norm_codes_for_level, codes_set) if codes_set else 0.0
        r_name = match_ratio(sample_names, names_set) if names_set else 0.0

        if r_code >= r_name:
            r, by = r_code, "code"
        else:
            r, by = r_name, "nom"

        if r > best["ratio"]:
            best = {"level": level, "by": by, "ratio": r}

    return best if best["level"] is not None else {}

# --------------- Spatial name-based extraction (filename/colname/sheetname) -

# Code patterns aligned with FR practice (incl. 2A/2B and 971–976).
_DEP_RE = re.compile(r"\b(0[1-9]|[1-8][0-9]|9[0-5]|2a|2b|97[1-6])\b", re.I)
_REG_RE = re.compile(r"\b(0?[1-9]|1[0-3]|11|24|27|28|32|44|52|53|75|76|84|93)\b", re.I)
_COM_RE = re.compile(r"\b(\d{5}|2a\d{3}|2b\d{3})\b", re.I)
_IRIS_RE = re.compile(r"\b\d{9}\b", re.I)

def _tokenise_for_names(text: str) -> List[str]:
    """
    Produce normalized tokens from any text. Keep the full normalized string
    plus alphanumeric splits to improve recall for multiword names.
    """
    t = normalise_colname(text)
    parts = re.split(r"[^a-z0-9]+", t)
    parts = [p for p in parts if p]
    return [t] + parts

def extract_spatial_hints_from_text_with_refs(text: str, ref_sets: Dict[str, Dict[str, Any]]) -> List[dict]:
    """
    Extract spatial hints (code or nom) from arbitrary text and validate against ref_sets.
    Returns a list of dicts with level/granularity/matched_by/value/confidence/evidence.
    """
    hits: List[dict] = []
    t_norm = normalise_colname(text)

    # 1) Code matches (low-noise) + validation using ref_sets codes
    code_trials = [
        ("departement", _DEP_RE.findall(t_norm)),
        ("region",      _REG_RE.findall(t_norm)),
        ("commune",     _COM_RE.findall(t_norm)),
        ("iris",        _IRIS_RE.findall(t_norm)),
    ]
    for level, codes in code_trials:
        for c in codes:
            c_norm = norm_code(c)
            if level in ref_sets and c_norm in ref_sets[level].get("codes", set()):
                hits.append({
                    "level": level, "granularity": level, "matched_by": "code",
                    "value": c_norm, "confidence": 0.97,
                    "evidence": f"Name fallback: text contains code {c_norm}."
                })

    # 2) Name (nom) matches using ref_sets names (normalized contains match with boundary proxy)
    tokens = _tokenise_for_names(text)
    hay = " " + " ".join(tokens) + " "
    for level, sets in ref_sets.items():
        for nm in sets.get("names", set()):
            # Avoid false positives for too-short names or numeric-only tokens
            if len(nm) < 4 or nm.isdigit():
                continue
            if f" {nm} " in hay or f"_{nm}_" in hay:
                hits.append({
                    "level": level, "granularity": level, "matched_by": "nom",
                    "value": nm, "confidence": 0.95,
                    "evidence": f"Name fallback: text mentions '{nm}'."
                })

    # Deduplicate by (level, matched_by, value)
    uniq = {(h["level"], h["matched_by"], h["value"]): h for h in hits}
    return list(uniq.values())

# Generic “geo” gate (lets us try all levels with a higher bar)
_SPATIAL_GENERIC_RE = re.compile(
    r"\b(zone|territoire|secteur|perimetre|unite(?:_?geo)?|codgeo|code_?geo|geo|geog|geographique|spatial)\b",
    re.I
)

def spatial_gate_from_colname(name: str) -> dict:
    """
    Return {'has_gate': bool, 'levels': [canonical levels], 'generic': bool, 'source': 'insee|code|alias|generic'}.
    """
    n = normalise_colname(name)
    hits_short = []
    source = None

    for key, pat in _SPATIAL_COLNAME_HINTS:
        if pat.search(n):
            hits_short.append(key)
            # Source hint (optional): tag INSEE/Code hits differently for logging
            if "insee" in pat.pattern:
                source = source or "insee"
            elif "code_" in pat.pattern:
                source = source or "code"
            else:
                source = source or "alias"

    generic = bool(_SPATIAL_GENERIC_RE.search(n))

    seen, out = set(), []
    for lv in hits_short:
        if lv not in seen:
            out.append(lv); seen.add(lv)

    return {"has_gate": bool(out) or generic, "levels": out, "generic": generic, "source": source or ("generic" if generic else None)}

# --- Column-name hints for geo formats (geometry / WKT-GeoJSON / geohash) ---
_GEOFORMAT_HINTS = {
    # Typical names seen across GIS exports
    "geometry": re.compile(r"\b(the_)?geom(?:etry)?\b|\bshape\b|\bgeom_wkt\b|\bgeomjson\b", re.I),
    "wkt":      re.compile(r"\bwkt\b|well[_\- ]?known[_\- ]?text", re.I),
    "geojson":  re.compile(r"\bgeojson\b|\bgeom?_?json\b|\bjson_?geom\b", re.I),
    "geohash":  re.compile(r"\bgeo_?hash\b|\bgeohash\b|\bgh\b", re.I),
}

def geoformat_hints_from_colname(name: str) -> dict:
    """Return flags telling whether the column name suggests a geo format."""
    n = normalise_colname(name)
    return {k: bool(p.search(n)) for k, p in _GEOFORMAT_HINTS.items()}