from general_function import *

def canon_code_str(v):
    """Clean code values: avoid float .0, strip spaces"""
    if pd.isna(v):
        return None
    # int / float → string
    if isinstance(v, (int, float)):
        if float(v).is_integer():
            return str(int(v))
        return str(v).rstrip("0").rstrip(".")
    # string case
    s = str(v).strip()
    # remove thousand/space separators
    s = re.sub(r"[ \u00A0\u2009\u202F]", "", s)
    # strip ".0" suffix if digit-like
    m = re.fullmatch(r"(\d+)\.0+", s)
    if m:
        return m.group(1)
    return s


path = r"C:\Users\ADMrechbay20\OneDrive\桌面\Données\carte\georef-france-iris.csv"
path2 = r"C:\Users\ADMrechbay20\OneDrive\桌面\Données\carte\communes-france-2025.csv"
df = csv_EL(path)
df2 = csv_EL(path2)

df3 = (df2[['code_insee', 'academie_code', 'academie_nom', 'canton_code', 'canton_nom']]
        .rename(columns={'code_insee':"Code Officiel Commune",
                         'academie_code': "Code Officiel Académie", 'academie_nom': "Nom Officiel Académie",
                         'canton_code': "Code Officiel Canton", 'canton_nom': "Nom Officiel Canton"}))

ref = pd.merge(df, df3, "left", on="Code Officiel Commune")

# 🔑 Clean all "Code Officiel ..." columns
for col in ref.columns:
    if col.startswith("Code Officiel"):
        ref[col] = ref[col].map(canon_code_str)

ref.to_csv(f"../data/ref_spatial/ref_spatial_complet.csv", index=False, encoding="utf-8")

pairs = {
    "Région":"reg",
    "Département":"dep",
    "Arrondissement départemental": "arr_dep",
    "EPCI":"epci",
    "Canton":"canton",
    "Commune":"com",
    "Commune / Arrondissement Municipal":"com_arr",
    "IRIS":"iris",
    "Académie":"academie"
}

ref_dict = {p: ref[[f"Code Officiel {p}", f"Nom Officiel {p}"]].drop_duplicates() for p in pairs.keys()}

ref_dict = {
    key: df.rename(columns={df.columns[0]: "code", df.columns[1]: "nom"})
    for key, df in ref_dict.items()
}

ref_dict = {
    key: df.assign(nom=df["nom"].map(strip_accents))
    for key, df in ref_dict.items()
}

# Save per-level reference files
for level, df in ref_dict.items():
    df.to_csv(f"../data/ref_spatial/{pairs.get(level)}.csv", index=False, encoding="utf-8")
