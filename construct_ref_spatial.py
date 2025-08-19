from general_function import *

path = r"C:\Users\ADMrechbay20\OneDrive\桌面\Données\carte\communes-france-2025.csv"
path_iris = r"C:\Users\ADMrechbay20\OneDrive\桌面\Données\carte\georef-france-iris.csv"
ref = csv_EL(path)
ref_iris = csv_EL(path_iris)

pairs = ["reg", "dep", "canton", "epci", "academie"]

ref_dict = {
    "iris": ref_iris[["Code Officiel IRIS","Nom Officiel IRIS"]].drop_duplicates(),
    "com": ref[['code_insee', 'nom_standard']].drop_duplicates(),
    **{p: ref[[f"{p}_code", f"{p}_nom"]].drop_duplicates() for p in pairs}
}

ref_dict = {
    key: df.rename(columns={df.columns[0]: "code", df.columns[1]: "nom"})
    for key, df in ref_dict.items()
}

ref_dict = {
    key: df.assign(nom=df["nom"].map(strip_accents))
    for key, df in ref_dict.items()
}

# ref_dict["com"]

for level, df in ref_dict.items():
    df.to_csv(f"../data/ref_spatial/{level}.csv", index=False, encoding="utf-8")