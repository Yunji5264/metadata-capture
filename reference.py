from general_function import *

pairs = ["reg", "dep", "canton", "epci", "academie", "com", "iris"]
ref_dict = {
    p: csv_EL(f"../data/ref_spatial/{p}.csv") for p in pairs
}

LEVEL_SPATIAL = [
    "region",          # région
    "departement",     # département
    "academie",        # académie
    "arrondissement",  # arrondissement départemental
    "epci",            # établissement public de coopération intercommunale
    "commune",         # commune
    "iris",            # IRIS (Insee 9-digit sub-commune unit)
    "address",         # adresse textuelle (numéro, rue, etc.)
    "geocode",         # generic geocode ID (place ID, external code, etc.)
    "geohash",         # geohash (grid-based spatial code)
    "geometry",        # explicit geometry column
    "wkt_geojson",     # WKT/GeoJSON string
    "latlon_pair"      # latitude/longitude numeric pair
]
LEVEL_TEMPORAL = ["year", "quarter", "month", "week", "day"]