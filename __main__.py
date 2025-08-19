import os
import json
import csv
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import geopandas as gpd
import pyarrow.parquet as pq

from attribute_classifier import *
from general_function import *
import argparse

# Import your worker and the multithreading helper
from multithreading_helper import collect_metadata_parallel   # <- contains collect_metadata_parallel
from yourpkg.inspectors import get_file_metadata           # <- your existing per-file inspector

data_source = r"C:\Users\ADMrechbay20\OneDrive\桌面\Données\Données\Opendata\Général\Logement\rpls2021_donnees_detaillees_geolocalisees_logement\RPLS_geoloc2021_OpenData\Open_Data\Region"


def _build_paths(root: Path, recursive: bool = True):
    """Collect file paths under the folder with allowed extensions."""
    exts = {
        ".csv", ".tsv",
        ".xlsx", ".xls", ".xlsm",
        ".json", ".geojson", ".ndjson",
        ".parquet",
        ".shp", ".gpkg",
    }
    it = root.rglob("*") if recursive else root.glob("*")
    return [str(p) for p in it if p.is_file() and p.suffix.lower() in exts]

def main():
    parser = argparse.ArgumentParser(description="Collect metadata from datasets concurrently.")
    parser.add_argument("folder", help="Root folder containing the datasets")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    parser.add_argument("--workers", type=int, default=None, help="Max worker threads (auto if omitted)")
    parser.add_argument("--out", default="metadata.ndjson", help="Output NDJSON file path")
    args = parser.parse_args()

    root = Path(args.folder)
    paths = _build_paths(root, recursive=args.recursive)
    if not paths:
        print("No matching files found.")
        return

    results = collect_metadata_parallel(paths, worker=get_file_metadata, max_workers=args.workers)

    with open(args.out, "w", encoding="utf-8") as f:
        for m in results:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {len(results)} records to: {args.out}")

if __name__ == "__main__":
    main()


