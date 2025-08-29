import pandas as pd
import geopandas as gpd
import numpy as np
import unicodedata
import re
import os
import zipfile
from typing import Dict, List, Any, Tuple, Set, Optional, Iterable, Union


# Normalisation helpers
def strip_accents(s: str):
    """Remove accents; normalise spaces/hyphens/apostrophes; fix common encoding issues."""
    if not isinstance(s, str):
        return s

    # --- Step 1: fix common mis-encodings (©, ¨e, etc.) ---
    replacements = {
        "Ã©": "é",   # common CSV bug: ma©tropole → métropole
        "Æ": "AE",
        "æ": "ae",
        "œ": "oe",  # œufs → oeufs
    }
    for bad, good in replacements.items():
        s = s.replace(bad, good)

    # --- Step 2: remove accents properly ---
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # --- Step 3: unify apostrophes and separators ---
    s = s.replace("’", "'").replace("`", "'").replace("´", "'")
    s = re.sub(r"[-_\s]+", " ", s)

    return s.strip()


def norm_name(v):
    """Normalise a display name for robust comparison."""
    if pd.isna(v):
        return ""
    return strip_accents(str(v)).lower()

def norm_code(v):
    """Normalise a code for robust comparison (uppercase, trimmed)."""
    if pd.isna(v):
        return ""
    return str(v).strip().upper()

def is_numeric_series(s):
    return pd.api.types.is_numeric_dtype(s)

def is_string_series(s):
    return pd.api.types.is_string_dtype(s) or s.dtype == object

def not_null_ratio(s):
    n = len(s)
    return float(s.notna().sum())/n if n else 0.0

def normalise_colname(name: str) -> str:
    """Lightweight colname normaliser for pattern matching."""
    # convert to string and lowercase
    s = str(name).lower()

    # remove accents using unicodedata
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # replace whitespace and hyphens with underscore
    s = re.sub(r"[\s\-]+", "_", s)

    return s

#file reader
def excel_EL(file_path):
    """Read Excel file"""
    df = pd.read_excel(file_path, header=None, nrows=1000)
    # Count rows with data
    row_counts = df.count(axis=1)
    # Check if the DataFrame is empty
    if row_counts.empty:
        raise ValueError("DataFrame is empty or contains no valid data.")
    # Find the row with the fewest null values
    max_row_count = row_counts.max()
    if max_row_count == 0:
        raise ValueError("All rows are empty.")
    # Set this row as the header and retrieve the table below it
    first_row = row_counts.idxmax() + 1
    # Read data from the title line
    df = pd.read_excel(file_path, header=first_row)
    return df

def csv_EL(file_path):
    """Read CSV/TSV with various encodings and delimiters"""
    for encoding in ["utf-8", "latin1"]:
        try:
            df = pd.read_csv(file_path, encoding=encoding, low_memory=False, on_bad_lines="skip")
        except Exception:
            continue
        if df.shape[1] > 1:
            return df
        # Try ; separator
        try:
            df = pd.read_csv(file_path, encoding=encoding, sep=";", low_memory=False, on_bad_lines="skip")
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
        # Try tab separator (TSV)
        try:
            df = pd.read_csv(file_path, encoding=encoding, sep="\t", low_memory=False, on_bad_lines="skip")
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    raise ValueError(f"Could not parse {file_path} as CSV/TSV with common delimiters.")

def geojson_EL(file_path):
    """Read GeoJSON data"""
    gdf = gpd.read_file(file_path)
    df = gdf.to_pandas()
    return df

def shapefile_EL(zip_file_path):
    """Read shape file data"""
    extracted_folder = os.getcwd()
    # Unzip the file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_folder)

    # Find Shapefile within extracted files
    shapefile_path = None
    for root, dirs, files in os.walk(extracted_folder):
        for file in files:
            if file.endswith('.shp'):
                shapefile_path = os.path.join(root, file)
                break

    if shapefile_path:
        gdf = gpd.read_file(shapefile_path)
        df = gdf.to_pandas()
    return df

def json_EL(file_path):
    """Read json data"""
    df = pd.read_json(file_path, lines=True)
    return df

def parquet_EL(file_path):
    """Read Parquet data"""
    df = pd.read_parquet(file_path, engine="pyarrow")  # or engine="fastparquet"
    return df


# mapping: extension → handler function
dict_EL = {
    '.xlsx': excel_EL,
    '.xls':  excel_EL,     # add legacy Excel
    '.csv':  csv_EL,
    '.tsv':  csv_EL,       # handled inside csv_EL with sep="\t"
    '.parquet': parquet_EL,
    '.geojson': geojson_EL,
    '.zip': shapefile_EL,  # zipped shapefiles
    '.json': json_EL,
}

# mapping: extension → data category
dict_category = {
    '.xlsx': "structured",
    '.xls':  "structured",
    '.csv':  "structured",
    '.tsv':  "structured",
    '.parquet': "structured",
    '.geojson': "semi-structured",
    '.zip': "semi-structured",
    '.json': "semi-structured",
}

def get_df(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in dict_EL:
        df = dict_EL[ext](path)
        return df, ext, dict_category[ext]
    else:
        return None, "unknown"

# Retrieve all files within the specified path
def get_all_files(file_path):
    file_list = []
    # Traverse directory
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if not file.endswith('.ini'):
                file_list.append(os.path.join(root, file))
    return file_list

def level_matches(level, granularities) -> bool:
    """Return True if `level` is present in `granularities`.
    If `level` is a tuple of aliases (e.g. ("geometry", "wkt_geojson")),
    match if ANY alias appears in `granularities`.
    """
    if isinstance(level, tuple):
        return any(alias in granularities for alias in level)
    return level in granularities


def level_name(level) -> str:
    """Normalise a level to its canonical name.
    For tuples, use the first alias as the canonical name.
    """
    return level[0] if isinstance(level, tuple) else level


def get_most_general_in_path(granularities, path) -> Union[str, None]:
    """Scan a single hierarchy path from most general to most specific.
    Return the first (i.e., most general) level that appears in `granularities`.
    If none match, return None.
    """
    for level in path:
        if level_matches(level, granularities):
            return level_name(level)
    return None

def get_most_specific_in_path(granularities, path) -> Union[str, None]:
    """Scan a single hierarchy path from most specific to most general.
    Return the first (i.e., most specific) level that appears in `granularities`.
    If none match, return None.
    """
    # Iterate in reverse so we check the most specific level first
    for level in reversed(path):
        if level_matches(level, granularities):
            return level_name(level)
    return None

def human_readable_size(num_bytes: Optional[int]) -> Optional[str]:
    """Convert bytes to a human-readable string. Return None if input is None."""
    if num_bytes is None:
        return None
    size = float(num_bytes)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"

def uncompressed_zip_size(path: str) -> Optional[int]:
    """Return total uncompressed size of a ZIP or None on failure/not a zip."""
    if not (os.path.exists(path) and path.lower().endswith(".zip")):
        return None
    try:
        with zipfile.ZipFile(path, 'r') as zf:
            return sum(info.file_size for info in zf.infolist())
    except Exception:
        return None