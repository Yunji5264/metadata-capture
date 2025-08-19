import pandas as pd
import numpy as np
import geopandas as gpd
import unicodedata
import re
import os
import zipfile
from typing import Dict, Any, Tuple, List
from os import listdir
from os.path import isfile, join

from babel.util import distinct


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
    """Read CSV with various encodings and delimiters"""
    try:
        df = pd.read_csv(file_path, low_memory=False, on_bad_lines='skip')
    except:
        df = pd.read_csv(file_path, encoding='latin1', low_memory=False, on_bad_lines='skip')
    if df.shape[1] == 1:
        try:
            df = pd.read_csv(file_path, sep=';', low_memory=False, on_bad_lines='skip')
        except:
            df = pd.read_csv(file_path, encoding='latin1', sep=';', low_memory=False, on_bad_lines='skip')
    return df

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

# mapping: extension → handler function
dict_EL = {
    '.xlsx': excel_EL,
    '.csv': csv_EL,
    '.geojson': geojson_EL,
    '.zip': shapefile_EL,
    '.json': json_EL
}

# mapping: extension → data category
dict_category = {
    '.xlsx': "structured",
    '.csv': "structured",
    '.geojson': "semi-structured",
    '.zip': "semi-structured",
    '.json': "semi-structured"
}

def get_df(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in dict_EL:
        handler = dict_EL[ext]
        return handler(path), dict_category[ext]
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

# Normalisation helpers
def strip_accents(s):
    """Remove accents; normalise spaces/hyphens/apostrophes."""
    if not isinstance(s, str):
        return s
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
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
