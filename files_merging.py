from general_function import *

def is_leaf_folder(folder):
    """Return True if the folder contains no subdirectories."""
    for root, subdirs, files in os.walk(folder):
        # Only check the top level
        if root == folder:
            return len(subdirs) == 0
        break
    return True

def list_files(folder):
    """List non-hidden files in the folder (top level only)."""
    return [
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and not f.startswith(".")
    ]

def check_df_structures(dfs, check_order=True):
    """
    Check whether a list of DataFrames have the same structure.

    Parameters
    ----------
    dfs : list[pd.DataFrame]
        List of DataFrames to compare.
    check_order : bool, default True
        If True, require same column order.
        If False, only require same set of column names.

    Returns
    -------
    bool, dict
        (True/False, diagnostics information)
    """
    if not dfs:
        return False

    # Take the first DataFrame as reference
    ref = dfs[0]
    if check_order:
        ref_cols = list(ref.columns)
    else:
        ref_cols = set(ref.columns)

    problems = []
    for i, df in enumerate(dfs[1:], start=1):
        cols = list(df.columns) if check_order else set(df.columns)

        if cols != ref_cols:
            problems.append((i, "Columns differ", cols))

    if problems:
        return False
    else:
        return True

# multi-file folder merging

def merge_if_consistent(files):
    file_types = {os.path.splitext(f)[1] for f in files}
    if len(file_types) == 1:
        file_type = file_types[0]
        dfs = [dict_EL[file_type](f) for f in files]
        if check_df_structures(dfs):
            df_all = pd.concat(dfs, ignore_index=True)
            return df_all
    return None





