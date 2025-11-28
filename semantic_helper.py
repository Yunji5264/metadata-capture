import os
import json
from openai import OpenAI
from reference import *

# Initialise OpenAI client.
# It will look for the API key in the environment variable OPENAI_API_KEY.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def semantic_helper(
    df: pd.DataFrame,
    model: str = "gpt-5-mini",
    sample_rows: int = 10,
    max_cols_per_batch: int = 30
) -> pd.DataFrame:
    """
    Enhanced semantic helper:
    - Takes a DataFrame
    - Samples first N rows
    - Splits into batches if too many columns
    - Calls GPT model for each batch
    - Returns concatenated classification results as DataFrame
    """

    all_results = []

    # Split columns into chunks
    columns = list(df.columns)
    for i in range(0, len(columns), max_cols_per_batch):
        batch_cols = columns[i:i + max_cols_per_batch]
        batch_df = df[batch_cols].head(sample_rows)
        samples = batch_df.to_dict(orient="list")

        # The thematic hierarchy dictionary (must be defined in your reference module or elsewhere)
        # Example: THEME_FOLDER_STRUCTURE = {"Well-being": {"Environment": ["Air quality", "Green space"]}}
        # Make sure it is a Python dictionary
        prompt = f"""
        You are a rigorous data steward. Classify each column using header names and sample values.

        Allowed classes (mutually exclusive; exactly one must be True):
        - Spatial: encodes a location (latitude/longitude, X/Y, address, admin code, region/department names or codes).
        - Temporal: encodes time or period (year, date, month, quarter, week, YYYY, YYYY-MM, timestamps).
        - Indicator: a measured variable used for analysis/monitoring (counts, rates, ratios, scores, categories, yes/no flags).
        - Other information: ONLY if it is NOT spatial, NOT temporal, and NOT an indicator (e.g., labels, IDs, free text, entity attributes).

        Tasks per column:
        1) Explain the meaning (one short sentence).
        2) Set is_spatial (bool).
        3) Set is_temporal (bool).
        4) Set is_indicator (bool).
        5) Set is_other_information (bool) = True only if the first three are all False.
        6) If is_indicator is True, set indicator_type to "Quantitative" or "Qualitative"; otherwise null.
        7) If is_indicator is True, assign a theme from the given hierarchy.
        8) If is_other_information is True:
           - If it is a supplementary field for Spatial or Temporal (e.g., geocoding precision, time parsing notes), set thematic_path = null.
           - OTHERWISE, you MUST assign a theme from the same hierarchy (full path from root to the most specific applicable leaf).
           - In other words, for non-spatial/non-temporal auxiliary fields, thematic_path is REQUIRED.

        Thematic hierarchy (use exactly as provided; do NOT invent nodes):
        {THEME_FOLDER_STRUCTURE}

        Column names and sample data (a few values each):
        {samples}

        STRICT formatting of thematic_path:
        - MUST start with "Well-being" (exact spelling and casing).
        - MUST be the FULL PATH from the root to the most specific applicable leaf.
        - Join levels with " > " and NO trailing/leading spaces.
        - Use ONLY nodes that exist in the provided hierarchy, preserving order and spelling.
        - If you truly cannot assign a valid path beginning with "Well-being", return null. This null is NOT allowed when is_other_information is True for a non-spatial/non-temporal auxiliary field.

        Decision steps (apply in order for each column):
        A) Detect class:
           - If matches location patterns (geo codes/names, lon/lat, address): Spatial.
           - Else if matches time patterns (year, date, month, etc.): Temporal.
           - Else if looks like a measure (numeric or categorical outcome used for analysis): Indicator.
           - Else: Other information.
        B) Thematic mapping:
           - If Indicator → REQUIRED to map to the deepest applicable leaf in "Well-being ...".
           - If Other information:
               * If clearly auxiliary to Spatial/Temporal (e.g., geocoding quality, date parsing source), thematic_path = null.
               * Else REQUIRED to map to the deepest applicable leaf in "Well-being ...".
           - If Spatial or Temporal → thematic_path = null.

        Common mappings (examples; use only if relevant and the node exists in the provided hierarchy):
        - Healthcare professional specialty → Well-being > Current Well-being > Health > Health systems & services > Workforce & resources
        - Hospital admission date → Well-being > Current Well-being > Health > Access to care
        - Chronic disease flag → Well-being > Current Well-being > Health > Physical health > Disease burden
        - Self-reported anxiety/depression score → Well-being > Current Well-being > Health > Mental health > Mental state
        
        - Education program type → Well-being > Current Well-being > Education & Skills > Skills & learning
        - Literacy test result → Well-being > Current Well-being > Education & Skills > Educational outcomes > Performance > Literacy
        
        - Household disposable income → Well-being > Current Well-being > Income & Wealth > Income > Household income
        - Net wealth of household → Well-being > Current Well-being > Income & Wealth > Wealth > Net wealth
        
        - Employment contract type → Well-being > Current Well-being > Jobs & Earnings > Job quality > Stability
        - Weekly working hours → Well-being > Current Well-being > Jobs & Earnings > Job quality > Working conditions
        
        - Housing occupancy status → Well-being > Current Well-being > Housing > Housing conditions
        - Housing cost burden → Well-being > Current Well-being > Housing > Housing affordability > Cost burden
        
        - PM2.5 concentration → Well-being > Current Well-being > Environment Quality > Environmental exposure > Air quality
        - Green space availability → Well-being > Current Well-being > Environment Quality > Perceptions & access > Green space accessibility
        
        - Recorded crime rate → Well-being > Current Well-being > Safety > Personal safety > Crime incidence
        - Road traffic injury counts → Well-being > Current Well-being > Safety > Road safety > Traffic injuries
        
        - Voter turnout percentage → Well-being > Current Well-being > Civic Engagement & Governance > Participation > Voter turnout
        - Trust in government score → Well-being > Current Well-being > Civic Engagement & Governance > Trust & satisfaction > Institutional trust
        
        - Number of close friends reported → Well-being > Current Well-being > Social Connections > Social support > Reliance network
        - Participation in community events → Well-being > Current Well-being > Social Connections > Social participation > Community participation
        
        - Life satisfaction score → Well-being > Current Well-being > Subjective Well-being > Life satisfaction
        - Positive affect index → Well-being > Current Well-being > Subjective Well-being > Affective balance > Positive affect
        
        - Average commuting time → Well-being > Current Well-being > Work-life Balance > Commuting time
        - Hours spent on unpaid work → Well-being > Current Well-being > Work-life Balance > Unpaid work
        
        - Religious affiliation → Well-being > Current Well-being > Spirituality / Religion / Personal Beliefs
        
        ---
        
        - Protected forest area share → Well-being > Resources for Future Well-being > Natural Capital > Ecosystems & biodiversity > Forest cover
        - Renewable energy share of total → Well-being > Resources for Future Well-being > Natural Capital > Climate & sustainability > Renewable energy
        
        - Child development index → Well-being > Resources for Future Well-being > Human Capital > Health stock > Child development
        - Adult skills survey result → Well-being > Resources for Future Well-being > Human Capital > Education & skills stock > Adult skills
        
        - Interpersonal trust index → Well-being > Resources for Future Well-being > Social Capital > Trust & norms > Interpersonal trust
        - Gender equality measure → Well-being > Resources for Future Well-being > Social Capital > Inclusion & cohesion > Gender equality
        
        - Public infrastructure investment → Well-being > Resources for Future Well-being > Economic & Produced Capital > Infrastructure & innovation > Fixed capital
        - Adjusted net savings → Well-being > Resources for Future Well-being > Economic & Produced Capital > Wealth sustainability > Adjusted savings


        Quality checks (HARD FAIL if violated):
        - Exactly ONE of [is_spatial, is_temporal, is_indicator, is_other_information] is True.
        - If is_indicator is False → indicator_type must be null.
        - thematic_path must either be a valid full path starting with "Well-being" or null.
        - If is_indicator is True → thematic_path is REQUIRED (not null).
        - If is_other_information is True and the field is NOT a Spatial/Temporal auxiliary → thematic_path is REQUIRED (not null).

        Return JSON strictly as:
        {{
          "columns": [
            {{
              "column_name": "str",
              "meaning": "str",
              "is_spatial": true/false,
              "is_temporal": true/false,
              "is_indicator": true/false,
              "is_other_information": true/false,
              "indicator_type": "Quantitative" | "Qualitative" | null,
              "thematic_path": "str" | null
            }}
          ]
        }}
        """

        # Call OpenAI Chat Completions API
        resp = client.chat.completions.create(
            model=model,  # Replace with the model you have access to
            messages=[{"role": "user", "content": prompt}],
            # temperature=0,
            response_format={"type": "json_object"}  # Enforce valid JSON output
        )

        # Parse JSON
        content = resp.choices[0].message.content
        result = json.loads(content)
        batch_res = pd.DataFrame(result["columns"])
        all_results.append(batch_res)

    # Concatenate all batch results
    out_df = pd.concat(all_results, ignore_index=True)
    return out_df

# semantic_helper.py
# -*- coding: utf-8 -*-

# """
# Semantic classification using a LOCAL Ollama model (no OpenAI API required).
#
# What this script does:
# 1) Load a dataset via your existing `get_df` from reference.py
# 2) Sample a few rows per column to build a compact prompt
# 3) Ask a local Ollama model (e.g., llama3/mistral) to:
#    - explain column meaning,
#    - detect spatial/temporal,
#    - detect indicator and its type (quantitative/qualitative),
#    - map indicator to the closest thematic hierarchy path (root → leaf)
#      using your `THEME_FOLDER_STRUCTURE` dictionary.
# 4) Parse the JSON result and return a pandas DataFrame.
#
# Notes:
# - Make sure Ollama is installed and running: https://ollama.com/
# - Pull a model before running, e.g.:
#     ollama pull llama3
# - If JSON parsing fails, a fallback using your detectors is applied
#   (spatial/temporal/indicator modules).
# """
#
# import json
# import os
# import sys
# from typing import Any, Dict, List
#
# import pandas as pd
# import ollama  # Local LLM via Ollama
#
# # Project-specific helpers and structures
# from reference import *                 # get_df, THEME_FOLDER_STRUCTURE, etc.
# from attribute_classifier import *      # classify_attributes (your detectors wrapper)
# from general_function import *          # norm_name, normalise_colname, etc.
#
#
# # -----------------------------
# # Config
# # -----------------------------
# MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3")  # e.g., "llama3", "mistral", "qwen2"
# SAMPLE_ROWS = int(os.getenv("SAMPLE_ROWS", 10))   # keep prompts small
# MAX_RETRY = 2                                     # how many times to retry on non-JSON
#
#
# # -----------------------------
# # Fallback (your detectors)
# # -----------------------------
# def _iter_theme_paths(theme_dict: Dict[str, Any], prefix: List[str] = None):
#     """Yield (path_list, leaf_text) for all nodes in a nested dict/list structure."""
#     prefix = prefix or []
#     if isinstance(theme_dict, dict):
#         for k, v in theme_dict.items():
#             yield from _iter_theme_paths(v, prefix + [k])
#     elif isinstance(theme_dict, list):
#         for v in theme_dict:
#             if isinstance(v, (dict, list)):
#                 yield from _iter_theme_paths(v, prefix)
#             else:
#                 yield (prefix + [str(v)], str(v))
#     else:
#         yield (prefix + [str(theme_dict)], str(theme_dict))
#
# def _best_theme_path_for_indicator(column_name: str, meaning: str, THEME_FOLDER_STRUCTURE: Dict[str, Any]) -> str | None:
#     """
#     Use your existing norm_name() to do light keyword containment for thematic path mapping.
#     """
#     cand = norm_name(f"{column_name} {meaning}")
#     best = None
#     best_score = 0
#     for path_list, leaf in _iter_theme_paths(THEME_FOLDER_STRUCTURE):
#         leaf_norm = norm_name(leaf)
#         score = 0
#         if leaf_norm and leaf_norm in cand:
#             score = len(leaf_norm)  # prefer longer matches
#         if score > best_score:
#             best_score = score
#             best = " > ".join(path_list)
#     return best if best_score > 0 else None
#
# def fallback_with_detectors(
#     df: pd.DataFrame,
#     THEME_FOLDER_STRUCTURE: Dict[str, Any],
#     *,
#     ref_dict: Dict[str, pd.DataFrame] | None = None,
#     min_ratio: float = 0.7,
#     sample_size: int = 300,
#     include_city_zip_in_address: bool = False,
#     address_colname: str = "__address__",
#     filename: str | None = None,
#     sheet_names: List[str] | None = None,
#     require_name_hint_for_geoformats: bool = True,
# ) -> pd.DataFrame:
#     """
#     Use your spatial/temporal/indicator detectors to build the same DataFrame
#     schema as the LLM output:
#       column_name, meaning, is_spatial, is_temporal, is_indicator, indicator_type, thematic_path
#     """
#     results = classify_attributes(
#         df,
#         ref_dict=ref_dict,
#         min_ratio=min_ratio,
#         sample_size=sample_size,
#         include_city_zip_in_address=include_city_zip_in_address,
#         address_colname=address_colname,
#         filename=filename,
#         sheet_names=sheet_names,
#         require_name_hint_for_geoformats=require_name_hint_for_geoformats,
#     )
#
#     # Build lookups
#     spatial_cols: Dict[str, Dict[str, Any]] = {}
#     for item in results.get("spatial", []):
#         for c in item.get("columns", []):
#             spatial_cols[c] = item
#
#     temporal_cols: Dict[str, Dict[str, Any]] = {}
#     for item in results.get("temporal", []):
#         c = item.get("column")
#         if c:
#             temporal_cols[c] = item
#
#     indicator_cols: Dict[str, Dict[str, Any]] = {}
#     for item in results.get("indicators", []):
#         c = item.get("column")
#         if c:
#             indicator_cols[c] = item
#
#     rows: List[Dict[str, Any]] = []
#     for col in df.columns:
#         is_spatial = col in spatial_cols
#         is_temporal = col in temporal_cols
#         is_indicator = col in indicator_cols
#
#         # Meaning from detectors
#         if is_spatial:
#             gran = spatial_cols[col].get("granularity")
#             meaning = f"Spatial field ({gran})" if gran else "Spatial field"
#         elif is_temporal:
#             gran = temporal_cols[col].get("granularity")
#             meaning = f"Temporal field ({gran})" if gran else "Temporal field"
#         elif is_indicator:
#             meta = indicator_cols[col]
#             meaning = meta.get("explanation") or "Indicator"
#         else:
#             meaning = "Unclassified field"
#
#         # Indicator type normalisation
#         indicator_type = None
#         if is_indicator:
#             t = indicator_cols[col].get("type") or indicator_cols[col].get("indicator_type")
#             if t:
#                 t_norm = str(t).lower()
#                 if any(k in t_norm for k in ("quant", "numeric", "continuous")):
#                     indicator_type = "Quantitative"
#                 elif any(k in t_norm for k in ("qual", "categor", "boolean", "binary")):
#                     indicator_type = "Qualitative"
#
#         # Thematic mapping (heuristic keywords; replace if you have a stronger matcher)
#         thematic_path = _best_theme_path_for_indicator(col, meaning, THEME_FOLDER_STRUCTURE) if is_indicator else None
#
#         rows.append({
#             "column_name": col,
#             "meaning": meaning,
#             "is_spatial": bool(is_spatial),
#             "is_temporal": bool(is_temporal),
#             "is_indicator": bool(is_indicator),
#             "indicator_type": indicator_type,
#             "thematic_path": thematic_path
#         })
#
#     return pd.DataFrame(rows, columns=[
#         "column_name", "meaning", "is_spatial", "is_temporal",
#         "is_indicator", "indicator_type", "thematic_path"
#     ])
#
#
# # -----------------------------
# # Prompt builder
# # -----------------------------
# def build_prompt(samples: Dict[str, List[Any]], theme_dict: Dict[str, Any]) -> str:
#     """Build a strict-instruction prompt to encourage JSON output."""
#     return f"""
# You are a data steward. Classify each column based on header names and sample values.
#
# Requirements:
# 1. Explain the meaning
# 2. Determine whether it is spatial information
# 3. Determine whether it is temporal information
# 4. Determine whether it is an indicator
# 5. If it is an indicator, decide whether it is quantitative or qualitative
# 6. If it is an indicator, assign it to the closest thematic hierarchy path (root → leaf)
#    using the following thematic hierarchy:
#
# {json.dumps(theme_dict, ensure_ascii=False, indent=2)}
#
# Column names and sample data (up to a few values each):
# {json.dumps(samples, ensure_ascii=False, indent=2)}
#
# Return the result STRICTLY as a single JSON object with the following schema:
# {{
#   "columns": [
#     {{
#       "column_name": "str",
#       "meaning": "str",
#       "is_spatial": true or false,
#       "is_temporal": true or false,
#       "is_indicator": true or false,
#       "indicator_type": "Quantitative" or "Qualitative" or null,
#       "thematic_path": "str or null"
#     }}
#   ]
# }}
#
# Rules:
# - Output only valid JSON. Do not include any explanations outside the JSON.
# - For each input column, return exactly one object in "columns".
# """.strip()
#
#
# # -----------------------------
# # Ollama call
# # -----------------------------
# def ask_ollama(prompt: str, model: str = MODEL_NAME) -> str:
#     """Call a local Ollama model with a chat-style interface and return the text."""
#     reply = ollama.chat(
#         model=model,
#         messages=[{"role": "user", "content": prompt}],
#         options={
#             "temperature": 0.1,
#             "num_predict": 1024
#         }
#     )
#     return reply["message"]["content"]
#
# def ask_ollama_streaming(prompt: str, model: str) -> str:
#     """
#     Stream tokens from Ollama and print basic live metrics (chars/sec etc.).
#     Returns the full response text.
#     """
#     t0 = time.time()
#     out = []
#     chars = 0
#
#     # stream=True gives incremental chunks
#     for chunk in ollama.generate(model=model, prompt=prompt, stream=True,
#                                  options={"temperature": 0.1, "num_predict": 256, "num_ctx": 2048}):
#         # Each chunk has 'response' when new text arrives; 'done' at the end.
#         piece = chunk.get("response", "")
#         if piece:
#             print(piece, end="", flush=True)     # live print like a terminal stream
#             out.append(piece)
#             chars += len(piece)
#
#         if chunk.get("done"):
#             # Some models return extra timing/stats at the end — use if available
#             print()  # newline after the stream
#             t1 = time.time()
#             dt = max(1e-6, t1 - t0)
#             print(f"[MONITOR] Elapsed: {dt:.2f}s | Chars: {chars} | Chars/sec: {chars/dt:.1f}")
#             # Optional extra stats if present
#             for k in ("total_duration", "load_duration", "eval_count", "eval_duration"):
#                 if k in chunk:
#                     print(f"[MONITOR] {k}: {chunk[k]}")
#             break
#
#     return "".join(out)
# # -----------------------------
# # Main pipeline
# # -----------------------------
# def semantic_analysis(file_path) -> pd.DataFrame:
#     """Main entry: returns a DataFrame with the classification results."""
#     # Path from argv or default
#     if len(sys.argv) >= 2:
#         path = sys.argv[1]
#     else:
#         path = file_path
#
#     # Load data
#     df, dataset_type = get_df(path)
#
#     # Build samples
#     samples = df.head(SAMPLE_ROWS).to_dict(orient="list")
#
#     # Build prompt
#     prompt = build_prompt(samples, THEME_FOLDER_STRUCTURE)
#
#     # Ask Ollama with lightweight retry
#     last_error = None
#     for _ in range(MAX_RETRY):
#         try:
#             content = ask_ollama_streaming(prompt, MODEL_NAME)
#             # Strip possible code fences
#             text = content.strip()
#             if text.startswith("```"):
#                 text = text.strip("`")
#                 i = text.find("\n")
#                 if i != -1:
#                     text = text[i + 1 :].strip()
#             result = json.loads(text)
#             if "columns" in result and isinstance(result["columns"], list):
#                 out_df = pd.DataFrame(result["columns"])
#                 print(out_df)
#                 return out_df
#             else:
#                 last_error = ValueError("JSON missing 'columns' list.")
#         except Exception as e:
#             last_error = e
#             # Optional: print a short excerpt to help debugging
#             try:
#                 snippet = content[:200].replace("\n", " ")
#                 print(f"[DEBUG] Model output head: {snippet}", file=sys.stderr)
#             except Exception:
#                 pass
#             continue
#
#     # Fallback to your detectors
#     print(f"[WARN] Falling back to detector-based classification due to JSON error: {last_error}", file=sys.stderr)
#     out_df = fallback_with_detectors(
#         df,
#         THEME_FOLDER_STRUCTURE,
#         ref_dict=None,                      # pass your reference dict here if available
#         min_ratio=0.7,
#         sample_size=300,
#         include_city_zip_in_address=False,
#         address_colname="__address__",
#         filename=None,
#         sheet_names=None,
#         require_name_hint_for_geoformats=True,
#     )
#     print(out_df)
#     return out_df






