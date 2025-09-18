import pandas as pd

from attribute_classifier import *
from general_function import *
from reference import *
from semantic_helper import semantic_helper
from spatial_detector import *
from scope_detector import *
from granularity_detector import *
from metadata_selector import *
from metadata_output import *
import tiktoken

path = r"C:\Users\ADMrechbay20\PycharmProjects\metadata capture\data\medecins.csv"

res = construct_dataset(path)
print(res[1])



# semantic_res = pd.read_json("../ref/ref_semantic/rpls2021_geolocalise_OD_REG11.csv.json")
# df = csv_EL(path)
#
# res = classify_attributes_with_semantic_helper(df, semantic_res)


# df = kml_EL(path)
# print(df)

# # samples = df.head(10)
# # geom_cols = find_geometry_columns(samples)
# # samples_for_sem = samples.drop(columns=geom_cols)

#
# enc = tiktoken.encoding_for_model("gpt-5-mini")
# text = samples_for_sem.to_json(orient="records", force_ascii=False)
# tokens = len(enc.encode(text))
# print("Token count:", tokens)

# dataset = construct_dataset(path)
# print(dataset.to_dict())

# meta_dict, meta_entry, perf_entry = process_file(path)


# result = list(iter_data_files("../data",DEFAULT_EXTS))
# print(result)



# df, ext, dataset_type = get_df(path)
# # df2 = df[['REG']]
#
# samples = df.head(10)
# #
# semantic_res = semantic_helper(samples)
# #
# semantic_res.to_json("../test/example_semantic.json", orient="records", force_ascii=False, indent=2)
# #
# semantic_res = pd.read_json("../test/example_semantic.json", orient="records")
# result = classify_attributes_with_semantic_helper(df,semantic_res)
# print(result)

#
# # print(df.columns)
#
# # print(df2)
#
# # dep = df['DEP'].astype(str).str.strip()
# # ref = ref_dict['dep']['code'].astype(str).str.strip()
# #
# # semantic_res = (dep[~dep.isin(ref)]       # WHERE DEP NOT IN (...)
# #          .dropna()               # 去掉 NaN（NaN 不在任何集合里）
# #          .drop_duplicates())     # DISTINCT
# #
# # print(res)
#
# # result = classify_attributes(df2, ref_dict)
# # with pd.option_context("display.max_columns", None):
# #     print(res)
#

#
# # from semantic_helper import *
# #
# # semantic_analysis(path)

# # 1) Contains commune and iris → expect ["commune"]
# print(get_granularity({"commune", "iris"}, HIER["spatial"]))
#
# # 2) Contains epci, departement, commune, region → expect ["region", "epci"]
# print(get_granularity({"epci", "departement", "commune", "region"}, HIER["spatial"]))
#
# # 3) Contains departement, commune, region → expect ["region"]
# print(get_granularity({"departement", "commune", "region"}, HIER["spatial"]))

