from openai import OpenAI
import os
import numpy as np
import json
from tqdm import tqdm
import pickle
import scanpy as sc
import argparse

# fmt: off
########################### NCBI UniProt Summary ###########################
dixit_path = "../data/Data_GEARS/dixit/perturb_processed.h5ad"
dixit_conds = list(sc.read_h5ad(dixit_path).obs.condition.unique())
adamson_path = "../data/Data_GEARS/adamson/perturb_processed.h5ad"
adamson_conds = list(sc.read_h5ad(adamson_path).obs.condition.unique())
norman_path = "../data/Data_GEARS/norman/perturb_processed.h5ad"
norman_conds = list(sc.read_h5ad(norman_path).obs.condition.unique())
k562_path = "../data/Data_GEARS/replogle_k562_essential/perturb_processed.h5ad"
k562_conds = list(sc.read_h5ad(k562_path).obs.condition.unique())
rpe1_path = "../data/Data_GEARS/replogle_rpe1_essential/perturb_processed.h5ad"
rpe1_conds = list(sc.read_h5ad(rpe1_path).obs.condition.unique())

all_conds = sorted(set(dixit_conds + adamson_conds + norman_conds + k562_conds + rpe1_conds))
all_conds.remove("ctrl")
all_conds = sorted(set(np.array([i.split('+') for i in all_conds]).flatten()))
all_conds.remove("ctrl")
rename_dict = {
    "AARS1": "AARS",
    "CENATAC": "CCDC84",
    "POLR1G": "CD3EAP",
    "DARS1": "DARS",
    "EPRS1": "EPRS",
    "HARS1": "HARS",
    "IARS1": "IARS",
    "KARS1": "KARS",
    "LARS1": "LARS",
    "MARS1": "MARS",
    "QARS1": "QARS",
    "RARS1": "RARS",
    "SARS1": "SARS",
    "TARS1": "TARS",
    "POLR1F": "TWISTNB",
    "VARS1": "VARS",
    "POLR1H": "ZNRD1",
    "MAP3K21": "KIAA1804",
    "FOXL2NB": "C3orf72",
    "RHOXF2B": "RHOXF2BB",
    "MIDEAS": "ELMSAN1",
    "CBARP": "C19orf26",
    "CARS1": "CARS",
    "SRPRA": "SRPR",
    "PRELID3B": "SLMO2",
    "ZZZ3": "AC118549.1",
    "ALG1": "ALG1L",
    "ATP5F1B": "ATP5B",
    "MACIR": "C5orf30",
    "H2AZ1": "H2AFZ",
    "NARS1": "NARS",
    "YARS1": "YARS",
}
with open("/Users/pancake/Downloads/GenePT_emebdding_v2/NCBI_UniProt_summary_of_genes.json", "r") as file:
    summary = json.load(file)
summary = {rename_dict.get(k, k): v for k, v in summary.items()}
summary = {k: v for k, v in summary.items() if k in all_conds}

with open("GeneSummary.json", "w") as file:
    json.dump(summary, file)
