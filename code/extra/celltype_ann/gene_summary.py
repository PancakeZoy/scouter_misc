import numpy as np
import json
import scanpy as sc

# fmt: off
########################### NCBI UniProt Summary ###########################
dixit_path = "../../data/Data_GEARS/dixit/perturb_processed.h5ad"
dixit_conds = list(sc.read_h5ad(dixit_path).obs.condition.unique())
adamson_path = "../../data/Data_GEARS/adamson/perturb_processed.h5ad"
adamson_conds = list(sc.read_h5ad(adamson_path).obs.condition.unique())
norman_path = "../../data/Data_GEARS/norman/perturb_processed.h5ad"
norman_conds = list(sc.read_h5ad(norman_path).obs.condition.unique())

all_conds = sorted(set(dixit_conds + adamson_conds + norman_conds))
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
with open("/Users/pancake/Downloads/GenePT_emebdding_v2/NCBI_summary_of_genes.json", "r") as file:
    summary = json.load(file)
summary = {rename_dict.get(k, k): v for k, v in summary.items()}
summary = {k: v for k, v in summary.items() if k in all_conds}
with open("GeneSummary.json", "w") as file:
    json.dump(summary, file)

k562_text = "K-562 are lymphoblast cells isolated from the bone marrow of a 53-year-old chronic myelogenous leukemia patient. The K-562 cell line is widely used in immune system disorder and immunology research."
summary_ann = {k: v+"\n"+k562_text for k, v in summary.items()}
with open("GeneSummary_ann.json", "w") as file:
    json.dump(summary_ann, file)
