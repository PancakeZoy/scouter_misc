import pandas as pd
import pickle
import numpy as np
from sklearn.decomposition import PCA

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
}
original_embd_path = "../data/Data_GeneEmbd/GenePT_V1.pickle"
with open(original_embd_path, "rb") as f:
    embd = pd.DataFrame(pickle.load(f)).T
    embd.rename(rename_dict, inplace=True)

pca = PCA(n_components=64)
embd_pca = pca.fit_transform(embd.values)
embd_pca_df = pd.DataFrame(embd_pca, index=embd.index)
embd_pca_dict = embd_pca_df.to_dict(orient="index")
renamed_embd64_path = "ReplaceInput/embeddings/GenePT_V1_64dim.pickle"
with open(renamed_embd64_path, "wb") as f:
    pickle.dump(embd_pca_dict, f)

embd_dict = embd.to_dict(orient="index")
renamed_embd_path = "ReplaceInput/embeddings/GenePT_V1.pickle"
with open(renamed_embd_path, "wb") as f:
    pickle.dump(embd_dict, f)

ess_perts_path = "../data/Data_GEARS/essential_all_data_pert_genes.pkl"
essential_genes = pickle.load(open(ess_perts_path, "rb"))
essential_genes = np.array([i for i in essential_genes if i in embd.index])
new_ess_perts_path = "ReplaceInput/gears_ess_perts/essential_all_data_pert_genes_embd.pkl"  # fmt: skip
with open(new_ess_perts_path, "wb") as f:
    pickle.dump(essential_genes, f)

go_path = "../data/Data_GEARS/go_essential_all/go_essential_all.csv"
go_csv = pd.read_csv(go_path)
go_csv = go_csv[(go_csv.source.isin(essential_genes)) & (go_csv.target.isin(essential_genes))]  # fmt: skip
new_go_path = "ReplaceInput/gears_ess_perts/go_essential_all_embd.csv"
go_csv.to_csv(new_go_path, index=False)
