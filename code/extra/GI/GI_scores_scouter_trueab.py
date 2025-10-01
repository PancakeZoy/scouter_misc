import pickle
import torch
import anndata as ad
import pandas as pd
import numpy as np
import random
from scouter import Scouter, ScouterData
from sklearn.linear_model import TheilSenRegressor
from dcor import distance_correlation


def set_seeds(seed=24):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Normalize the condition name. Make "A+B" and "B+A" the same
def condition_sort(x):
    return "+".join(sorted(x.split("+")))


def get_coeffs(singles_expr, first_expr, second_expr, double_expr):
    """
    Get coefficients for GI calculation
    This function is from the GEARS package

    Args:
        singles_expr (np.array): single perturbation expression
        first_expr (np.array): first perturbation expression
        second_expr (np.array): second perturbation expression
        double_expr (np.array): double perturbation expression

    """
    results = {}
    results["ts"] = TheilSenRegressor(
        fit_intercept=False, max_subpopulation=1e5, max_iter=1000, random_state=1000
    )
    X = singles_expr
    y = double_expr
    results["ts"].fit(X, y.ravel())
    Zts = results["ts"].predict(X)
    results["c1"] = results["ts"].coef_[0]
    results["c2"] = results["ts"].coef_[1]
    results["mag"] = np.sqrt((results["c1"] ** 2 + results["c2"] ** 2))

    results["dcor"] = distance_correlation(singles_expr, double_expr)
    results["dcor_singles"] = distance_correlation(first_expr, second_expr)
    results["dcor_first"] = distance_correlation(first_expr, double_expr)
    results["dcor_second"] = distance_correlation(second_expr, double_expr)
    results["corr_fit"] = np.corrcoef(Zts.flatten(), double_expr.flatten())[0, 1]
    results["dominance"] = np.abs(np.log10(results["c1"] / results["c2"]))
    results["eq_contr"] = np.min(
        [results["dcor_first"], results["dcor_second"]]
    ) / np.max([results["dcor_first"], results["dcor_second"]])
    del results["ts"]
    return results


############################### Initialize the model #####################################
# Set seeds for reproducibility
set_seeds(24)
data_path = "../data/Data_GEARS/norman/perturb_processed.h5ad"
embd_path = "../data/Data_GeneEmbd/GenePT_V1.pickle"

# Load the processed scRNA-seq dataset as Anndata
adata = ad.read_h5ad(data_path)
adata.obs["condition"] = (
    adata.obs["condition"]
    .astype(str)
    .apply(lambda x: condition_sort(x))
    .astype("category")
)
adata.uns = {}
adata.obs.drop("condition_name", axis=1, inplace=True)

# Load the gene embedding as the dataframe, and rename its gene alias to match the Anndata
with open(embd_path, "rb") as f:
    embd = pd.DataFrame(pickle.load(f)).T
ctrl_row = pd.DataFrame([np.zeros(embd.shape[1])], columns=embd.columns, index=["ctrl"])
embd = pd.concat([ctrl_row, embd])
embd.rename(
    index={
        "MAP3K21": "KIAA1804",
        "FOXL2NB": "C3orf72",
        "RHOXF2B": "RHOXF2BB",
        "MIDEAS": "ELMSAN1",
        "CBARP": "C19orf26",
    },
    inplace=True,
)

pertdata = ScouterData(adata, embd, "condition", "gene_name")
pertdata.setup_ad("embd_index")
pertdata.gene_ranks()
pertdata.get_dropout_non_zero_genes()
pertdata.split_Train_Val_Test(if_test=False, val_ratio=0.1, seed=1)

scouter_model = Scouter(pertdata)
scouter_model.model_init()

gi_combos = pickle.load(open("GI_truth.pkl", "rb"))
del gi_combos["additive"]
combos_gi = {combo: gi_type for gi_type, combos in gi_combos.items() for combo in combos}  # fmt: skip
################################ Evaluation #################################
ctrl_data = scouter_model.ctrl_adata.copy()
g_names_all = list(ctrl_data.var.gene_name.values)
genes_path = "genes_with_hi_mean.npy"
GI_genes = np.load(genes_path, allow_pickle=True)
GI_genes = np.intersect1d(GI_genes, g_names_all)
GI_genes_idx = [g_names_all.index(i) for i in GI_genes]
ctrl = ctrl_data.X.toarray().mean(axis=0).astype(np.float64)[GI_genes_idx]

gi_df = []
for split in range(6):
    model_path = f"Scouter_model/scouter_{split}.pth"
    scouter_model.network.load_state_dict(torch.load(model_path))
    split_path = f"splits_scouter/split_dict_{split}.pkl"
    split_test = pickle.load(open(split_path, "rb"))["test"]
    split_dict = {combo: combos_gi[combo] for combo in split_test}

    for pert_ab, gi_type in split_dict.items():
        print(f"Split {split}, Perturbation: {pert_ab}")
        pert_a, pert_b = (i + "+ctrl" for i in pert_ab.split("+"))

        pred_ab = scouter_model.pred([pert_ab])[pert_ab].mean(axis=0)[GI_genes_idx]
        pred_a = adata[adata.obs.condition == pert_a].X.toarray().mean(axis=0)[GI_genes_idx]  # fmt: skip
        pred_b = adata[adata.obs.condition == pert_b].X.toarray().mean(axis=0)[GI_genes_idx]  # fmt: skip

        delta_ab = pred_ab - ctrl
        delta_a = pred_a - ctrl
        delta_b = pred_b - ctrl
        design = np.array([delta_a, delta_b]).T

        result = get_coeffs(design, delta_a, delta_b, delta_ab)
        result["perturbation"] = pert_ab
        result["gi_type"] = gi_type
        gi_df.append(pd.DataFrame(result, index=[0]))

gi_df = pd.concat(gi_df)
gi_df.sort_values(by=["gi_type", "perturbation"], inplace=True)
gi_df.to_csv("GI_scores_scouter_trueab.csv", index=False)
