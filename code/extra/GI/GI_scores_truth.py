import scanpy as sc
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import TheilSenRegressor
from dcor import distance_correlation
from tqdm import tqdm


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


def condition_sort(x):
    return "+".join(sorted(x.split("+")))


####################################################################
data_path = "../data/Data_GEARS/norman/perturb_processed.h5ad"
adata = sc.read(data_path)
adata.obs["condition"] = adata.obs["condition"].astype(str).apply(lambda x: condition_sort(x)).astype("category")  # fmt: skip
adata.uns = {}
adata.obs.drop("condition_name", axis=1, inplace=True)
adata.var.set_index("gene_name", inplace=True)

GI_genes = np.load("genes_with_hi_mean.npy", allow_pickle=True)
GI_genes = np.intersect1d(GI_genes, adata.var_names)
adata = adata[:, GI_genes]
ctrl = adata[adata.obs.condition == "ctrl"].X.toarray().mean(axis=0).astype(np.float64)

gi_truth = pickle.load(open("GI_truth.pkl", "rb"))
del gi_truth["additive"]
gi_df = []
for gi_type, combos in gi_truth.items():
    for pert_ab in tqdm(combos, desc=gi_type):
        pert_a, pert_b = (i + "+ctrl" for i in pert_ab.split("+"))
        a = adata[adata.obs.condition == pert_a].X.toarray().mean(axis=0) - ctrl
        b = adata[adata.obs.condition == pert_b].X.toarray().mean(axis=0) - ctrl
        design = np.array([a, b]).T
        ab = adata[adata.obs.condition == pert_ab].X.toarray().mean(axis=0) - ctrl
        result = get_coeffs(design, a, b, ab)
        result["perturbation"] = pert_ab
        result["gi_type"] = gi_type
        gi_df.append(pd.DataFrame(result, index=[0]))

gi_df = pd.concat(gi_df)
gi_df.sort_values(by=["gi_type", "perturbation"], inplace=True)
gi_df.to_csv("GI_scores_truth.csv", index=False)
