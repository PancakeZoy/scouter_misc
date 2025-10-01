from gears import PertData, GEARS
import torch
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import TheilSenRegressor
from dcor import distance_correlation


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


# fmt: off
pert_data = PertData("../data/Data_GEARS")
pert_data.load(data_name="norman")
pert_data.prepare_split(split="no_test", seed=1)
pert_data.get_dataloader(batch_size=32, test_batch_size=32)

gi_combos = pickle.load(open("GI_truth.pkl", "rb"))
del gi_combos["additive"]
combos_gi = {combo: gi_type for gi_type, combos in gi_combos.items() for combo in combos}

adata = pert_data.adata.copy()
ctrl_data = adata[adata.obs.condition == 'ctrl'].copy()
g_names_all = list(ctrl_data.var.gene_name.values)
genes_path = "genes_with_hi_mean.npy"
GI_genes = np.load(genes_path, allow_pickle=True)
GI_genes = np.intersect1d(GI_genes, g_names_all)
GI_genes_idx = [g_names_all.index(i) for i in GI_genes]
ctrl = ctrl_data.X.toarray().mean(axis=0).astype(np.float64)[GI_genes_idx]

gi_df = []
for split in range(6):
    split_path = f"splits_gears/split_dict_{split}.pkl"
    split_test = pickle.load(open(split_path, 'rb'))['test']
    model_path = f"GEARS_model/split_{split}"

    gears_model = GEARS(
        pert_data,
        device="cuda" if torch.cuda.is_available() else "cpu",
        weight_bias_track=False,
        proj_name="norman",
        exp_name="GI_GEARS",
    )
    gears_model.load_pretrained(model_path)
    
    for pert_ab in split_test:
        print(f"Split {split}, Perturbation: {pert_ab}")
        if pert_ab not in combos_gi.keys():
            pert_both = "+".join(pert_ab.split("+")[::-1])
        else:
            pert_both = pert_ab
        pert_a, pert_b = pert_both.split("+")
        pred_ab = gears_model.predict([[pert_a, pert_b]])['_'.join([pert_a, pert_b])]
        pred_a = adata[adata.obs.condition == pert_a+'+ctrl'].X.toarray().mean(axis=0)
        pred_b = adata[adata.obs.condition == pert_b+'+ctrl'].X.toarray().mean(axis=0)
        
        delta_ab = pred_ab[GI_genes_idx] - ctrl
        delta_a = pred_a[GI_genes_idx] - ctrl
        delta_b = pred_b[GI_genes_idx] - ctrl
        design = np.array([delta_a, delta_b]).T
        
        result = get_coeffs(design, delta_a, delta_b, delta_ab)
        result["perturbation"] = pert_both
        result["gi_type"] = combos_gi[pert_both]
        gi_df.append(pd.DataFrame(result, index=[0]))
        
gi_df = pd.concat(gi_df)
gi_df.sort_values(by=["gi_type", "perturbation"], inplace=True)
gi_df.to_csv("GI_scores_gears_trueab.csv", index=False)
