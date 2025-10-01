import scanpy as sc
import pickle


def condition_sort(x):
    return "+".join(sorted(x.split("+")))


def list_to_combo(ls, all_combos):
    combos = []
    for i in ls:
        if "+".join(i) in all_combos:
            combos.append("+".join(i))
        elif i[1] + "+" + i[0] in all_combos:
            combos.append(i[1] + "+" + i[0])
    missing = [i for i in combos if i not in all_combos]
    if len(missing) > 0:
        raise ValueError(f"Some combinations are not found: {missing}")

    return combos


data_path = "../data/Data_GEARS/norman/perturb_processed.h5ad"
adata = sc.read(data_path)
adata.obs["condition"] = adata.obs["condition"].astype(str).apply(lambda x: condition_sort(x)).astype("category")  # fmt: skip
adata.uns = {}
adata.obs.drop("condition_name", axis=1, inplace=True)
all_combos = list(adata.obs[~adata.obs.condition.str.contains("ctrl")].condition.unique())  # fmt: skip
############################ Synergy ############################
synergy = [
    ["CNN1", "UBASH3A"],
    ["ETS2", "MAP7D1"],
    ["FEV", "CBFA2T3"],
    ["FEV", "ISL2"],
    ["FEV", "MAP7D1"],
    ["PTPN12", "UBASH3A"],
    ["CBL", "CNN1"],
    ["CBL", "PTPN12"],
    ["CBL", "PTPN9"],
    ["CBL", "UBASH3B"],
    ["FOXA3", "FOXL2"],
    ["FOXA3", "HOXB9"],
    ["FOXL2", "HOXB9"],
    ["UBASH3B", "CNN1"],
    ["UBASH3B", "PTPN12"],
    ["UBASH3B", "PTPN9"],
    ["UBASH3B", "ZBTB25"],
    ["AHR", "FEV"],
    ["DUSP9", "SNAI1"],
    ["FOXA1", "FOXF1"],
    ["FOXA1", "FOXL2"],
    ["FOXA1", "HOXB9"],
    ["FOXF1", "FOXL2"],
    ["FOXF1", "HOXB9"],
    ["FOXL2", "MEIS1"],
    ["IGDCC3", "ZBTB25"],
    ["POU3F2", "CBFA2T3"],
    ["PTPN12", "ZBTB25"],
    ["SNAI1", "DLX2"],
    ["SNAI1", "UBASH3B"],
]
synergy = list_to_combo(synergy, all_combos)
############################ REDUNDANT ############################
redundant = [
    ["CDKN1C", "CDKN1A"],
    ["MAP2K3", "MAP2K6"],
    ["CEBPB", "CEBPA"],
    ["CEBPE", "CEBPA"],
    ["CEBPE", "SPI1"],
    ["ETS2", "MAPK1"],
    ["FOSB", "CEBPE"],
    ["FOXA3", "FOXA1"],
]
redundant = list_to_combo(redundant, all_combos)
############################ NEOMORPHIC ############################
neomorphic = [
    ["CBL", "TGFBR2"],
    ["KLF1", "TGFBR2"],
    ["MAP2K6", "SPI1"],
    ["SAMD1", "TGFBR2"],
    ["TGFBR2", "C19orf26"],
    ["TGFBR2", "ETS2"],
    ["CBL", "UBASH3A"],
    ["CEBPE", "KLF1"],
    ["DUSP9", "MAPK1"],
    ["FOSB", "PTPN12"],
    ["PLK4", "STIL"],
    ["PTPN12", "OSR2"],
    ["ZC3HAV1", "CEBPE"],
]
neomorphic = list_to_combo(neomorphic, all_combos)
############################ EPISTASIS ############################
epistasis = [
    ["AHR", "KLF1"],
    ["MAPK1", "TGFBR2"],
    ["TGFBR2", "IGDCC3"],
    ["TGFBR2", "PRTG"],
    ["UBASH3B", "OSR2"],
    ["DUSP9", "ETS2"],
    ["KLF1", "CEBPA"],
    ["MAP2K6", "IKZF3"],
    ["ZC3HAV1", "CEBPA"],
]
epistasis = list_to_combo(epistasis, all_combos)
############################ suppresor ############################
suppressor = [
    ["CEBPB", "PTPN12"],
    ["CEBPE", "CNN1"],
    ["CEBPE", "PTPN12"],
    ["CNN1", "MAPK1"],
    ["ETS2", "CNN1"],
    ["ETS2", "IGDCC3"],
    ["ETS2", "PRTG"],
    ["FOSB", "UBASH3B"],
    ["IGDCC3", "MAPK1"],
    ["LYL1", "CEBPB"],
    ["MAPK1", "PRTG"],
    ["PTPN12", "SNAI1"],
]
suppressor = list_to_combo(suppressor, all_combos)
############################ ADDITIVE ############################
additive = [
    ["BPGM", "SAMD1"],
    ["CEBPB", "MAPK1"],
    ["CEBPB", "OSR2"],
    ["DUSP9", "PRTG"],
    ["FOSB", "OSR2"],
    ["IRF1", "SET"],
    ["MAP2K3", "ELMSAN1"],
    ["MAP2K6", "ELMSAN1"],
    ["POU3F2", "FOXL2"],
    ["RHOXF2BB", "SET"],
    ["SAMD1", "PTPN12"],
    ["SAMD1", "UBASH3B"],
    ["SAMD1", "ZBTB1"],
    ["SGK1", "TBX2"],
    ["TBX3", "TBX2"],
    ["ZBTB10", "SNAI1"],
]
additive = list_to_combo(additive, all_combos)

GI_truth = {
    "synergy": synergy,
    "redundant": redundant,
    "neomorphic": neomorphic,
    "epistasis": epistasis,
    "suppressor": suppressor,
    "additive": additive,
}
with open("GI_truth.pkl", "wb") as f:
    pickle.dump(GI_truth, f)
