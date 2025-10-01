from gears import PertData, GEARS
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("seed", type=int, default=24)
args = parser.parse_args()
split = args.seed

pert_data = PertData('DataGEARS')
pert_data.load(data_name = 'norman')
split_path = f"splits_gears/split_dict_{split}.pkl"
pert_data.prepare_split(split="custom", split_dict_path=split_path)
pert_data.get_dataloader(batch_size = 32, test_batch_size = 32)

gears_model = GEARS(
    pert_data,
    device="cuda" if torch.cuda.is_available() else "cpu",
    weight_bias_track=False,
    proj_name="norman",
    exp_name="GI_GEARS",
)
gears_model.model_initialize(hidden_size = 64)
gears_model.train(epochs = 15)
gears_model.save_model(f"GEARS_model/split_{split}")