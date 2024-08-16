from gears import PertData, GEARS
import os
import glob

# To generate the dixit dataset
pert_data = PertData('../Data_GEARS')
pert_data.load(data_name = 'dixit')
for split_seed in range(1,11):
    pert_data.prepare_split(split = 'simulation', seed = split_seed, train_gene_set_size=0.9)
pert_data.get_dataloader(batch_size = 32, test_batch_size = 32)
gears_model = GEARS(pert_data, 
                    device = "cpu",
                    weight_bias_track = False, 
                    proj_name = 'dixit')
gears_model.model_initialize(hidden_size = 64)
    
# To generate the adamson dataset
pert_data = PertData('../Data_GEARS')
pert_data.load(data_name = 'adamson')
for split_seed in range(1,6):
    pert_data.prepare_split(split = 'simulation', seed = split_seed)
    
# To generate the norman dataset
pert_data = PertData('../Data_GEARS')
pert_data.load(data_name = 'norman')
for split_seed in range(1,6):
    pert_data.prepare_split(split = 'simulation', seed = split_seed)
    
# To generate the k562 dataset
pert_data = PertData('../Data_GEARS')
pert_data.load(data_name = 'replogle_k562_essential')
for split_seed in range(1,6):
    pert_data.prepare_split(split = 'simulation', seed = split_seed)
    
# To generate the rpe1 dataset
pert_data = PertData('../Data_GEARS')
pert_data.load(data_name = 'replogle_rpe1_essential')
for split_seed in range(1,6):
    pert_data.prepare_split(split = 'simulation', seed = split_seed)

# Delete .zip files, they are unnecessary once unzipped
# Get the list of all .zip files in the current directory
zip_files = glob.glob("*.zip")
# Loop through the list and remove each file
for file in zip_files:
    os.remove(file)
    print(f"Deleted: {file}")
    
gz_files = glob.glob("*.tar.gz")
for file in gz_files:
    os.remove(file)
    print(f"Deleted: {file}")