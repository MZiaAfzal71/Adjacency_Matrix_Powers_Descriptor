import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from chemprop import data, featurizers, models, nn #, uncertainty

from chemprop.models import save_model, load_model
import sys
import os
# from chemprop.cli.predict import find_models

out_dir = Path("chemprop_model/ch_pt_pows_scaffold_chemeleon") # directory for storing the best model after training
os.makedirs(out_dir, exist_ok=True)

input_path =  "Excel Files/boiling_point_data.csv" # path to your data .csv file

# split_path = 'Split Indices/random_split_42.npz' # path containing random split indices
split_path = 'Split Indices/scaffold_split_Murcko.npz' # path containing scaffold split indices

desc_path = 'Excel Files/apow_descriptors.npz'  # path of extra molecule descriptors
# desc_path = 'Excel Files/atomic_desc.npz'  # path of extra molecule descriptors
# desc_path = 'Excel Files/OurDescriptorWeighted.xlsx'  # path of extra molecule descriptors
# desc_path = 'Excel Files/MordredDescriptor.xlsx'  # path of extra molecule descriptors
# desc_path = 'Excel Files/MorganDescriptor.xlsx'  # path of extra molecule descriptors

data_file = pd.read_excel('Excel Files/OurDescriptorWeighted.xlsx') # Just to extract chemical names at the end

# num_workers = 1
smiles_column = 'smiles' # name of the column containing SMILES strings
target_columns = ['boiling_point'] # list of names of the columns containing targets

df_input = pd.read_csv(input_path)
smis = df_input.loc[:, smiles_column].values
ys = df_input.loc[:, target_columns].values

if desc_path.endswith('.npz'):
    ext_desc = np.load(desc_path)['X_d']
elif desc_path.endswith('.xlsx'):
    desc_file = pd.read_excel(desc_path)
    desc_file.fillna(0, inplace=True)
    ext_desc = np.array(desc_file.iloc[:, 3:].to_numpy(), dtype=np.float32) # In my file descrptor start from 3 column onward
elif desc_path.endswith('.csv'):
    desc_file = pd.read_csv(desc_path)
    desc_file.fillna(0, inplace=True)
    ext_desc = np.array(desc_file.iloc[:, 3:].to_numpy(), dtype=np.float32)  # In my file descrptor start from 3 column onward
else:
    print(f'The file {desc_path} must be an excel (.xlsx/.csv) file/ .npz saved with numpy!')
    sys.exit()

all_data = [data.MoleculeDatapoint.from_smi(smi, y, x_d=X_d)
            for smi, y, X_d in zip(smis, ys, ext_desc)]

# Load split indices
split_data = np.load(split_path)
train_ind, val_ind, test_ind = [split_data['train_idx']], [split_data['val_idx']], [split_data['test_idx']]


train_data, val_data, test_data = data.split_data_by_indices(
    all_data, train_ind, val_ind, test_ind
)

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

train_dset = data.MoleculeDataset(train_data[0], featurizer)
scaler = train_dset.normalize_targets()
extra_desc_scaler = train_dset.normalize_inputs("X_d")

val_dset = data.MoleculeDataset(val_data[0], featurizer)
val_dset.normalize_targets(scaler)
val_dset.normalize_inputs("X_d", extra_desc_scaler)

test_dset = data.MoleculeDataset(test_data[0], featurizer)

# Featurize the train and val datasets to save computation time.
train_dset.cache = True
val_dset.cache = True

train_loader = data.build_dataloader(train_dset)
val_loader = data.build_dataloader(val_dset, shuffle=False)
test_loader = data.build_dataloader(test_dset, shuffle=False)

agg = nn.NormAggregation()
chemeleon_mp = torch.load('chemprop_model/chemeleon/chemeleon_mp.pt', weights_only=True)
mp = nn.BondMessagePassing(**chemeleon_mp['hyper_parameters'])
mp.load_state_dict(chemeleon_mp['state_dict'])


ffn_input_dim = mp.output_dim + ext_desc.shape[1]
output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
ffn = nn.RegressionFFN(input_dim=ffn_input_dim, output_transform=output_transform)
mp.eval()
mp.apply(lambda module: module.requires_grad_(False))
# batch_norm = True
#
metric_list = [nn.metrics.RMSE(), nn.metrics.MAE(), nn.metrics.R2Score()] # Only the first metric is used for training and early stopping

X_d_transform = nn.ScaleTransform.from_standard_scaler(extra_desc_scaler)
mpnn = models.MPNN(mp, agg, ffn, batch_norm=False, X_d_transform=X_d_transform, metrics=metric_list)

# Configure model checkpointing
check_pointing = ModelCheckpoint(
    out_dir,  # Directory where model checkpoints will be saved
    "best-{epoch}-{val_loss:.3f}",  # Filename format for checkpoints, including epoch and validation loss
    "val_loss",  # Metric used to select the best checkpoint (based on validation loss)
    mode="min",  # Save the checkpoint with the lowest validation loss (minimization objective)
    save_last=True,  # Always save the most recent checkpoint, even if it's not the best
)


trainer = pl.Trainer(
    logger=False,
    enable_checkpointing=True, # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
    enable_progress_bar=True,
    accelerator="auto",
    devices=1,
    max_epochs=30, # number of epochs to train for
    callbacks=[check_pointing], # Use the configured checkpoint callback
)
trainer.fit(mpnn, train_loader, val_loader)
best_model_path = check_pointing.best_model_path
trained_model = mpnn.__class__.load_from_checkpoint(best_model_path)
# trained_model = mpnn.__class__.load_from_checkpoint('chemprop_model/ch_pt_morgan_scaffold/best-epoch=9-val_loss=0.262.ckpt')
# save_model(out_dir / "best.pt", trained_model)
#
# model_paths = find_models([out_dir])
# my_models = [load_model(path, multicomponent=False) for path in model_paths]
#
# best_model = my_models[0]

#
results = trainer.test(dataloaders=test_loader)
#
alldata_dset = data.MoleculeDataset(all_data, featurizer)
alldata_loader = data.build_dataloader(alldata_dset, shuffle=False)

with torch.inference_mode():
    trainer = pl.Trainer(
        logger=None,
        enable_progress_bar=True,
        accelerator="cpu",
        devices=1
    )
    alldata_preds = trainer.predict(trained_model, alldata_loader)

alldata_preds = np.concatenate(alldata_preds, axis=0)


# Construct result dataframe
new_results = pd.DataFrame({
    'Name': data_file['Name'],
    'SMILES': data_file['SMILES'],
    'Observed': data_file['Boiling Point'],
    'Predicted': alldata_preds.ravel(),
    'Category': [''] * len(data_file)})
combined_test_ind = np.concatenate([val_ind[0], test_ind[0]])
new_results.loc[train_ind[0], 'Category'] = 'Train'
new_results.loc[combined_test_ind, 'Category'] = 'Test'

# Report performance
print(f'🧬 Processing descriptor: D-MPNN (chemprop + pows + chemeleon)')
print(f'📂 Using split: scaffold')
print(f"🔍 MAE:  {mean_absolute_error(new_results['Observed'], new_results['Predicted']):.2f}")
print(f"🔍 RMSE: {np.sqrt(mean_squared_error(new_results['Observed'], new_results['Predicted'])):.2f}")
print(f"🔍 R²:   {r2_score(new_results['Observed'], new_results['Predicted']):.3f}")

new_results.to_excel('Output/DMPNN_chemprop_pows_scaf_chemeleon.xlsx', index=False)

print(f'The file is saved to the output directory!')

