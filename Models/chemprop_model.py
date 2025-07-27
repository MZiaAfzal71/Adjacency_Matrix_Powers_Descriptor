from sklearn.metrics import mean_squared_error
import sklearn.metrics
sklearn.metrics.root_mean_squared_error = lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# from pathlib import Path

import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# from lightning.pytorch.callbacks import ModelCheckpoint

# from chemprop import data, models, nn #, uncertainty
from chemprop.models import save_model, load_model
# from chemprop.cli.predict import find_models




from chemprop import data, featurizers, models, nn


input_path =  "Excel Files/boiling_point_data.csv" # path to your data .csv file
# split_path = '/kaggle/input/weighted-desc/random_split_42.npz'
split_path = 'Split Indices/scaffold_split_Murcko.npz'
# num_workers = 1
smiles_column = 'smiles' # name of the column containing SMILES strings
target_columns = ['boiling_point'] # list of names of the columns containing targets

df_input = pd.read_csv(input_path)
smis = df_input.loc[:, smiles_column].values
ys = df_input.loc[:, target_columns].values

all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]

# Load split indices
split_data = np.load(split_path)
train_ind, val_ind, test_ind = [split_data['train_idx']], [split_data['val_idx']], [split_data['test_idx']]


train_data, val_data, test_data = data.split_data_by_indices(
    all_data, train_ind, val_ind, test_ind
)

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

train_dset = data.MoleculeDataset(train_data[0], featurizer)
scaler = train_dset.normalize_targets()

val_dset = data.MoleculeDataset(val_data[0], featurizer)
val_dset.normalize_targets(scaler)

test_dset = data.MoleculeDataset(test_data[0], featurizer)

train_loader = data.build_dataloader(train_dset)
val_loader = data.build_dataloader(val_dset, shuffle=False)
test_loader = data.build_dataloader(test_dset, shuffle=False)

mp = nn.BondMessagePassing()
agg = nn.MeanAggregation()
output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
ffn = nn.RegressionFFN(output_transform=output_transform)
batch_norm = True

metric_list = [nn.metrics.RMSE(), nn.metrics.MAE(), nn.metrics.R2Score()] # Only the first metric is used for training and early stopping
mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list)

# Configure model checkpointing
checkpointing = ModelCheckpoint(
    "chemprop_model/ch_points",  # Directory where model checkpoints will be saved
    "best-{epoch}-{val_loss:.2f}",  # Filename format for checkpoints, including epoch and validation loss
    "val_loss",  # Metric used to select the best checkpoint (based on validation loss)
    mode="min",  # Save the checkpoint with the lowest validation loss (minimization objective)
    save_last=True,  # Always save the most recent checkpoint, even if it's not the best
)


trainer = pl.Trainer(
    logger=False,
    # enable_checkpointing=True, # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
    enable_progress_bar=True,
    accelerator="auto",
    devices=1,
    # max_epochs=30, # number of epochs to train for
    # callbacks=[checkpointing], # Use the configured checkpoint callback
)
# trainer.fit(mpnn, train_loader, val_loader)
best_model_path = 'chemprop_model/ch_points/best-epoch=8-val_loss=0.21.ckpt' # checkpointing.best_model_path
trained_model = mpnn.__class__.load_from_checkpoint(best_model_path)
# save_model("chemprop_model/ch_points" / "best.pt", trained_model)

# model_paths = find_models(best_model_path)
# models = [load_model(path, multicomponent=False) for path in model_paths]

# model = models[0]


# results = trainer.test(dataloaders=test_loader)

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


data_file = pd.read_excel('Excel Files/OurDescriptorWeighted.xlsx')
# Construct result dataframe
results = pd.DataFrame({
    'Name': data_file['Name'],
    'SMILES': data_file['SMILES'],
    'Observed': data_file['Boiling Point'],
    'Predicted': alldata_preds.ravel(),
    'Category': [''] * len(data_file)
})
combined_test_ind = np.concatenate([val_ind[0], test_ind[0]])
results.loc[train_ind[0], 'Category'] = 'Train'
results.loc[combined_test_ind, 'Category'] = 'Test'

# Report performance
print(f'🧬 Processing descriptor: D-MPNN (chemprop)')
print(f'📂 Using split: scaffold')
print(f"🔍 MAE:  {mean_absolute_error(results['Observed'], results['Predicted']):.2f}")
print(f"🔍 RMSE: {np.sqrt(mean_squared_error(results['Observed'], results['Predicted'])):.2f}")
print(f"🔍 R²:   {r2_score(results['Observed'], results['Predicted']):.3f}")

results.to_excel('Output/DMPNN_chemprop_scaf.xlsx', index=False)

print(f'The file is saved to the output directory!')

