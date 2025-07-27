import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from grakel import Graph, GraphKernel

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# -------------------------
# Settings
# -------------------------
input_csv = "Excel Files/boiling_point_data.csv"
output_dir = "Output"
split_files = {
    "random": "Split Indices/random_split_42.npz",
    "scaffold": "Split Indices/scaffold_split_Murcko.npz"
}
os.makedirs(output_dir, exist_ok=True)

# -------------------------
# Load and Featurize Data
# -------------------------
print("📥 Loading and converting SMILES...")
df = pd.read_csv(input_csv)

def smiles_to_grakel(smiles):
    mol = Chem.MolFromSmiles(smiles)
    nodes = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    return Graph(edges, node_labels={i: l for i, l in enumerate(nodes)})

graphs = [smiles_to_grakel(smi) for smi in tqdm(df["smiles"])]

print("🧠 Computing Weisfeiler-Lehman kernel...")
gk = GraphKernel(kernel=[{"name": "weisfeiler_lehman", "n_iter": 3},
                         {"name": "subtree_wl"}], normalize=True)
X = gk.fit_transform(graphs)
y_raw = df["boiling_point"].values.reshape(-1, 1)

# Scale target
sc_y = StandardScaler()
y = sc_y.fit_transform(y_raw).ravel()
std_y, mean_y = sc_y.scale_[0], sc_y.mean_[0]

# -------------------------
# Define Models
# -------------------------
models = {
    "SVM": SVR(kernel="rbf"),
    "RF": RandomForestRegressor(random_state=42),
    "XGB": XGBRegressor(random_state=42)
}

# -------------------------
# Loop Over Splits and Models
# -------------------------
for split_name, split_path in split_files.items():
    print(f"\n📂 Using split: {split_name}")
    split_data = np.load(split_path)
    train_idx, val_idx, test_idx = split_data["train_idx"], split_data["val_idx"], split_data["test_idx"]
    combined_test_idx = np.concatenate([val_idx, test_idx])

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[combined_test_idx]
    y_test = y[combined_test_idx]

    for model_name, model in models.items():
        print(f"\n🔧 Training model: {model_name}")
        model.fit(X_train, y_train)
        y_pred_scaled = model.predict(X)

        # Rescale predictions
        y_true = y * std_y + mean_y
        y_pred = y_pred_scaled * std_y + mean_y

        # Evaluation
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        print(f"✅ {model_name} ({split_name}): MAE = {mae:.2f}, RMSE = {rmse:.2f}, R² = {r2:.3f}")

        # Save predictions
        results = pd.DataFrame({
            "SMILES": df["smiles"],
            "Observed": y_true,
            "Predicted": y_pred,
            "Category": [''] * len(df)
        })
        results.loc[train_idx, "Category"] = "Train"
        results.loc[combined_test_idx, "Category"] = "Test"

        output_file = os.path.join(output_dir, f"{model_name}_WL_{split_name}.xlsx")
        results.to_excel(output_file, index=False)
        print(f"📁 Results saved to: {output_file}")
