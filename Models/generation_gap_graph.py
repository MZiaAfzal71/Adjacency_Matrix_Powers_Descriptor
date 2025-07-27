import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set seaborn style
sns.set(style="whitegrid")

# Load both CSV files
train_df = pd.read_csv("performance_summary_train_new.csv")  # contains: Model, Descriptor, Split, MAE
test_df = pd.read_csv("performance_summary_new.csv")    # contains: Model, Descriptor, Split, MAE

# Rename MAE columns for clarity
train_df = train_df.rename(columns={"MAE": "Train_MAE"})
test_df = test_df.rename(columns={"MAE": "Test_MAE"})

# Merge on common identifiers
merged_df = pd.merge(train_df, test_df, on=["Model", "Descriptor", "Split"])

# Compute generalization gap
merged_df["GenGap"] = merged_df["Test_MAE"] - merged_df["Train_MAE"]

# Create the plot
g = sns.catplot(
    data=merged_df,
    kind="bar",
    x="Descriptor",
    y="GenGap",
    hue="Model",
    col="Split",         # Separate plots for 'random' and 'scaffold'
    height=5,
    aspect=1.2,
    legend_out=False,
    palette="muted"
)

# Add title
g.fig.suptitle("Generalization Gap (Test MAE − Train MAE) by Descriptor and Model", fontsize=14)
g.fig.subplots_adjust(top=0.85)

# Rotate x-ticks for clarity

g.set_xticklabels(rotation=45)

# Optional: draw horizontal line at 0 for reference
for ax in g.axes.flat:
    ax.axhline(0, color='black', linestyle='--', linewidth=1)

# Save the plot
g.savefig("../plots/generalization_gap_by_split.png", bbox_inches="tight")
plt.close()