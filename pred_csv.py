import pandas as pd
import os

# -------------------------- Configuration --------------------------
# Paths to the two prediction CSV files
S4_CSV_PATH = "./results/val_results/val_results_S4/val_predictions.csv"
S1_CSV_PATH = "./results/val_results/val_results_S1/val_predictions.csv"

# Output path for merged CSV
OUTPUT_PATH = "./results/LiFS_pred.csv"


# -------------------------- Load & Process Data --------------------------
# Load S4 predictions: keep Case, Setting, and Subtask1_prob_S4 columns
print(f"Loading S4 predictions from: {S4_CSV_PATH}")
df_s4 = pd.read_csv(S4_CSV_PATH)
# Keep only required columns (Case, Setting, Subtask1_prob_S4)
df_s4 = df_s4[["Case", "Setting", "Subtask1_prob_S4"]]

# Load S1 predictions: keep Case and Subtask1_prob_S1 (rename to Subtask2_prob_S1)
print(f"Loading S1 predictions from: {S1_CSV_PATH}")
df_s1 = pd.read_csv(S1_CSV_PATH)
# Keep & rename target column (match your required output name)
df_s1 = df_s1[["Case", "Subtask1_prob_S1"]].rename(
    columns={"Subtask1_prob_S1": "Subtask2_prob_S1"}
)


# -------------------------- Merge by Case (Handle Different Orders) --------------------------
# Merge DataFrames on 'Case' (inner join: only keep cases present in both files)
print("Merging files by 'Case' column...")
df_merged = pd.merge(
    df_s4,          # Base DataFrame (provides Case/Setting/Subtask1_prob_S4)
    df_s1,          # Merge with S1 DataFrame
    on="Case",      # Match rows by 'Case' value
    how="inner"     # Keep only cases existing in both files
)


# -------------------------- Reorder Columns (Match Required Output) --------------------------
df_merged = df_merged[["Case", "Setting", "Subtask1_prob_S4", "Subtask2_prob_S1"]]


# -------------------------- Save Merged CSV --------------------------
# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Save merged DataFrame
df_merged.to_csv(OUTPUT_PATH, index=False)
print(f"Merged CSV saved to: {OUTPUT_PATH}")


# -------------------------- Verify Result (Optional) --------------------------
print("\nFirst 5 rows of merged CSV:")
print(df_merged.head())