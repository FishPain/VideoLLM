from datasets import load_dataset
import pandas as pd

# Load dataset
dataset = load_dataset("lmms-lab/AISG_Challenge", split="test")
output_file = "submission.csv"
input_file = "aisg_predictions.jsonl"

# Load predictions
df = pd.read_json(input_file, lines=True)

# Check required columns
assert "qid" in df.columns and "pred" in df.columns, f"❌ '{input_file}' must contain 'qid' and 'pred' columns"

# Keep only necessary columns
df = df[["qid", "pred"]]

# Validation checks
assert df["qid"].isna().sum() == 0, "❌ Null values found in 'qid'"
assert df["pred"].isna().sum() == 0, "❌ Null values found in 'pred'"
assert (df["pred"] == "").sum() == 0, "❌ Some predictions are empty strings ('')"
assert df["qid"].duplicated().sum() == 0, "❌ Duplicated 'qid' entries found"
assert len(df) == 1500, f"❌ Expected 1500 entries, found {len(df)}"

# Save to CSV
df.to_csv(output_file, index=False)
print(f"✅ Submission saved to '{output_file}' with 1500 rows and no null/duplicate entries.")
