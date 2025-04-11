from datasets import load_dataset
import pandas as pd

# Load dataset
dataset = load_dataset("lmms-lab/AISG_Challenge", split="test")
output_file = "submission.csv"
input_file = "aisg_predictions.jsonl"

# Load predictions
df = pd.read_json(input_file, lines=True)

# Check required columns
assert (
    "qid" in df.columns and "pred" in df.columns
), f"❌ '{input_file}' must contain 'qid' and 'pred' columns"

df = df[["qid", "pred"]]

df.to_csv(output_file, index=False)
