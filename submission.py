from datasets import load_dataset
import pandas as pd

# Load dataset
dataset = load_dataset("lmms-lab/AISG_Challenge", split="test")
output_file = "submission.csv"
input_file = "submission.jsonl"

# Load predictions
df = pd.read_json(input_file, lines=True)

# Check required columns
assert (
    "qid" in df.columns and "pred" in df.columns
), f"❌ '{input_file}' must contain 'qid' and 'pred' columns"

# Ensure correct data types
df["qid"] = df["qid"].astype(str)
df["pred"] = df["pred"].astype(str)

# Fill in missing predictions
missing = []
for row in dataset:
    if str(row["qid"]) not in df["qid"].values:
        missing.append(
            {
                "qid": str(row["qid"]),
                "pred": "Video Unavailable",
            }
        )

# Combine original + missing
mdf = pd.DataFrame(missing)
df = pd.concat([df, mdf], ignore_index=True)

# Drop duplicates, sort, and reset index
df = df.drop_duplicates(subset=["qid"], keep="last")
df = df.sort_values(by=["qid"])
df = df.reset_index(drop=True)

# Final check
assert len(df) == len(
    dataset
), f"❌ Submission has {len(df)} entries, but dataset has {len(dataset)} examples"

# Save output
df.to_csv(output_file, index=False)
print(f"✅ Submission saved to {output_file}")
print(f"🔎 Missing predictions filled: {len(missing)}")
