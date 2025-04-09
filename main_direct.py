# Author: Tony
# This file uses Qwen2.5-VL-32B-Instruct to process the AISG Challenge dataset.

import os
import tqdm
import json
import pandas as pd
from collections import defaultdict
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Set video reader
os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchvision"


def load_model_and_processor():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-32B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct")
    return model, processor


def build_prompt(video_local_path, question_pairs):
    """
    question_pairs: List of (question_prompt, question) tuples
    """
    combined_text = "**Task**\nAnalyze the video step by step, and answer the following questions clearly.\n\n"
    for idx, (q_prompt, q_text) in enumerate(question_pairs, 1):
        combined_text += f"**Prompt {idx}**\n{q_prompt.strip()}\n\n"
        combined_text += f"**Question {idx}**\n{q_text.strip()}\n\n"
        combined_text += f"**Answer {idx}**\n\n"

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": """
**System**
You are a helpful and knowledgeable assistant.

You will be shown a video and asked multiple questions about it. Your task is to analyze the video carefully and provide accurate answers based on both visual cues and real-world scientific reasoning.

Please note:
- Videos may be edited, stylized, or contain visual illusions.
- What you see might not reflect physical reality — use scientific principles and common sense to ground your answers.
- All phenomena can be explained by natural laws or video editing; avoid assuming supernatural or impossible events.
- Physically impossible scenarios (e.g., reverse gravity, teleportation, infinite motion, etc.) should be treated as visual effects, camera tricks, or post-processing.

Always provide a concise explanation for each answer, rooted in logical and scientific interpretation.

Return your answers as a **JSON array** in the same order as the questions.
- Example: ["Answer to Q1", "Answer to Q2", ...]
- **Only** output the JSON array. Do not include any extra formatting such as ```json or commentary.

You must respond in the following format:

[
  "Answer to Question 1",
  "Answer to Question 2",
  ...
]
Respond clearly and factually.
                    """,
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{video_local_path}",
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                },
                {"type": "text", "text": combined_text},
            ],
        },
    ]


def clean_json_fenced_output(output: str) -> str:
    output = output.strip()
    if output.startswith("```"):
        output = output.strip("`").strip()
        if output.startswith("json"):
            output = output[len("json") :].strip()
    return output


def group_questions_by_video(test_set):
    video_to_questions = defaultdict(list)
    for row in test_set:
        video_to_questions[row["video_id"]].append(
            {
                "qid": row["qid"],
                "question_prompt": row["question_prompt"],
                "question": row["question"],
            }
        )
    return video_to_questions


def process_dataset(
    model,
    processor,
    dataset_name="lmms-lab/AISG_Challenge",
    split="test",
    data_dir="/workspace/data/",
):
    dataset = load_dataset(dataset_name, split=split)
    video_to_questions = group_questions_by_video(dataset)
    results = []

    with tqdm.tqdm(total=len(video_to_questions), desc="Processing videos") as pbar:
        for video_id, qlist in video_to_questions.items():
            video_local_path = os.path.join(data_dir, f"{video_id}.mp4")
            if not os.path.exists(video_local_path):
                print(f"❌ Missing video: {video_local_path}")
                pbar.update(1)
                continue

            try:
                question_pairs = [(q["question_prompt"], q["question"]) for q in qlist]
                message = build_prompt(video_local_path, question_pairs)

                text = processor.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                )

                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    message, return_video_kwargs=True
                )

                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    **video_kwargs,
                ).to("cuda")

                generated_ids = model.generate(**inputs, max_new_tokens=1024)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]

                output_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                try:
                    output_text = clean_json_fenced_output(output_text)
                    answers = json.loads(output_text)
                    if not (isinstance(answers, list)):
                        raise ValueError("Output is not a list")

                except Exception as e:
                    print(
                        f"❌ Failed to parse JSON from model output for {video_id}: {e}"
                    )
                    answers = []

                # Fallback for mismatch
                if len(answers) != len(qlist):
                    print(
                        f"⚠️ Mismatch: Expected {len(qlist)} answers, got {len(answers)} for {video_id}"
                    )
                    answers += [""] * (len(qlist) - len(answers))

                for q, ans_list in zip(qlist, answers):
                    results.append(
                        {
                            "qid": q["qid"],
                            "video_id": video_id,
                            "question": q["question"],
                            "question_prompt": q["question_prompt"],
                            "pred": ans_list,  # Save the list directly
                        }
                    )

                    print(f"📄 QID: {q['qid']}")
                    print(f"❓ Question: {q['question']}")
                    for idx, ans in enumerate(ans_list, 1):
                        print(f"🔹 Answer {idx}: {ans}")
                    print("=" * 50)

            except Exception as e:
                print(f"❌ Error processing video {video_id}: {e}")

            pbar.update(1)

    return results


def save_results(results, output_path="aisg_predictions.jsonl"):
    with open(output_path, "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    print(f"✅ Saved results to {output_path}")


if __name__ == "__main__":
    model, processor = load_model_and_processor()
    results = process_dataset(model, processor)
    save_results(results)
