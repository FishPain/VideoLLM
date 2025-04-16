# Author: Tony
# This file uses Qwen2.5-VL-32B-Instruct to process the AISG Challenge dataset.

import os
import tqdm
import torch
from datasets import load_dataset
from qwen_vl_utils import process_vision_info

from utils import (
    load_model_and_processor,
    save_results,
    build_prompt
)

from prompts import INDIVIDUAL_SYSTEM_PROMPT

# Set video reader
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def process_dataset(
    model,
    processor,
    dataset_name="lmms-lab/AISG_Challenge",
    split="test",
    data_dir="/workspace/data/",
):
    dataset = load_dataset(dataset_name, split=split)
    results = []

    with tqdm.tqdm(total=len(dataset), desc="Processing videos") as pbar:
        dataset = dataset.sort("video_id")
        prev_video_id = None
        image_inputs, video_inputs, video_kwargs = None, None, None
        for row in dataset:
            video_id = row["video_id"]
            video_local_path = os.path.join(data_dir, f"{video_id}.mp4")
            if not os.path.exists(video_local_path):
                print(f"❌ Missing video: {video_local_path}")
                video_local_path = None

            try:
                question_pairs = [(row["question_prompt"], row["question"])]
                message = build_prompt(video_local_path, question_pairs, INDIVIDUAL_SYSTEM_PROMPT)

                text = processor.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                )
                if video_id != prev_video_id:
                    image_inputs, video_inputs, video_kwargs = process_vision_info(
                        message, return_video_kwargs=True
                    )
                    prev_video_id = video_id

                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    **video_kwargs,
                ).to("cuda")
                
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=512)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :]
                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]

                    output_text = processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0]

                results.append(
                    {
                        "qid": row["qid"],
                        "video_id": video_id,
                        "question": row["question"],
                        "question_prompt": row["question_prompt"],
                        "pred": output_text,  # Save the list directly
                    }
                )

                print(f"📄 QID: {row['qid']}")
                print(f"Question: {row['question']}")
                print(f"Answer: {output_text}")
                print("=" * 50)

            except Exception as e:
                print(f"❌ Error processing video {video_id}: {e}")
            
            torch.cuda.empty_cache()
            pbar.update(1)

    return results


if __name__ == "__main__":
    model, processor = load_model_and_processor("Qwen/Qwen2.5-VL-32B-Instruct")
    results = process_dataset(model, processor)
    save_results(results)
