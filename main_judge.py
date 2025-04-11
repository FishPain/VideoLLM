# Author: Tony
# This file uses Qwen2.5-VL-32B-Instruct to process the AISG Challenge dataset.
# It generates top-k candidate answers and then uses LLM as a judge to select the best one.

import os
import json
import tqdm
import torch
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from prompts import EVALUATOR_SYSTEM_PROMPT
from utils import (
    build_prompt,
    load_model_and_processor,
    clean_json_fenced_output,
    group_questions_by_video,
    save_results,
)

os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchvision"

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
                video_local_path = None

            try:
                ### Stage 1: Generate Top-K Answers ###
                question_pairs = [(q["question_prompt"], q["question"]) for q in qlist]
                message = build_prompt(video_local_path, question_pairs, EVALUATOR_SYSTEM_PROMPT)

                text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs, video_kwargs = process_vision_info(message, return_video_kwargs=True)

                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    **video_kwargs,
                ).to("cuda")

                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=1024)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
                
                try:
                    output_text = clean_json_fenced_output(output_text)
                    top_k_answers = json.loads(output_text)
                    if not isinstance(top_k_answers, list):
                        raise ValueError("Top-k output is not a list")
                except Exception as e:
                    print(f"❌ Failed to parse top-k JSON for video {video_id}: {e}")
                    top_k_answers = []

                if len(top_k_answers) != len(qlist):
                    print(f"⚠️ Top-k mismatch: {len(top_k_answers)} ≠ {len(qlist)}")
                    top_k_answers += [[""]] * (len(qlist) - len(top_k_answers))

                ### Stage 2: Judge & Select Best ###
                judging_pairs = [
                    (
                        q["question_prompt"],
                        f'{q["question"]}\n\n**Possible Answers**\n{top_k}\n\n'
                    )
                    for q, top_k in zip(qlist, top_k_answers)
                ]
                judge_message = build_prompt(video_local_path, judging_pairs)

                judge_text = processor.apply_chat_template(judge_message, tokenize=False, add_generation_prompt=True)
                judge_inputs = processor(
                    text=[judge_text],
                    images=image_inputs,  # reuse cached vision inputs
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    **video_kwargs,
                ).to("cuda")

                with torch.no_grad():
                    judge_ids = model.generate(**judge_inputs, max_new_tokens=1024)
                    judge_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(judge_inputs.input_ids, judge_ids)
                    ]
                    judge_output = processor.batch_decode(judge_ids_trimmed, skip_special_tokens=True)[0]

                try:
                    judge_output = clean_json_fenced_output(judge_output)
                    final_answers = json.loads(judge_output)
                    if not isinstance(final_answers, list):
                        raise ValueError("Judgment output is not a list")
                except Exception as e:
                    print(f"❌ Failed to parse judged JSON for {video_id}: {e}")
                    final_answers = []

                if len(final_answers) != len(qlist):
                    print(f"⚠️ Final answer mismatch: {len(final_answers)} ≠ {len(qlist)}")
                    final_answers += [""] * (len(qlist) - len(final_answers))

                for q, final in zip(qlist, final_answers):
                    results.append({
                        "qid": q["qid"],
                        "video_id": video_id,
                        "question": q["question"],
                        "question_prompt": q["question_prompt"],
                        "pred": final,
                    })

                    print(f"📄 QID: {q['qid']}")
                    print(f"❓ Question: {q['question']}")
                    print(f"✅ Answer: {final}")
                    print("=" * 50)

            except Exception as e:
                print(f"❌ General error processing video {video_id}: {e}")
            finally:
                torch.cuda.empty_cache()
                pbar.update(1)

    return results


if __name__ == "__main__":
    model, processor = load_model_and_processor()
    results = process_dataset(model, processor)
    save_results(results)