# Author: Tony
# This file uses Qwen2.5-VL-32B-Instruct to process the AISG Challenge dataset.

import os
import tqdm
import json
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
import torch

from prompts import DEFAULT_SYSTEM_PROMPT
from utils import (
    load_model_and_processor,
    clean_json_fenced_output,
    group_questions_by_video,
    save_results,
    build_prompt,
    AudioTranscriber,
)

# Set video reader
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
                question_pairs = [(q["question_prompt"], q["question"]) for q in qlist]
                transcription = at.transcribe_audio()
                transcription_DEFAULT_SYSTEM_PROMPT = f"""
                    **Video Transcription Context:**\n\n{transcription}\n\n{DEFAULT_SYSTEM_PROMPT}
                """
                message = build_prompt(video_local_path, question_pairs, transcription_DEFAULT_SYSTEM_PROMPT)

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
                    print(f"Question: {q['question']}")
                    print(f"Answer: {ans_list}")
                    print("=" * 50)

            except Exception as e:
                print(f"❌ Error processing video {video_id}: {e}")
            
            torch.cuda.empty_cache()
            pbar.update(1)

    return results


if __name__ == "__main__":
    global at
    at = AudioTranscriber()
    model, processor = load_model_and_processor()
    results = process_dataset(model, processor)
    save_results(results)
