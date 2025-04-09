import cv2
import json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from collections import defaultdict

def get_video_fps(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def clean_json_fenced_output(output: str) -> str:
    output = output.strip()
    if output.startswith("```"):
        output = output.strip("`").strip()
        if output.startswith("json"):
            output = output[len("json") :].strip()
    return output

def load_model_and_processor(model_name="Qwen/Qwen2.5-VL-32B-Instruct"):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor



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


def save_results(results, output_path="aisg_predictions.jsonl"):
    with open(output_path, "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    print(f"✅ Saved results to {output_path}")