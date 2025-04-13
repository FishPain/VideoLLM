import cv2
import subprocess
import os
import json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from collections import defaultdict
from faster_whisper import WhisperModel
from prompts import DEFAULT_SYSTEM_PROMPT

def get_video_info(path):
    """
    Return (width, height, fps) of the video.
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height, fps


def clean_json_fenced_output(output: str) -> str:
    output = output.strip()
    if output.startswith("```"):
        output = output.strip("`").strip()
        if output.startswith("json"):
            output = output[len("json") :].strip()
    return output


def load_model_and_processor(model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2.5-VL-7B-Instruct",
    #     torch_dtype="auto",
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

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


def extract_audio(video_path, output_audio_path=None, sample_rate=16000):
    """
    Extract mono audio from video and save as .wav.

    Args:
        video_path (str): Path to the input video.
        output_audio_path (str): Optional output .wav path.
        sample_rate (int): Sample rate for audio (default 16kHz).
    """
    if output_audio_path is None:
        output_audio_path = video_path.rsplit(".", 1)[0] + ".wav"

    command = [
        "ffmpeg",
        "-i",
        video_path,
        "-ac",
        "1",  # mono channel
        "-ar",
        str(sample_rate),  # audio sample rate
        "-f",
        "wav",  # format
        "-y",  # overwrite
        output_audio_path,
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_audio_path


def change_video_speed(input_path, output_path, speed_factor):
    # Handle audio tempo (ffmpeg supports 0.5–2.0 per filter, so we may need chaining)
    def get_atempo_chain(factor):
        if factor <= 0:
            raise ValueError("Speed factor must be positive")

        chain = []
        while factor < 0.5:
            chain.append("atempo=0.5")
            factor /= 0.5
        while factor > 2.0:
            chain.append("atempo=2.0")
            factor /= 2.0
        chain.append(f"atempo={factor:.5f}")
        return ",".join(chain)

    if speed_factor == 1.0:
        print("Speed factor is 1.0, skipping processing.")
        return

    video_filter = f"setpts={1/speed_factor}*PTS"
    audio_filter = get_atempo_chain(speed_factor)

    ffmpeg_cmd = [
        "ffmpeg",
        "-i", input_path,
        "-filter_complex", f"[0:v]{video_filter}[v];[0:a]{audio_filter}[a]",
        "-map", "[v]",
        "-map", "[a]",
        "-y",  # Overwrite output if exists
        output_path
    ]

    subprocess.run(ffmpeg_cmd, check=True)

class AudioTranscriber:
    def __init__(self):
        # Load the Whisper model
        self.whisper_model = WhisperModel("base.en", compute_type="auto")

    def transcribe_audio(self, video_path):
        audio_path = video_path.replace(".mp4", ".wav") if video_path.endswith(".mp4") else None
        if not audio_path or not os.path.exists(audio_path):
            # Extract audio if not already done
            audio_path = extract_audio(video_path)
        segments, _ = self.whisper_model.transcribe(audio_path)
        transcription = " ".join([seg.text for seg in segments])

        return transcription


def build_prompt(video_local_path, question_pairs, custom_system_message=DEFAULT_SYSTEM_PROMPT, transcription=""):
    """
    question_pairs: List of (question_prompt, question) tuples
    """
    content = []
    
    if video_local_path is not None:
        w, h, fps = get_video_info(video_local_path)
        content.append({
            "type": "video",
            "video": f"file://{video_local_path}",
            "max_pixels": w * h,
            "fps": 1,
        })

        video_instruction = "Analyze the video step by step, and answer the following questions clearly."
    else:
        video_instruction = (
            "The video file is missing. Based on the available information (prompt), "
            "make your best guess to answer each question. ALWAYS GIVE AN ANSWER."
        )

    combined_text = f"**Task**\n{video_instruction}\n\n"
    for idx, (q_prompt, q_text) in enumerate(question_pairs):
        combined_text += f"**Prompt {idx}**\n{q_prompt.strip()}\n\n"
        if transcription:
            combined_text += f"**Video Transcription {idx}**\n{transcription.strip()}\n\n"
        combined_text += f"**Question {idx}**\n{q_text.strip()}\n\n"
        combined_text += f"**Answer {idx}**\n\n"

    content.append({"type": "text", "text": combined_text})
    
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": custom_system_message,
                },
            ],
        },
        {
            "role": "user",
            "content": content,
        },
    ]

if __name__ == "__main__":
    l = os.listdir("/workspace/data")
    for path in l:
        if path.endswith(".mp4"):
            input_path = os.path.join("/workspace/data", path)
            output_path = os.path.join("/workspace/data/slow05", f"{path}")
            change_video_speed(input_path, output_path, 0.5)  # Change speed to 0.5x
            print(f"Processed {input_path} to {output_path}")