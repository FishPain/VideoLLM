import cv2
import subprocess
import os
import json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from collections import defaultdict
from faster_whisper import WhisperModel


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


class AudioTranscriber:
    def __init__(self):
        # Load the Whisper model
        self.whisper_model = WhisperModel("base.en", compute_type="auto")

    def transcribe_audio(self, video_path):
        audio_path = extract_audio(video_path)
        segments, _ = self.whisper_model.transcribe(audio_path)
        transcription = " ".join([seg.text for seg in segments])

        # Clean up
        if os.path.exists(audio_path):
            os.remove(audio_path)

        return transcription
