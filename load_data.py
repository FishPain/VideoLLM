import os
import yt_dlp
from datasets import load_dataset
from utils import extract_audio

def download_videos(
    dataset_name="lmms-lab/AISG_Challenge",
    split="test",
    column_url="youtube_url",
    column_id="video_id",
    output_dir="data"
):
    os.makedirs(output_dir, exist_ok=True)
    dataset = load_dataset(dataset_name)[split]

    video_ids = dataset[column_id]
    video_urls = dataset[column_url]
    unique_videos = list(set(zip(video_ids, video_urls)))

    existing_files = {os.path.splitext(f)[0] for f in os.listdir(output_dir)}

    ydl_opts = {
        "format": "mp4",
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "quiet": False,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for vid, url in unique_videos:
            if not url or not isinstance(url, str):
                continue
            if vid in existing_files:
                print(f"✔️ Skipping {vid} (already downloaded)")
                continue
            try:
                print(f"⬇️ Downloading {vid}: {url}")
                ydl.download([url])
            except Exception as e:
                print(f"❌ Failed to download {url}: {e}")


def extract_audio_batch(video_dir):
    for file in os.listdir(video_dir):
        if file.endswith(".mp4"):
            video_path = os.path.join(video_dir, file)
            output_audio_path = extract_audio(video_path)
            print(f"🎵 Audio extracted to: {output_audio_path}")


if __name__ == "__main__":
    data_dir = "data"
    download_videos(output_dir=data_dir)
    extract_audio_batch(data_dir)
