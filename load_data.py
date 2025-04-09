from datasets import load_dataset
import yt_dlp
import os

# Load the dataset
dataset = load_dataset("lmms-lab/AISG_Challenge")
split = "test"
column_url = "youtube_url"
column_id = "video_id"

# Output directory
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# Configure yt-dlp options
ydl_opts = {
    "format": "mp4",
    "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
    "quiet": False,
    "noplaylist": True,
}

# Deduplicate and zip video IDs with URLs
video_ids = dataset[split][column_id]
video_urls = dataset[split][column_url]

unique_videos = list(set(zip(video_ids, video_urls)))

# Existing downloaded video IDs
existing_files = {os.path.splitext(f)[0] for f in os.listdir(output_dir)}

# Download videos
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
