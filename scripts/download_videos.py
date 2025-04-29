import os
import pandas as pd
import yt_dlp  # Youtube Downloader library

# Function to download a single video
def download_video(url, output_dir):
    # yt-dlp download options
    ydl_opts = {
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),  # Save video using its title
        'format': 'bestvideo+bestaudio/best',  # Download best video and best audio
        'merge_output_format': 'mp4',  # Save merged file as MP4
        'quiet': False,  # Show download progress
        'noplaylist': True,  # Only download the video, not entire playlists
    }

    # Create a YoutubeDL object with the specified options
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])  # Attempt to download the given URL
        except Exception as e:
            print(f"Failed to download {url}: {e}")  # Print any error

# Main function to download all videos from a CSV file
def main(csv_path, url_column, output_dir="data/raw"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read CSV file
    df = pd.read_csv(csv_path)
    # Extract non-empty URLs into a list
    urls = df[url_column].dropna().tolist()

    # Download each video one by one
    for url in urls:
        print(f"Downloading {url}")
        download_video(url, output_dir)


if __name__ == "__main__":
    import argparse

    # Argument parser to accept command-line inputs
    parser = argparse.ArgumentParser(description="Download videos from a CSV file containing URLs.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV file with video URLs.")
    parser.add_argument("--url_column", type=str, required=True, help="Name of the column with URLs.")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Where to save downloaded videos.")

    args = parser.parse_args()

    # Call main function with parsed arguments
    main(args.csv_path, args.url_column, args.output_dir)
