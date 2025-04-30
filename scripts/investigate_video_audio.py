from moviepy.video.io.VideoFileClip import VideoFileClip
import subprocess
import os
import platform
import cv2
import numpy as np
import pandas as pd


def extract_video_and_audio_data(video_path, output_dir="data/extracted", save=False, frame_interval=1):
    clip = VideoFileClip(video_path)
    fps = clip.fps
    duration = clip.duration
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    # Output paths (conditionally used if save=True)
    frames_dir = os.path.join(output_dir, base_name, "frames")
    audio_path = os.path.join(output_dir, base_name, "audio.wav")

    # Prepare collection
    frame_data = []

    # --- Extract frame data ---
    n_frames = int(duration * fps)
    for i in range(n_frames):
        if i % frame_interval != 0:
            continue
        t = i / fps
        frame_entry = {
            "video_name": base_name,
            "frame_index": i,
            "timestamp": round(t, 5),
            "resolution": clip.size,
            "fps": fps,
        }

        if save:
            os.makedirs(frames_dir, exist_ok=True)
            frame_path = os.path.join(frames_dir, f"frame_{i:05}.jpg")
            clip.save_frame(frame_path, t=t)
            frame_entry["frame_path"] = frame_path

        frame_data.append(frame_entry)

    # --- Extract audio if available ---
    if clip.audio is not None:
        audio_info = {
            "video_name": base_name,
            "has_audio": True,
            "audio_fps": clip.audio.fps,
            "audio_duration": clip.audio.duration,
            "audio_channels": clip.audio.nchannels,
        }
        if save:
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)
            clip.audio.write_audiofile(audio_path, logger=None)
            audio_info["audio_path"] = audio_path
    else:
        audio_info = {"video_name": base_name, "has_audio": False}

    # --- Wrap up ---
    frame_df = pd.DataFrame(frame_data)

    if save:
        meta_path = os.path.join(output_dir, base_name, "metadata.csv")
        frame_df.to_csv(meta_path, index=False)
        print(f"📁 Saved frame metadata to: {meta_path}")
        if "audio_path" in audio_info:
            print(f"🎵 Audio saved to: {audio_info['audio_path']}")
        else:
            print("🔇 No audio saved.")

    return frame_df, audio_info


def read_clip(video_path):

    cap = cv2.VideoCapture(video_path)

    success, frame = cap.read()  # Read first frame
    if success:
        print("Frame shape:", frame.shape)  # (height, width, channels)
        print("Top-left 5x5 pixel values:\n", frame[:5, :5])  # Print a small region
    else:
        print("Failed to read frame.")
    cap.release()


def play_video(video_path):
    if platform.system() == "Darwin":  # macOS
        subprocess.run(["open", video_path])
    elif platform.system() == "Windows":
        os.startfile(video_path)
    elif platform.system() == "Linux":
        subprocess.run(["xdg-open", video_path])
    else:
        print("Unsupported OS for autoplay.")


def inspect_video_with_audio(video_path):
    if not os.path.exists(video_path):
        print("File not found:", video_path)
        return

    clip = VideoFileClip(video_path)

    print(f"📄 File        : {video_path}")
    print(f"🎥 Duration    : {clip.duration:.2f} seconds")
    print(f"📏 Resolution  : {clip.w} x {clip.h}")
    print(f"🎞 FPS         : {clip.fps}")
    print(f"🔊 Has Audio   : {'Yes' if clip.audio else 'No'}")
    if clip.audio:
        print(f"🎵 Audio FPS   : {clip.audio.fps}")
        print(f"🎵 Audio Duration: {clip.audio.duration:.2f} seconds")
        print(f"🎵 Audio Channels: {clip.audio.nchannels}")

    # Save a frame preview
    preview_path = "data/frames/sample_preview.jpg"
    os.makedirs(os.path.dirname(preview_path), exist_ok=True)
    clip.save_frame(preview_path, t=0.5)
    print(f"🖼  Frame saved at 0.5s -> {preview_path}")

    clip.close()

if __name__ == "__main__":
    import argparse

    # Hardcoded path for local PyCharm runs
    default_path = "../data/raw/sample_with_audio.mp4"

    # Extract clip data
    clip_df, audio_info = extract_video_and_audio_data(default_path)

    read_clip(default_path)
    play_video(default_path)

    # Use manual path if no command-line args are passed
    if len(os.sys.argv) == 1:
        print("No arguments provided. Using default path:")
        print(f"--> {default_path}")
        inspect_video_with_audio(default_path)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--video", type=str, required=True, help="Path to video file")
        args = parser.parse_args()
        inspect_video_with_audio(args.video)

    print()