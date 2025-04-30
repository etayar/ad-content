import cv2
import os

def inspect_video(video_path):
    print(f"\nInspecting: {video_path}")

    if not os.path.exists(video_path):
        print("File not found.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video.")
        return

    # Basic metadata
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"Resolution    : {width} x {height}")
    print(f"Frame count   : {total_frames}")
    print(f"FPS           : {fps:.2f}")
    print(f"Duration      : {duration:.2f} seconds")

    # Grab and show one sample frame
    success, frame = cap.read()
    if success:
        cv2.imshow("Sample Frame", frame)
        print("Press any key to close frame preview...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Couldn't read a frame.")

    cap.release()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    args = parser.parse_args()

    inspect_video(args.video)
