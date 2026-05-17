import cv2
import os

def video_to_frames(video_path, output_folder, interval=0.5, prefix="frame_0000", start_num=0):
    """
    Convert a video into frames every 'interval' seconds.
    Saved filenames look like: foom_frame6003.jpg, foom_frame6004.jpg, ...
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps * interval)

    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Saving every {interval} seconds (~every {frame_interval} frames)")
    print(f"Starting from: {prefix}{start_num}.jpg")

    frame_count = 0
    current_num = start_num

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = os.path.join(output_folder, f"{prefix}{current_num}.jpg")
            cv2.imwrite(filename, frame)
            current_num += 1

        frame_count += 1

    cap.release()
    print(f"Done! Saved up to {prefix}{current_num - 1}.jpg in '{output_folder}'.")


if __name__ == "__main__":
    # Example usage:
    video_path = "E:\\SKRIPSI\\Video_Original_WarnaCampuran\\WarnaCampuran(Full).mp4"       # ← replace with your video path
    output_folder = "E:\\data\\output"      # ← where frames are saved
    video_to_frames(video_path, output_folder, interval=0.5, start_num=1)
