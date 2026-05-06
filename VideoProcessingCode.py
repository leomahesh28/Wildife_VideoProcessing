import os
import cv2
import torch
import time
import pandas as pd
from ultralytics import YOLO
from multiprocessing import Pool, set_start_method
from functools import partial

# Function to extract video file paths from folders
def extract_videos_from_folders(parent_folder, video_extensions=None):
    if video_extensions is None:
        video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.flv']

    video_files = []
    for root, dirs, files in os.walk(parent_folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))
    return video_files

# Function to process a single video
def process_video(video_file, output_dirs, animal_classes, device, target_fps, save_fps):
    start_time = time.time()
    print(f"Processing video: {video_file}")

    model = YOLO("/home/maheshsharma/Downloads/Widlife_Yolo11mwithoptuna/runs/runs/detect/train8/weights/best.pt").to(device)
    cap = cv2.VideoCapture(video_file)
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = 0
    wildlife_frame_counts = {animal: 0 for animal in animal_classes[:-1]}
    detected_classes = []
    extracted_frame_details = []

    video_name = os.path.splitext(os.path.basename(video_file))[0]
    output_path = os.path.join(output_dirs["Other"], f"{video_name}_processed.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, save_fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % (original_fps // target_fps) == 0:
            results = model.predict(source=frame, conf=0.40, device=device)

            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy
                    classes = result.boxes.cls

                    for cls in classes:
                        class_name = model.names[int(cls)]
                        if class_name in animal_classes[:-1]:
                            wildlife_frame_counts[class_name] += 1
                            if class_name not in detected_classes:
                                detected_classes.append(class_name)

            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy
                    confs = result.boxes.conf
                    classes = result.boxes.cls

                    for box, conf, cls in zip(boxes, confs, classes):
                        class_name = model.names[int(cls)]
                        if class_name in animal_classes[:-1]:
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{class_name}: {conf:.2f}"
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            extracted_frame_details.append({
                "frame_index": frame_count,
                "detected_classes": detected_classes,
                "wildlife_frame_counts": wildlife_frame_counts.copy()
            })

            out.write(frame)

        frame_count += 1

    cap.release()
    out.release()

    total_frames_processed = frame_count
    detected_class = max(wildlife_frame_counts, key=wildlife_frame_counts.get)
    max_detected_frames = wildlife_frame_counts[detected_class]

    MIN_FRAMES = 5
    MIN_RATIO = 0.1
    detection_ratio = max_detected_frames / (total_frames_processed + 1e-6)

    save_video = (max_detected_frames >= MIN_FRAMES) and (detection_ratio >= MIN_RATIO)

    if save_video:
        final_output_path = os.path.join(output_dirs[detected_class], f"{video_name}_processed.mp4")
        os.rename(output_path, final_output_path)
        output_path = final_output_path
        wildlife_detected = detected_class
    else:
        wildlife_detected = "Other"

    video_processing_time = time.time() - start_time
    return {
        "video_name": os.path.basename(video_file),
        "full_path": video_file,
        "processing_time": video_processing_time,
        "detected_classes": detected_classes,
        "detected_frame_counts": wildlife_frame_counts,
        "output_path": output_path,
        "wildlife_detected": wildlife_detected,
    }

if __name__ == "__main__":
    set_start_method("spawn")
    parent_directory = "/home/maheshsharma/Downloads/Data"
    output_dir_base = "/home/maheshsharma/Downloads/EnvironmentSetuo_Wildlife/Results"
    csv_output_path = os.path.join(output_dir_base, "video_processing_details.csv")

    os.makedirs(output_dir_base, exist_ok=True)

    animal_classes = ['Bear', 'Bobcat', 'Coyote', 'Deer', 'Elk', 'MountainLion', 'Other']
    output_dirs = {animal: os.path.join(output_dir_base, animal) for animal in animal_classes}
    for path in output_dirs.values():
        os.makedirs(path, exist_ok=True)

    videos = extract_videos_from_folders(parent_directory)
    print(f"Total videos found: {len(videos)}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    target_fps = 10
    save_fps = 10

    with Pool(processes=4) as pool:
        process_func = partial(
            process_video,
            output_dirs=output_dirs,
            animal_classes=animal_classes,
            device=device,
            target_fps=target_fps,
            save_fps=save_fps,
        )
        results = pool.map(process_func, videos)

    df = pd.DataFrame(results)
    df.to_csv(csv_output_path, index=False)

    folder_file_counts = {animal: 0 for animal in animal_classes}
    for animal, folder_path in output_dirs.items():
        if os.path.exists(folder_path):
            folder_file_counts[animal] = len([
                file for file in os.listdir(folder_path) 
                if os.path.isfile(os.path.join(folder_path, file))
            ])

    print("\n--- File Summary Per Folder ---")
    for animal, count in folder_file_counts.items():
        print(f"{animal}: {count}")

    print(f"\nTotal videos processed: {len(videos)}")
    print(f"CSV report saved at {csv_output_path}.")
