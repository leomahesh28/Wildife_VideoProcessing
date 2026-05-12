import os
import cv2
import torch
import pandas as pd
import datetime
import pytesseract
import re
import gc
import multiprocessing as mp
from multiprocessing import Pool
from ultralytics import YOLO

# =========================
# Constants
# =========================
TARGET_FPS = 5
SAVE_FPS = 5
TEXT_ROI = [1020, 1080, 510, 1550]

pytesseract.pytesseract.tesseract_cmd = "tesseract"

# Global model for each worker
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# OCR Functions
# =========================
def ocr_get_text_roi(image):
    r1, r2, c1, c2 = TEXT_ROI
    roi = image[r1:r2, c1:c2]
    return roi


def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    enhanced = cv2.convertScaleAbs(blur, alpha=1.8, beta=30)
    _, thresh = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresh


def classify_token(token):
    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2,4}", token):
        return "date"
    if re.fullmatch(r"\d{1,2}:\d{2}[APMapm]{2}", token):
        return "time"
    if re.fullmatch(r"[A-Z]{3,6}\d{1,4}", token, re.IGNORECASE):
        return "cam"
    if re.match(r"\d{2,3}[fFoO]$", token):
        return "temp"
    return None


def ocr_get_texts(image):
    roi = ocr_get_text_roi(image)
    processed = preprocess_for_ocr(roi)
    config = r"--oem 3 --psm 7"

    raw_text = pytesseract.image_to_string(processed, config=config)
    tokens = raw_text.strip().split()
    tokens = [re.sub(r"[^\w:/]", "", t) for t in tokens if len(t.strip()) > 1]

    temp = date = time_val = cam = "UNKNOWN"

    for t in tokens:
        tag = classify_token(t)
        if tag == "temp" and temp == "UNKNOWN":
            temp = t
        elif tag == "date" and date == "UNKNOWN":
            date = t
        elif tag == "time" and time_val == "UNKNOWN":
            time_val = t
        elif tag == "cam" and cam == "UNKNOWN":
            cam = t

    return temp, date, time_val, cam, f"OCR tokens: {tokens}"


# =========================
# DetectionInfo Class
# =========================
class DetectionInfo:
    def __init__(self):
        self.id = 0
        self.infile = ""
        self.loc = ""
        self.has_animal = False
        self.labels = ""
        self.confidence = 0.00
        self.cam = ""
        self.date = ""
        self.fname = ""
        self.temp = ""
        self.time = ""
        self.camid = ""
        self.output_path = ""

    def update_filename(self, filename, order):
        fn = filename.split("/")
        N = len(fn)

        if N < 6:
            print("[WARNING] Data folder structure may be invalid.")
            self.fname = os.path.basename(filename)
            self.infile = filename
            self.id = order
            return

        self.loc = fn[N - 5]
        self.cam = fn[N - 4]

        date_folder = fn[N - 3]
        self.date = date_folder[:-6] if len(date_folder) > 6 else date_folder

        self.fname = fn[N - 1]
        self.infile = filename
        self.id = order


# =========================
# Helper Functions
# =========================
def save_info_csv(file_handle, info=None):
    if info is None:
        line = (
            "order,location,camera,date,filename,full_path,has_animal,"
            "confidence,temperature,time,camid,output_path\n"
        )
    else:
        line = (
            f"{info.id},{info.loc},{info.cam},{info.date},{info.fname},"
            f"{info.infile},{info.has_animal},{info.confidence},"
            f"{info.temp},{info.time},{info.camid},{info.output_path}\n"
        )

    file_handle.write(line)


def extract_videos_from_folders(parent_folder, video_extensions=None):
    if video_extensions is None:
        video_extensions = [".mp4", ".avi", ".mkv", ".mov", ".flv"]

    video_files = []

    for root, _, files in os.walk(parent_folder):
        for file in files:
            if file.startswith("._"):
                continue

            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))

    return video_files


def init_worker(model_path):
    global model, device

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"[INFO] Loading YOLO model on {device}...")

    model = YOLO(model_path)
    model.to(device)

    try:
        model.fuse()
    except Exception:
        pass

    print("[INFO] Model loaded successfully.")


# =========================
# Main Video Processing
# =========================
def process_video(args):
    global model, device

    video_file, order, animal_classes, output_dir_base = args

    info = DetectionInfo()
    info.update_filename(video_file, order)

    # -------------------------
    # OCR from first frame
    # -------------------------
    cap = cv2.VideoCapture(video_file)
    success, ocr_frame = cap.read()

    if success:
        try:
            ocr_result = ocr_get_texts(ocr_frame)

            if ocr_result and len(ocr_result) >= 4:
                temp, date_val, time_val, camid = ocr_result[:4]
                info.temp = temp
                info.date = date_val
                info.time = time_val
                info.camid = camid
            else:
                print(f"[WARNING] OCR returned too few fields for {video_file}")

        except Exception as e:
            print(f"[WARNING] OCR failed on {video_file}: {e}")

    cap.release()

    # -------------------------
    # Open video
    # -------------------------
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_file}")
        return info

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    if original_fps <= 0:
        original_fps = 30

    frame_skip = max(1, int(original_fps) // TARGET_FPS)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_width == 0 or frame_height == 0:
        print(f"[ERROR] Invalid video dimensions: {video_file}")
        cap.release()
        return info

    # -------------------------
    # Prepare output video
    # -------------------------
    save_dir = os.path.join(output_dir_base, "Other")
    os.makedirs(save_dir, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(video_file))[0]
    temp_out_path = os.path.join(save_dir, f"{video_name}_processed.mp4")

    out = cv2.VideoWriter(
        temp_out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        original_fps,
        (frame_width, frame_height),
    )

    if not out.isOpened():
        print(f"[ERROR] VideoWriter failed for: {temp_out_path}")
        cap.release()
        return info

    wildlife_frame_counts = {animal: 0 for animal in animal_classes[:-1]}
    wildlife_confidences = {animal: 0.0 for animal in animal_classes[:-1]}

    frame_count = 0

    # -------------------------
    # Frame processing loop
    # -------------------------
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_skip == 0:
            try:
                with torch.no_grad():
                    results = model.predict(
                        source=frame,
                        conf=0.40,
                        imgsz=640,
                        device=0 if device == "cuda" else "cpu",
                        verbose=False,
                    )

                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confs = result.boxes.conf.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy()

                        for box, conf, cls in zip(boxes, confs, classes):
                            class_name = model.names[int(cls)]

                            if class_name in animal_classes[:-1]:
                                wildlife_frame_counts[class_name] += 1
                                wildlife_confidences[class_name] += float(conf)

                                x1, y1, x2, y2 = map(int, box)

                                cv2.rectangle(
                                    frame,
                                    (x1, y1),
                                    (x2, y2),
                                    (0, 255, 0),
                                    2,
                                )

                                cv2.putText(
                                    frame,
                                    f"{class_name}: {conf:.2f}",
                                    (x1, max(20, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (0, 255, 0),
                                    2,
                                )

                del results

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[WARNING] CUDA OOM skipped frame: {video_file}")

                    if device == "cuda":
                        torch.cuda.empty_cache()

                    gc.collect()
                    continue
                else:
                    print(f"[ERROR] Inference failed for {video_file}: {e}")

            except Exception as e:
                print(f"[ERROR] Unexpected inference error for {video_file}: {e}")

        # Write every frame to preserve original video duration
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    if device == "cuda":
        torch.cuda.empty_cache()

    gc.collect()

    # -------------------------
    # Post-processing decision
    # -------------------------
    total_frames_processed = frame_count

    if total_frames_processed == 0:
        print(f"[WARNING] No frames processed: {video_file}")
        info.output_path = "NOT_SAVED"
        return info

    detected_class = max(wildlife_frame_counts, key=wildlife_frame_counts.get)
    max_detected_frames = wildlife_frame_counts[detected_class]

    avg_conf = (
        wildlife_confidences[detected_class] / max_detected_frames
        if max_detected_frames > 0
        else 0
    )

    MIN_FRAMES = max(2, int(0.005 * total_frames_processed))
    MIN_RATIO = 0.01
    MIN_CONF = 0.40

    detection_ratio = max_detected_frames / (total_frames_processed + 1e-6)

    if os.path.exists(temp_out_path):
        if (
            max_detected_frames >= MIN_FRAMES
            and detection_ratio >= MIN_RATIO
            and avg_conf >= MIN_CONF
        ):
            timestamp = datetime.datetime.now().strftime("%H%M%S")

            final_path = os.path.join(
                output_dir_base,
                detected_class,
                f"{video_name}_{timestamp}_processed.mp4",
            )

            os.makedirs(os.path.dirname(final_path), exist_ok=True)

            try:
                os.rename(temp_out_path, final_path)

                info.output_path = final_path
                info.labels = detected_class
                info.confidence = round(avg_conf, 3)
                info.has_animal = True

            except Exception as e:
                print(f"[ERROR] Failed to move file: {e}")
                info.output_path = temp_out_path

        else:
            info.output_path = temp_out_path
    else:
        print(f"[WARNING] Temp output video missing: {temp_out_path}")
        info.output_path = "NOT_SAVED"

    info.confidence = round(avg_conf, 3)

    return info


# =========================
# Full Pipeline
# =========================
def extract_all_videos_and_process(
    input_folder,
    model_path,
    output_csv,
    output_dir_base,
    animal_classes=[
        "Bear",
        "Bobcat",
        "Coyote",
        "Deer",
        "Elk",
        "MountainLion",
        "Other",
    ],
):
    videos = extract_videos_from_folders(input_folder)

    print(f"Total videos found: {len(videos)}")

    all_results = []

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(output_dir_base, exist_ok=True)

    args = [
        (video_file, idx, animal_classes, output_dir_base)
        for idx, video_file in enumerate(videos)
    ]

    with open(output_csv, "w") as f:
        save_info_csv(f, None)

        # IMPORTANT:
        # For a 4 GB GPU, use only 1 CUDA process.
        with Pool(
            processes=1,
            initializer=init_worker,
            initargs=(model_path,),
        ) as pool:
            for info in pool.imap_unordered(process_video, args):
                save_info_csv(f, info)
                f.flush()
                all_results.append(info)

                print(
                    f"[DONE] {info.fname} | "
                    f"Animal: {info.has_animal} | "
                    f"Label: {info.labels} | "
                    f"Conf: {info.confidence}"
                )

    df = pd.DataFrame([vars(info) for info in all_results])
    df.to_csv(output_csv, index=False)

    print("\nFile Summary Per Folder")
    for animal in animal_classes:
        folder_path = os.path.join(output_dir_base, animal)

        if os.path.exists(folder_path):
            count = len(
                [
                    file
                    for file in os.listdir(folder_path)
                    if os.path.isfile(os.path.join(folder_path, file))
                ]
            )
            print(f"{animal}: {count}")

    print(f"\nTotal videos processed: {len(videos)}")
    print(f"CSV report saved at: {output_csv}")


# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    torch.backends.cudnn.benchmark = True

    INPUT_FOLDER = "/media/maheshsharma/Expansion/Peterson Research Lab/WVC Raton Pass Phase #1 Project/Aim 2/MP 3 at grade/BRec14 MC#75"

    MODEL_PATH = "/home/maheshsharma/Downloads/trained_YOLOmodels/YOLOv26_bestpt_file/best.pt"

    OUTPUT_DIR_BASE = "/media/maheshsharma/Expansion/VideoProcesssed_Final/WVC Raton Pass Phase #1 Project/Aim 2/MP 3 at grade/BRec14 MC#75/Processed"

    #OUTPUT_CSV = "/media/maheshsharma/Expansion/VideoProcesssed_Final/WVC Raton Pass Phase #1 Project/Aim 2/Aim 2/MP 2 CBC/BRec54 MC#115 was BRec23 MC#84/Processed/full_video_results.csv"
    OUTPUT_CSV = os.path.join(OUTPUT_DIR_BASE, "full_video_procesed_results.csv")
    extract_all_videos_and_process(
        INPUT_FOLDER,
        MODEL_PATH,
        OUTPUT_CSV,
        OUTPUT_DIR_BASE,
    )