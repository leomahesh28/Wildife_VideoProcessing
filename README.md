# Wildlife Video Processing Pipeline using YOLO

A high-performance wildlife video processing pipeline for **animal detection in camera-trap videos** using a trained YOLO model, **OCR-based metadata extraction**, and **automated video organization**.

This pipeline is designed for **ecological monitoring, wildlife–vehicle collision (WVC) studies, and large-scale camera-trap video analysis**.

---

## Features

✔ **YOLO-based wildlife detection**  
✔ **OCR metadata extraction**:
- Temperature
- Date
- Time
- Camera ID

✔ **Annotated video generation** with bounding boxes  
✔ **Confidence-based wildlife detection filtering**  
✔ **Automatic organization of videos into class folders**  
✔ **CSV report generation**  
✔ **CUDA GPU acceleration**  
✔ **Memory-optimized inference for low-VRAM GPUs**  
✔ **Corrupted video handling**  
✔ **macOS hidden file filtering (`._` files)**

---

## Supported Wildlife Classes

The trained model currently supports the following wildlife classes:

| Class |
|--------|
| Bear |
| Bobcat |
| Coyote |
| Deer |
| Elk |
| MountainLion |
| Other |

---

## Project Structure

### Input Folder Structure

Organize input videos in the following hierarchy:

```text
Input_Folder/
│── Location/
│   ├── Camera/
│   │   ├── Date/
│   │   │   ├── video1.mp4
│   │   │   ├── video2.mp4
│   │   │   ├── video3.mp4
```

### Output Folder Structure

Processed videos are automatically organized into wildlife-specific folders.

```text
Processed/
│── Bear/
│── Bobcat/
│── Coyote/
│── Deer/
│── Elk/
│── MountainLion/
│── Other/
│── full_video_processed_results.csv
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/leomahesh28/Wildife_VideoProcessing.git

```

---

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

---

### 3. Activate Virtual Environment

#### Linux / macOS

```bash
source .venv/bin/activate
```

#### Windows

```cmd
.venv\Scripts\activate
```

---

### 4. Install Dependencies

Install required Python packages:

```bash
pip install ultralytics
pip install torch torchvision
pip install opencv-python
pip install pandas
pip install pytesseract
```

Or install everything together:

```bash
pip install ultralytics torch torchvision opencv-python pandas pytesseract
```

---

## Install Tesseract OCR

### Ubuntu / Linux

```bash
sudo apt update
sudo apt install tesseract-ocr
```

### Verify Installation

```bash
tesseract --version
```

---

## GPU Optimization (Recommended)

For systems with **low GPU VRAM (e.g., NVIDIA T1000 4GB)**, use:

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

This helps reduce **CUDA memory fragmentation**.

---

## Configuration

Update the following paths in the script:

```python
INPUT_FOLDER = "/path/to/input/videos"

MODEL_PATH = "/path/to/best.pt"

OUTPUT_DIR_BASE = "/path/to/output/folder"

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR_BASE,
    "full_video_processed_results.csv"
)
```

---

## Detection Parameters

### Confidence Threshold

Current threshold:

```python
conf = 0.40
```

- **Lower value** → More detections, higher false positives  
- **Higher value** → Fewer detections, possible missed animals

---

### Frame Processing Speed

Current processing frame rate:

```python
TARGET_FPS = 5
```

This means:

- The model analyzes **selected frames only**
- **Original video duration is preserved**
- Faster processing for large datasets

---

## OCR Metadata Extraction

The pipeline extracts metadata directly from video overlays using OCR.

### Extracted Metadata

- Temperature
- Date
- Time
- Camera ID

### OCR Region of Interest (ROI)

```python
TEXT_ROI = [1020, 1080, 510, 1550]
```

Modify this region depending on your camera overlay position.

---

## CSV Output

The pipeline automatically generates:

```text
full_video_processed_results.csv
```

### CSV Fields

| Column | Description |
|----------|-------------|
| order | Video processing order |
| location | Location folder |
| camera | Camera folder |
| date | Recording date |
| filename | Original video name |
| full_path | Input video path |
| has_animal | True / False |
| confidence | Detection confidence |
| temperature | OCR temperature |
| time | OCR time |
| camid | Camera ID |
| output_path | Saved processed video path |

---

## Performance Notes

This pipeline is optimized for:

- NVIDIA T1000 (4GB VRAM)
- CUDA memory efficiency
- Large-scale ecological datasets
- Long camera-trap videos

### Key Optimizations

- **Single GPU worker**

```python
Pool(processes=1)
```

- **Memory cleanup**

```python
torch.cuda.empty_cache()
gc.collect()
```

- **Efficient inference**

```python
torch.no_grad()
```

- **Corrupted video handling**
- **Hidden macOS file filtering**

---

## Running the Pipeline

Run the script:

```bash
python VideoProcessingCode.py
```

Example output:

```text
[INFO] Loading YOLO model on cuda...
[INFO] Model loaded successfully.

Total videos found: 250

[DONE] video_001.mp4 | Animal: True | Label: Deer | Conf: 0.91
[DONE] video_002.mp4 | Animal: False | Label: | Conf: 0.00

CSV report saved at:
Processed/full_video_processed_results.csv
```

---

## Common Issues

### CUDA Out of Memory (OOM)

If you encounter CUDA memory issues:

Use:

```python
Pool(processes=1)
```

and run:

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

---

### CSV File Not Saving

Ensure:

```python
OUTPUT_CSV = os.path.join(
    OUTPUT_DIR_BASE,
    "full_video_processed_results.csv"
)
```

Avoid incorrect or duplicated folder paths.

---

### Tesseract Not Found

Install Tesseract:

```bash
sudo apt install tesseract-ocr
```

Verify:

```bash
tesseract --version
```

---

## Research Applications

This pipeline is suitable for:

- Wildlife monitoring
- Wildlife–vehicle collision (WVC) studies
- Ecological monitoring
- Camera-trap analysis
- Automated biodiversity assessment
- Large-scale wildlife video annotation

---


