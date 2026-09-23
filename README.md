# Iranian License Plate Detection and Recognition

An end-to-end deep learning pipeline for detecting and recognizing Iranian license plates using **YOLO** and **PyTorch**.

The project consists of two main stages:

1. **License Plate Detection (LPD)** using YOLO
2. **License Plate Recognition (LPR)** using CNN-based multi-head character recognition

The final pipeline detects one or more license plates in an image, crops each detected plate, recognizes its eight logical positions, and returns the decoded plate number.

---

## Pipeline

```text
Input image
    ↓
YOLO License Plate Detector
    ↓
Plate bounding box
    ↓
Crop detected plate
    ↓
License Plate Recognition Model
    ↓
Eight position-specific predictions
    ↓
Decode class indices
    ↓
Final license plate string
```

---

## License Plate Detection

The detection stage uses an Ultralytics YOLO model trained to detect Iranian license plates.

The detector was evaluated using:

- Precision
- Recall
- mAP@50
- mAP@50–95

The trained detector is later used in the end-to-end pipeline to extract plate crops for recognition.

---

## License Plate Recognition

Iranian license plates are represented as eight **logical positions**:

```text
digit | digit | Persian letter | digit | digit | digit | digit | digit
```

The Persian-letter position required special preprocessing because some labels contain multi-character Unicode representations.

For example:

- `ه‍` contains `ه` followed by a zero-width joiner.
- `الف` consists of multiple Unicode characters but represents one logical plate class.

Labels are therefore parsed structurally as:

```text
2 leading digits + Persian-letter token + 5 trailing digits
```

rather than assuming every plate contains exactly eight Unicode characters.

---

## Recognition Models

### Custom CNN

A custom PyTorch CNN was trained from scratch.

Its architecture follows this structure:

```text
Input: [3, 64, 256]
        ↓
ConvBlock
        ↓
ConvBlock
        ↓
ConvBlock
        ↓
ConvBlock
        ↓
Feature Map: [256, 4, 16]
        ↓
AdaptiveAvgPool2d((1, 8))
        ↓
8 Position-Specific Feature Vectors
        ↓
8 Independent Classification Heads
```

Seven output heads predict digits, while one head predicts the Persian-letter class.

Before full training, a small-batch overfitting test was performed to verify that:

- label encoding was correct
- the loss function worked
- gradients propagated correctly
- the optimizer could update model parameters

The model successfully memorized the small debugging batch before full training.

---

### MobileNetV3-Small

A pretrained **MobileNetV3-Small** backbone was adapted for license plate recognition.

The original ImageNet classifier was removed and replaced with eight position-specific classification heads.

Training was performed in two stages:

1. Train the new classification heads while keeping the pretrained backbone frozen.
2. Unfreeze the full backbone and fine-tune the network on the license plate dataset using a smaller learning rate.

The frozen pretrained features alone performed poorly on this fine-grained recognition task. Performance improved substantially after the MobileNet backbone was unfrozen and fine-tuned.

---

## Recognition Results

Both models were evaluated on the same untouched test set.

| Model | Test Loss | Character Accuracy | Full Plate Accuracy | Misclassified Plates | Parameters |
|---|---:|---:|---:|---:|---:|
| Custom CNN | 0.1316 | 96.07% | 82.54% | 429 | 413,020 |
| MobileNetV3-Small | 0.1044 | 96.98% | 86.16% | 340 | 980,092 |

MobileNetV3-Small improved character accuracy by approximately **0.91 percentage points** and full-plate accuracy by approximately **3.62 percentage points** compared with the custom CNN.

Because a full plate is considered correct only when **all eight positions are predicted correctly**, relatively small character-level improvements can lead to larger differences in full-plate accuracy.

MobileNetV3-Small also reduced the number of misclassified plates from **429 to 340**.

---

## Evaluation Metrics

Recognition models were evaluated using:

- Training loss
- Validation loss
- Character-level accuracy
- Full-plate accuracy
- Misclassified plate count
- Gradient norm
- Learning rate history

The notebook also includes visualizations of:

- training and validation loss curves
- character accuracy curves
- full-plate accuracy curves
- misclassified recognition examples
- YOLO detection predictions
- end-to-end detection and recognition results

---

## End-to-End Pipeline

The final system combines the trained detector and the best recognition model.

```text
Input image
    ↓
YOLO License Plate Detector
    ↓
Plate bounding box
    ↓
Crop detected plate
    ↓
MobileNetV3-Small Recognizer
    ↓
Eight position-specific predictions
    ↓
Decode class indices
    ↓
Final license plate string
```

The pipeline supports images containing multiple detected license plates.

For each detected plate, it returns:

- bounding box coordinates
- detection confidence
- predicted plate text
- confidence for each predicted character
- average recognition confidence
- cropped plate image

---

## Project Structure

```text
license_plate_detection/
│
├── Licence_Plate_detection.ipynb
├── requirements.txt
├── .gitignore
│
├── configs/
│   └── data.yaml
│
├── LPD_FILES/
├── LPR_FILES/
│
├── models/
│   ├── detector/
│   └── recognizer/
│       ├── custom_cnn/
│       └── mobilenet_v3_small/
│           ├── frozen_backbone/
│           └── fine_tuning/
│
├── logs/
├── outputs/
└── runs/
```

The datasets, trained weights, virtual environment, training logs, and generated outputs are excluded from Git where appropriate through `.gitignore`.

---

## Installation

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The project uses PyTorch and supports Apple Silicon acceleration through the **MPS** backend when available.

---

## Main Dependencies

- PyTorch
- torchvision
- Ultralytics
- NumPy
- pandas
- matplotlib
- Pillow
- OpenCV
- scikit-learn
- PyYAML
- tqdm
- Jupyter / ipykernel

---

## Running the Project

Open:

```text
Licence_Plate_detection.ipynb
```

and run the notebook sequentially.

The notebook covers:

```text
Dataset inspection and preparation
        ↓
YOLO detector preparation and evaluation
        ↓
Recognition dataset preparation
        ↓
Unicode-aware label parsing
        ↓
Label encoding
        ↓
Custom CNN training
        ↓
MobileNetV3-Small transfer learning
        ↓
Model evaluation and comparison
        ↓
End-to-end detection and recognition pipeline
```

---

## Final System

The final `LicensePlatePipeline` performs the following steps:

1. Receives an input image.
2. Detects license plates using YOLO.
3. Crops each detected plate.
4. Applies recognition preprocessing.
5. Predicts all eight logical plate positions.
6. Decodes the predicted class indices into plate text.
7. Returns the final prediction together with detection and recognition information.
