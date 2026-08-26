# Emotion Detection

A CNN-based emotion detection system that classifies facial emotions into 7 categories using TensorFlow/Keras and OpenCV.

## Overview

This project implements a convolutional neural network trained to detect and classify human emotions from facial images. The system includes a graphical user interface (GUI) for easy interaction, allowing users to upload images and receive real-time emotion predictions with confidence scores.

**Emotions Detected:** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprised

## Requirements

- **Python:** 3.11+ (tested with 3.11.5)
- **TensorFlow:** 2.15.0
- **OpenCV:** 4.8.1.78
- **Pillow (PIL):** 10.1.0
- **NumPy:** 1.26.2

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/tafeemamair/Emotion_Detection.git
cd Emotion_Detection
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the GUI

After installation, start the emotion detection GUI:

```bash
python gui.py
```

The GUI window will open. To detect emotions:

1. **Click "Upload Image"** - Select a JPG, PNG, or BMP image containing faces
2. **Click "Detect Emotion"** - The model will analyze detected faces and display:
   - Predicted emotion (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprised)
   - Confidence score (percentage)

## Model Architecture

The model is a custom 3-layer Convolutional Neural Network (CNN):

- **Input:** 48×48 grayscale images
- **Layer 1:** Conv2D (128 filters, 3×3 kernel) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
- **Flatten:** 73,728 units
- **Dense Layer:** 256 units → BatchNorm → ReLU → Dropout(0.25)
- **Output:** Softmax (7 emotion classes)
- **Total Parameters:** ~18.9M

**Compilation:**
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Metrics: Accuracy

## Model Performance

**Training Results (15 epochs, 28,709 training images):**
- Training Accuracy: 92.09%
- Validation Accuracy: 49.73%
- Training Loss: 0.2429
- Validation Loss: 1.9512

**Note:** The 42% gap between training and validation accuracy indicates significant overfitting. The model performs well on the training data but has limited generalization to unseen faces. This is typical for small CNNs on limited facial emotion datasets and is a known limitation.

## Dataset

Trained on approximately 28,709 training images and 7,178 validation images across 7 emotion classes. Images are 48×48 pixels in grayscale format.

**Dataset Source:** Derived from a facial emotion recognition dataset (based on training/test directory structure)

## Face Detection

The system uses **Haar Cascade Classifiers** from OpenCV for face detection:
- Classifier: `haarcascade_frontalface_default.xml`
- Detection Parameters: Scale Factor = 1.3, Min Neighbors = 5, Min Size = 40×40

**Limitations:**
- Works best on frontal, well-lit faces
- May struggle with profile views, poor lighting, or partially occluded faces
- Can produce false positives/negatives depending on image conditions

## Project Files

```
Emotion_Detection/
├── gui.py                              # Main GUI application
├── ED_model.ipynb                      # Training notebook
├── model_a.json                        # Model architecture (JSON)
├── model_weights1.h5                   # Trained model weights
├── haarcascade_frontalface_default.xml # Haar Cascade for face detection
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## Known Limitations

1. **Overfitting:** Large train-test accuracy gap suggests the model memorizes training data
2. **Limited Generalization:** Poor performance on unseen faces, especially with different:
   - Lighting conditions
   - Face angles and orientations
   - Demographics different from training data
3. **Face Detection:** Haar Cascade is outdated; modern detectors (MTCNN, RetinaFace) would be more robust
4. **No Confidence Threshold Tuning:** Currently uses fixed threshold (30%) for "uncertain" classifications
5. **Single Face Per Image:** Displays prediction for highest-confidence detection only

## Future Improvements

- [ ] Implement data augmentation to reduce overfitting
- [ ] Use transfer learning (MobileNetV2, EfficientNet) for better generalization
- [ ] Replace Haar Cascade with modern face detector (MTCNN, MediaPipe Face)
- [ ] Add confidence-based uncertainty estimation
- [ ] Support for real-time webcam input
- [ ] REST API for integration with other applications
- [ ] Batch processing for multiple images
- [ ] Per-emotion performance metrics (precision, recall, F1-score)

## Error Handling

The application includes robust error handling for:

- Missing model files or weights
- Invalid image formats
- No faces detected in image
- File read errors
- User cancellation of file dialog

All errors are logged to console and displayed to the user with descriptive messages.

## Development Notes

**Training Details (from notebook):**
- Framework: TensorFlow 2.15.0 with Keras API
- Batch Size: 64
- Epochs: 15
- No explicit data augmentation applied
- Validation split: test/ directory (7,178 images)

**Preprocessing:**
- Images converted to grayscale during training and inference
- Resized to 48×48 pixels
- Pixel values in range 0-255 (not normalized)

## License

This project is provided as-is for educational and personal use.

## Author

Aisan Tafeem Amair

---

**Last Updated:** August 2026
