# Emotion Detection

A convolutional neural network (CNN) that recognizes seven facial expressions from grayscale face images and provides real-time webcam predictions.

The repository contains:

- A TensorFlow/Keras training and evaluation notebook.
- A saved Keras model for inference.
- A real-time OpenCV webcam application.
- Evaluation plots and metrics generated from the test set.
- An optional TensorFlow Lite model for lightweight deployment.

## Supported emotions

The model uses the following class-to-index mapping:

| Index | Label |
| ---: | --- |
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Sad |
| 5 | Surprise |
| 6 | Neutral |

## Project structure

```text
Emotion_Detection/
├── app.py                              # Real-time webcam inference
├── Emotion_Detection.ipynb             # Data loading, training, and evaluation
├── requirements.txt                    # Python dependencies
├── ML_Model/
│   └── emotion_detection_model.h5      # Model used by app.py
├── best_model.h5                       # Best training checkpoint
├── emotion_detection.tflite            # TensorFlow Lite export
├── Data_Sets/
│   ├── train/<emotion>/*.jpg           # Training images
│   └── test/<emotion>/*.jpg            # Test images
├── Evaluation_Metrics/
│   ├── classification_report.json
│   ├── roc_auc_values.json
│   ├── Confusion_Matrix.png
│   ├── F1_Scores.png
│   ├── ROC_Curve.png
│   ├── PR_Curve.png
│   ├── Radar_Chart.png
│   └── Confidence_Distribution.png
└── .gitignore
```

## Requirements

- Windows, macOS, or Linux
- Python 3.9 or 3.10 recommended for TensorFlow 2.10
- A working webcam for real-time inference
- TensorFlow 2.10.0
- OpenCV 4.7.0.72

The complete dependency list is in [`requirements.txt`](requirements.txt).

## Installation

Create and activate a virtual environment from the repository root:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks script execution for the current user, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate the environment again.

## Run real-time webcam detection

[`app.py`](app.py) opens the default camera, detects faces with OpenCV's Haar Cascade, preprocesses each face as a `48 x 48` grayscale image, and displays the predicted emotion and confidence.

Before running it, check the model path near the top of `app.py`. The current source contains an absolute Windows path:

```python
model = load_model(
    r'D:\ML_project\Emotion_Detection\ML_Model\emotion_detection_model.h5'
)
```

Replace it with a path valid on your machine if the repository is located elsewhere. From the repository root, run:

```powershell
python app.py
```

Press **q** while the camera window is focused to stop the application.

### Inference pipeline

1. Capture a frame from camera index `0`.
2. Convert the frame to grayscale for face detection.
3. Detect faces using `haarcascade_frontalface_default.xml`.
4. Crop each detected face.
5. Resize and normalize the crop to `(48, 48, 1)` with pixel values in `[0, 1]`.
6. Run the Keras model and select the highest-probability class.
7. Draw the face bounding box, emotion label, and confidence on the frame.

The application uses `ML_Model/emotion_detection_model.h5`. The root-level `best_model.h5` is the checkpoint produced during notebook training and is not loaded by `app.py` unless it is copied or the path is changed.

## Train and evaluate the model

Open [`Emotion_Detection.ipynb`](Emotion_Detection.ipynb) in Jupyter or VS Code and run the cells in order. The notebook:

1. Loads images from `Data_Sets/train` and `Data_Sets/test`.
2. Converts images to grayscale and resizes them to `48 x 48`.
3. Creates a validation split from the training data.
4. Builds and trains the CNN with categorical cross-entropy and Adam.
5. Uses model checkpointing, early stopping, and learning-rate reduction.
6. Evaluates the model with accuracy, a classification report, a confusion matrix, ROC curves, precision-recall curves, F1 scores, and confidence distributions.
7. Saves the best checkpoint as `best_model.h5`.
8. Exports a TensorFlow Lite model as `emotion_detection.tflite`.

The notebook expects these relative paths:

```text
Data_Sets/train
Data_Sets/test
```

Run the notebook with the repository root as its working directory so those paths resolve correctly.

### Model architecture

The CNN accepts a single-channel `(48, 48, 1)` image and has:

- Three convolutional blocks with 64, 128, and 256 filters.
- Batch normalization, max pooling, and dropout in the convolutional blocks.
- Fully connected layers with 1024 and 512 units.
- A seven-unit softmax output layer.
- Approximately 11.1 million parameters.

Training uses Adam with an initial learning rate of `0.0001`. The notebook configures early stopping with a patience of 15 epochs and reduces the learning rate when validation loss plateaus.

## Evaluation results

The committed evaluation artifacts were generated on 7,178 test images:

| Metric | Result |
| --- | ---: |
| Accuracy | 67.15% |
| Macro precision | 0.662 |
| Macro recall | 0.639 |
| Macro F1 | 0.645 |
| Weighted precision | 0.670 |
| Weighted recall | 0.671 |
| Weighted F1 | 0.666 |

Per-class results from [`classification_report.json`](Evaluation_Metrics/classification_report.json):

| Emotion | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Angry | 0.599 | 0.632 | 0.615 |
| Disgust | 0.698 | 0.541 | 0.609 |
| Fear | 0.563 | 0.392 | 0.462 |
| Happy | 0.862 | 0.879 | 0.870 |
| Sad | 0.556 | 0.520 | 0.538 |
| Surprise | 0.777 | 0.776 | 0.777 |
| Neutral | 0.576 | 0.731 | 0.644 |

One-vs-rest ROC AUC values are available in [`roc_auc_values.json`](Evaluation_Metrics/roc_auc_values.json). The plots are available in [`Evaluation_Metrics/`](Evaluation_Metrics/).

## Dataset layout

Each split is organized by class directory:

```text
Data_Sets/
├── train/
│   ├── angry/
│   ├── disgusted/
│   ├── fearful/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprised/
└── test/
    ├── angry/
    ├── disgusted/
    ├── fearful/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprised/
```

The current dataset contains 28,709 training images and 7,178 test images. The classes are imbalanced, particularly `disgusted`, so aggregate accuracy should be interpreted together with per-class recall and F1 scores.

## TensorFlow Lite

The repository includes [`emotion_detection.tflite`](emotion_detection.tflite), a TensorFlow Lite export created by the notebook. It is not currently used by [`app.py`](app.py). A TensorFlow Lite application must separately implement image preprocessing, interpreter input/output handling, and face detection.

## Troubleshooting

- **Model file not found:** update the absolute model path in `app.py`, or use a path relative to the repository.
- **Camera cannot be opened:** check camera permissions, close other camera applications, and verify that camera index `0` is correct.
- **No face is detected:** improve lighting, face the camera, and keep the face reasonably close to the camera.
- **TensorFlow installation fails:** use a supported Python version for TensorFlow 2.10 and install the dependencies inside a fresh virtual environment.
- **Notebook cannot find the dataset:** set the notebook working directory to the repository root; the notebook uses relative `Data_Sets/train` and `Data_Sets/test` paths.

## Limitations and responsible use

This model predicts facial-expression categories from a camera image; it does not reliably determine a person's internal emotional state. Predictions can be affected by lighting, pose, occlusion, image quality, dataset imbalance, and demographic or cultural variation. Do not use the output as the sole basis for decisions about people, access, employment, education, health, or safety. Obtain consent before processing camera images and avoid storing or sharing frames without a valid purpose.

## License

No license file is currently included in the repository. Add an appropriate license before distributing or reusing the project.