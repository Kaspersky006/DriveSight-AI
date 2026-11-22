# DriveSight-AI

## Vision-Based Camera Health Monitoring for Autonomous Vehicles

DriveSight-AI is a deep learning project that automatically detects camera degradation conditions in autonomous driving systems using a Convolutional Neural Network (CNN).

The system classifies images into different camera health conditions and provides confidence scores and maintenance recommendations. The project demonstrates a complete end-to-end machine learning pipeline, from dataset generation and preprocessing to model training, evaluation, and inference.

---

## Features

- Synthetic dataset generation from BDD10K
- Camera degradation simulation
- Custom CNN implemented in PyTorch
- Model training with checkpoint saving
- Evaluation using multiple performance metrics
- Confusion matrix generation
- Single image prediction
- Batch prediction with CSV export
- Modular project structure

---

## Camera Conditions

The model classifies the following camera conditions:

- Clean
- Fog
- Rain
- Motion Blur
- Defocus Blur
- Overexposure
- Dirty Lens
- Partial Occlusion

---

## Project Structure

```text
DriveSight-AI/

├── data/
├── outputs/
├── src/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Tech Stack

- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Pillow
- Scikit-learn
- Matplotlib

---

## Installation

```bash
git clone <repository-url>

cd DriveSight-AI

pip install -r requirements.txt
```

---

## Training

```bash
python src/train.py
```

The best model is automatically saved as:

```text
outputs/models/best_model.pth
```

---

## Model Evaluation

```bash
python src/evaluate.py
```

Evaluation generates:

- Classification Report
- Confusion Matrix
- Test Accuracy

Saved under:

```text
outputs/models/metrics/
```

---

## Single Image Prediction

```bash
python src/predict.py --image path/to/image.jpg
```

Example output:

```text
Prediction : fog
Confidence : 91.34%

Recommendation:
Heavy fog detected.
Visibility may be reduced.
```

---

## Batch Prediction

```bash
python src/batch_predict.py --folder path/to/images
```

Results are exported to:

```text
outputs/predictions/predictions.csv
```

---

## Dataset

This project uses images from the BDD10K dataset.

Additional camera degradation conditions are synthetically generated to simulate common perception failures encountered in autonomous driving systems.

---

## Future Improvements

- Improve model accuracy
- Hyperparameter optimization
- Transfer learning using ResNet
- Real-world camera degradation dataset
- Real-time video inference
- Deployment using Streamlit

---

## License

This project is released under the MIT License.