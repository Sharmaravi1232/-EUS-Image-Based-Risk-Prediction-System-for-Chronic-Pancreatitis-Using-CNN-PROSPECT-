# -EUS-Image-Based-Risk-Prediction-System-for-Chronic-Pancreatitis-Using-CNN-PROSPECT-
A PyTorch pipeline for classifying handwritten digits from a CSV-based MNIST dataset using InceptionV3. Images are extracted, transformed to grayscale, resized, and passed through a fine-tuned pretrained model. Includes custom dataset, transfer learning, auxiliary loss, and evaluation via accuracy &amp; F1-score.
Overview
PROSPECT is a deep learning-based medical imaging pipeline designed to classify and assess chronic pancreatitis risk using Endoscopic Ultrasound (EUS) images. This system uses a customized InceptionV3 CNN architecture, fine-tuned on image data derived from a CSV-formatted MNIST structure, simulating medical images.
 Features
 Custom Dataset Class for CSV-based grayscale image extraction

 InceptionV3 Architecture with auxiliary outputs for enhanced learning

 Image Preprocessing pipeline to convert and normalize grayscale data

 Transfer Learning on pretrained ImageNet weights

 Evaluation Metrics: Accuracy and Weighted F1-Score

 Model Checkpointing based on best validation accuracy

 Dataset
Source: CSV-formatted dataset (mnist_train_small.csv)

Each row includes a label followed by 784 pixel values representing a 28x28 grayscale image

Images are reshaped, converted to PIL format, and resized to 299x299 for model input.

Tech Stack
Python 3.11+

PyTorch

Torchvision

scikit-learn

PIL (Pillow)

Pandas, NumPy
 1. Load dataset from CSV
 2. Preprocess images (Resize, Normalize, Grayscale to RGB)
 3. Load InceptionV3 model with ImageNet weights
 4. Modify FC layers for 10-class classification
 5. Train using auxiliary loss
 6. Evaluate using accuracy and F1 score
    Evaluation
Metric 1: Accuracy

Metric 2: F1-Score (Weighted)

Validation occurs after each epoch with checkpointing of best model weights.

Model Output
Best-performing model is saved as:

bash
Copy
Edit
best_inception_mnist.pth
References
PyTorch InceptionV3 Documentation

MNIST Dataset Format


