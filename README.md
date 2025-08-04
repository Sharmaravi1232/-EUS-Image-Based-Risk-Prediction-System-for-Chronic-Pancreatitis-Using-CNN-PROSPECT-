🧠 EUS Image-Based Risk Prediction System for Chronic Pancreatitis Using CNN (PROSPECT)
A deep learning-based pipeline for Endoscopic Ultrasound (EUS) image classification, designed to predict chronic pancreatitis risk using a fine-tuned InceptionV3 CNN architecture.

📌 Overview
PROSPECT is a PyTorch-based medical imaging system that classifies and assesses risk for chronic pancreatitis using grayscale ultrasound images. It simulates real-world scenarios using a CSV-formatted dataset inspired by the MNIST structure and includes transfer learning, custom preprocessing, auxiliary loss training, and model evaluation.

🚀 Key Features
🗃️ Custom Dataset Class for CSV-based grayscale image extraction

🧠 InceptionV3 CNN Architecture with auxiliary outputs for improved learning

🧪 Transfer Learning on pretrained ImageNet weights

🧼 Image Preprocessing: Resizing, normalization, grayscale-to-RGB conversion

🎯 Evaluation Metrics: Accuracy & Weighted F1-Score

💾 Model Checkpointing: Save best-performing model

📦 Hugging Face Deployment: Fully deployable as an interactive inference tool

📚 Dataset
Source: mnist_train_small.csv

Format:
Each row contains:

1 label (0–9)

784 pixel values representing a 28x28 grayscale image

Preprocessing:

Convert row to 28x28 numpy array

Transform to PIL Image and resize to 299×299 (required by InceptionV3)

Normalize using ImageNet mean and std

Convert grayscale to RGB

🏗️ Model Architecture
Base: InceptionV3 (pretrained on ImageNet)

Modified final FC layer for 10-class classification

Auxiliary classifiers enabled for intermediate supervision

Optimizer: Adam / SGD

Loss: CrossEntropy with auxiliary loss weighted

🎓 Training
Runs with early stopping and model checkpointing

Validation runs after each epoch

Logs metrics:

Accuracy

Weighted F1-Score

Final model saved as:

Copy
Edit
best_inception_mnist.pth
📊 Evaluation Metrics
✅ Metric 1: Overall Accuracy

✅ Metric 2: Weighted F1-Score (to address class imbalance)

⚙️ Tech Stack
Python 3.11+

PyTorch

Torchvision

scikit-learn

Pillow (PIL)

Pandas

NumPy

🌐 Deployment on Hugging Face Spaces
The model is deployed as a live interactive web app using Gradio on Hugging Face Spaces.

🔗 Live App:

🚀 Deployment Details:
Built using Gradio as frontend

app.py defines the inference pipeline

requirements.txt includes all dependencies

Model (best_inception_mnist.pth) loaded at runtime

