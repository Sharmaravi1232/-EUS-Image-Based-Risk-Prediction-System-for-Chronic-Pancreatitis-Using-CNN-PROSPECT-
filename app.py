import gradio as gr
import torch
from torchvision import transforms
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the TorchScript model
model = torch.jit.load("inception_mnist_traced.pt", map_location=device)
model.eval()

# Define preprocessing (must match training transforms)
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Prediction function
def predict(image):
    if image is None:
        return "No image provided."
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).item()
    return f"Predicted Digit: {pred}"

# Gradio Interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(image_mode="L", label="Upload a 28x28 grayscale digit image"),
    outputs="text",
    title="MNIST Digit Classifier (InceptionV3)",
    description="Upload a digit image (0–9) and get the predicted digit using a PyTorch InceptionV3 model."
)

demo.launch()
