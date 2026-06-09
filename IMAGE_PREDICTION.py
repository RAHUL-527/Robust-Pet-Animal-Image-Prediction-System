import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import json
import os
import urllib.request


#Function to load and preprocess images
def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


#Function to load ImageNet class labels
def load_class_names():
    class_idx_path = 'dataset/imagenet_class_index.json'
    if not os.path.exists(class_idx_path):
        print("Downloading ImageNet class index...")
        url = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
        urllib.request.urlretrieve(url, class_idx_path)
        print("Download completed.")

    with open(class_idx_path) as f:
        class_idx = json.load(f)
    return {int(key): value[1] for key, value in class_idx.items()}


#Function to load the models
def load_models():
    try:
        print("Loading models...")

        # Fix Windows path issue
        cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
        
        # MobileNetV2
        mobilenet_path = os.path.join(cache_dir, "mobilenet_v2-b0353104.pth")
        if not os.path.exists(mobilenet_path):
            print("Downloading MobileNetV2 model...")
            torch.hub.download_url_to_file(
                "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth", mobilenet_path
            )

        mobilenet_model = models.mobilenet_v2(weights="DEFAULT")
        mobilenet_model.load_state_dict(torch.load(mobilenet_path, map_location=torch.device('cpu')))
        mobilenet_model.eval()

        # ResNet50
        resnet_path = os.path.join(cache_dir, "resnet50-11ad3fa6.pth")
        if not os.path.exists(resnet_path):
            print("Downloading ResNet50 model...")
            torch.hub.download_url_to_file(
                "https://download.pytorch.org/models/resnet50-11ad3fa6.pth", resnet_path
            )

        resnet_model = models.resnet50(weights="DEFAULT")
        resnet_model.load_state_dict(torch.load(resnet_path, map_location=torch.device('cpu')))
        resnet_model.eval()

        # Vision Transformer (ViT)
        vit_path = os.path.join(cache_dir, "vit_b_16-c867db91.pth")
        if not os.path.exists(vit_path):
            print("Downloading Vision Transformer model...")
            torch.hub.download_url_to_file(
                "https://download.pytorch.org/models/vit_b_16-c867db91.pth", vit_path
            )

        vit_model = models.vit_b_16(weights="DEFAULT")
        vit_model.load_state_dict(torch.load(vit_path, map_location=torch.device('cpu')))
        vit_model.eval()

        print("Models loaded successfully!")
        return mobilenet_model, resnet_model, vit_model
    except Exception as e:
        print(f"An error occurred while loading models: {e}")
        return None, None, None


#Function to perform image recognition
def recognize_image(image_path, mobilenet_model, resnet_model, vit_model, idx_to_label):
    try:
        image_tensor = load_and_preprocess_image(image_path)

        # MobileNetV2 prediction
        with torch.no_grad():
            mob_preds = mobilenet_model(image_tensor)
            mob_probs = torch.nn.functional.softmax(mob_preds[0], dim=0)
            mob_conf, mob_class = torch.max(mob_probs, dim=0)

        # ResNet50 prediction
        with torch.no_grad():
            res_preds = resnet_model(image_tensor)
            res_probs = torch.nn.functional.softmax(res_preds[0], dim=0)
            res_conf, res_class = torch.max(res_probs, dim=0)

        # Vision Transformer (ViT) prediction
        with torch.no_grad():
            vit_preds = vit_model(image_tensor)
            vit_probs = torch.nn.functional.softmax(vit_preds[0], dim=0)
            vit_conf, vit_class = torch.max(vit_probs, dim=0)

        # Map class indices to class names
        mob_label = idx_to_label[mob_class.item()]
        res_label = idx_to_label[res_class.item()]
        vit_label = idx_to_label[vit_class.item()]

        return {
            "MobileNetV2": (mob_label, mob_conf.item()),
            "ResNet50": (res_label, res_conf.item()),
            "Vision Transformer": (vit_label, vit_conf.item())
        }
    except Exception as e:
        print(f"An error occurred during image recognition: {e}")
        return None


#Function to display predictions
def display_predictions(image_path, predictions):
    try:
        image = Image.open(image_path)
        plt.imshow(image)
        plt.axis("off")
        plt.title("Predictions")
        plt.show()

        for model, (label, confidence) in predictions.items():
            print(f"\n{model} Prediction:")
            print(f"Class: {label}, \nConfidence: {confidence * 100:.2f}%")
    except Exception as e:
        print(f"An error occurred during display: {e}")


#Main function to run the process
def main(image_path):
    idx_to_label = load_class_names()
    mobilenet_model, resnet_model, vit_model = load_models()

    if mobilenet_model and resnet_model and vit_model:
        predictions = recognize_image(image_path, mobilenet_model, resnet_model, vit_model, idx_to_label)
        if predictions:
            display_predictions(image_path, predictions)
        else:
            print("Failed to recognize the image.")
    else:
        print("Model loading failed. Exiting...")


#Run the function with a test image
image_path = 'assets/test_image_2.jpg'  # Replace with your image file path
main(image_path)
