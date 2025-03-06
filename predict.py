import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from collections import defaultdict

MODEL_FOLDER = "models/" 
IMAGE_PATH = "custom imgs.jpg"
device = "cuda" if torch.cuda.is_available() else "cpu"

def preprocess_image(image_path, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

def load_models(models_folder):
    models = {}
    for filename in os.listdir(models_folder):
        if filename.endswith(".pt") or filename.endswith(".pth"):
            model_path = os.path.join(models_folder, filename)
            model = torch.load(model_path, map_location=device)
            model.eval() 
            models[filename] = model
    return models

def predict_with_all_models(models, image_tensor):
    predictions = dict(list)  
    model_results = {} 

    for model_name, model in models.items():
        with torch.inference_mode():
            outputs = model(image_tensor)  # Forward pass
            probabilities = torch.softmax(outputs, dim=1)  # Convert to probabilities
            confidence, predicted_class = torch.max(probabilities, 1)  # Get top class

        pred_label = predicted_class.item()
        conf_score = confidence.item()

        # Store model's prediction and confidence
        model_results[model_name] = (pred_label, conf_score)
        predictions[pred_label].append((model_name, conf_score))

    return predictions, model_results

def find_best_model_for_majority(predictions):
    # Find label with the most votes
    most_voted_label = max(predictions, key=lambda label: len(predictions[label]))
    
    # Find model with highest confidence for this label
    best_model, best_confidence = max(predictions[most_voted_label], key=lambda x: x[1])

    return most_voted_label, best_model, best_confidence

def main():
    image_tensor = preprocess_image(IMAGE_PATH)
    models = load_models(MODEL_FOLDER)
    if not models:
        print("No models found in the folder!")
        return

    predictions, model_results = predict_with_all_models(models, image_tensor)

    majority_label, best_model, best_confidence = find_best_model_for_majority(predictions)

    print(f"Most Voted Label: {majority_label}")
    print(f"Best Model for This Label: {best_model}")
    print(f"Confidence: {best_confidence:.4f}")

if __name__ == "__main__":
    main()
