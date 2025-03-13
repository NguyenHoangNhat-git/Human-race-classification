import torch
import os

import torchvision.transforms as transforms
from torch import nn

from PIL import Image
from collections import Counter
from tqdm.auto import tqdm

from model_architecture import ResidualBlock, CustomResnet

# Configuration
MODELS_FOLDER = "models/"
TEST_FOLDER = "test imgs/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

transform = transforms.Compose(
    [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
)

class_names = [
    "Black",
    "East Asian",
    "Indian",
    "Latino_Hispanic",
    "Middle Eastern",
    "Southeast Asian",
    "White",
]


def process_img(img_path):
    img = Image.open(img_path)
    return transform(img).unsqueeze(dim=0).to(device)


def load_models(model_folder):
    models = {}
    for filename in os.listdir(model_folder):
        if filename.endswith(".pt") or filename.endswith(".pth"):
            model_path = os.path.join(model_folder, filename)
            model = torch.load(f=model_path, weights_only=False)
            model.eval()
            models[filename] = model
    return models


def load_test_imgs(folder):
    test_imgs = []
    test_labels = []

    for class_name in os.listdir(folder):
        class_folder = os.path.join(folder, class_name)
        for file in os.listdir(class_folder):
            if file.endswith((".jpg", ".png", ".jpeg")):
                test_imgs.append(os.path.join(class_folder, file))
                test_labels.append(class_names.index(class_name))
    return test_imgs, test_labels


def evaluate_ensemble(models: dict, test_imgs, test_labels):
    correct_preds = 0
    total_samples = len(test_imgs)
    for img_path, true_label in tqdm(zip(test_imgs, test_labels)):
        img_tensor = process_img(img_path=img_path)

        votes = []
        confidences = {}

        for model_name, model in models.items():
            with torch.inference_mode():
                logits = model(img_tensor)
                probs = torch.softmax(logits, dim=1)
                pred_label = probs.argmax(dim=1)
                votes.append(pred_label.item())

                if pred_label not in confidences:
                    confidences[pred_label] = 0
                confidence = probs[0, pred_label].item()
                confidences[pred_label] = max(confidences.get(pred_label, 0), confidence)
                
        # Find the label with the most votes
        votes_count = Counter(votes)
        highest_label, highest_count = votes_count.most_common(1)[0]

        picked_label = [
            label for label, count in votes_count.items() if count == highest_count
        ]
        final_pred = max(picked_label, key=lambda label: confidences.get(label, 0))

        if final_pred == true_label:
            correct_preds += 1

    acc = (correct_preds / total_samples) * 100
    print(f"Accuracy from {total_samples} samples : {acc: .5f}")


def main():
    models = load_models(MODELS_FOLDER)
    if not models:
        print("No models found")
        return

    test_imgs, test_labels = load_test_imgs(TEST_FOLDER)
    if not test_imgs:
        print("No test imgs found")
        return

    evaluate_ensemble(models=models, test_imgs=test_imgs, test_labels=test_labels)


if __name__ == "__main__":
    main()
