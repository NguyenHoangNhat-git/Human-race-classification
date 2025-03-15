import torch
import os
import sys

from PIL import Image
from collections import Counter

from model_architecture import ResidualBlock, CustomResnet
from evaluate_models import transform, process_img, class_names, load_models

MODELS_FOLDER = "models/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    try:
        custom_img_path = sys.argv[1]

        models = load_models(MODELS_FOLDER)
        if models:
            print(f"Models loaded successfully")
        torch.set_float32_matmul_precision("high")

        img_tensor = process_img(custom_img_path)

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
                confidences[pred_label] = max(
                    confidences.get(pred_label, 0), confidence
                )

        # Find the label with the most votes
        votes_count = Counter(votes)
        highest_label, highest_count = votes_count.most_common(1)[0]

        picked_label = [
            label for label, count in votes_count.items() if count == highest_count
        ]
        final_pred = max(picked_label, key=lambda label: confidences.get(label, 0))

        print(class_names[final_pred])
    except Exception as e:
        # print(f"Require 1 command line argument")
        print(e)


if __name__ == "__main__":
    main()
