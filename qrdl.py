import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image


MODEL_PATH = "best_qr_model.pth"
THRESHOLD = 0.5845
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])


def load_model():

    model = models.resnet18(weights=None)

    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features,2)
    )

    model.load_state_dict(torch.load(MODEL_PATH,map_location=DEVICE))

    model.to(DEVICE)

    model.eval()

    return model


def predict_image(model,image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError("Image not found")

    image = Image.open(image_path).convert("RGB")

    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image)

        probs = F.softmax(outputs,dim=1)

        distortion_prob = probs[0][1].item()

    if distortion_prob >= THRESHOLD:
        label = "DISTORTED"
    else:
        label = "CLEAN"

    return label, distortion_prob