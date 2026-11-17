import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from model import build_model

CLASS_NAMES = ["NORMAL", "PNEUMONIA", "COVID-19"]

def generate_gradcam(image_path, checkpoint="models/best_resnet18.pt", class_idx=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(3).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    image = Image.open(image_path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    tensor = tfm(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
    predicted = int(probs.argmax(1).item())
    target_idx = predicted if class_idx is None else class_idx

    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    grayscale_cam = cam(
        input_tensor=tensor,
        targets=[ClassifierOutputTarget(target_idx)]
    )[0]

    rgb = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
    visualization = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
    return visualization, CLASS_NAMES[predicted], probs[0].cpu().numpy()
