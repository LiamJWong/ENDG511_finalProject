import os
import torch
from PIL import Image
from torchvision import transforms

from model import ProtoNet

# =========================
# CONFIG
# =========================
CKPT_PATH = "MobileNet_LessDataAugmented_LrPlateau.ckpt"
SUPPORT_ROOT = "support"
TEST_IMAGE = "test.jpg"
IMG_SIZE = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def load_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img)


def build_support_set():
    support_tensors = []
    support_labels = []
    class_names = []

    for label_idx, class_name in enumerate(sorted(os.listdir(SUPPORT_ROOT))):
        class_dir = os.path.join(SUPPORT_ROOT, class_name)

        if not os.path.isdir(class_dir):
            continue

        class_names.append(class_name)

        for fname in sorted(os.listdir(class_dir)):
            fpath = os.path.join(class_dir, fname)

            try:
                x = load_image(fpath)
                support_tensors.append(x)
                support_labels.append(label_idx)
            except Exception as e:
                print(f"Skipping {fpath}: {e}")

    if not support_tensors:
        raise RuntimeError("No support images found in support/")

    support_x = torch.stack(support_tensors).to(DEVICE)
    support_y = torch.tensor(support_labels, dtype=torch.long).to(DEVICE)

    print(f"Loaded {len(support_tensors)} support images across {len(class_names)} classes.")
    return support_x, support_y, class_names


def load_model():
    print("Loading model...")
    model = ProtoNet(proto_dim=64).to(DEVICE)

    state_dict = torch.load(CKPT_PATH, map_location=DEVICE)

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception:
        print("Direct load failed. Trying cleaned checkpoint keys...")
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace("model.", "")
            cleaned_state_dict[new_k] = v
        model.load_state_dict(cleaned_state_dict, strict=False)

    model.eval()
    print("Model loaded successfully.")
    return model


def main():
    model = load_model()
    support_x, support_y, class_names = build_support_set()

    if not os.path.exists(TEST_IMAGE):
        raise FileNotFoundError(f"Could not find test image: {TEST_IMAGE}")

    query_x = load_image(TEST_IMAGE).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        preds, classes = model.predict(support_x, support_y, query_x)

        pred_idx = torch.argmax(preds, dim=1).item()
        pred_class_id = classes[pred_idx].item()
        pred_name = class_names[pred_class_id]

    print("Prediction successful.")
    print(f"Predicted class: {pred_name}")


if __name__ == "__main__":
    main()