import os
import time
import cv2
import torch
from PIL import Image
from torchvision import transforms

from model import ProtoNet

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "MobileNet_LessDataAugmented_LrPlateau.ckpt"
SUPPORT_ROOT = "support"
IMG_SIZE = 128
INFERENCE_INTERVAL = 1.0  # seconds

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


def build_support_set(support_root):
    support_tensors = []
    support_labels = []
    class_names = []

    for label_idx, class_name in enumerate(sorted(os.listdir(support_root))):
        class_dir = os.path.join(support_root, class_name)

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


def preprocess_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = transform(pil).unsqueeze(0).to(DEVICE)
    return x


def main():
    print("Using device:", DEVICE)

    model = load_model()
    support_x, support_y, class_names = build_support_set(SUPPORT_ROOT)

    # Precompute support embeddings once for speed
    with torch.no_grad():
        support_feats = model.encode(support_x)
        prototypes, classes = model.calculate_prototypes(support_feats, support_y)

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    last_inference_time = 0
    pred_name = "Waiting..."

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break

            current_time = time.time()

            if current_time - last_inference_time >= INFERENCE_INTERVAL:
                query_x = preprocess_frame(frame)
                query_feats = model.encode(query_x)

                preds, classes = model.classify_feats(prototypes, classes, query_feats)

                pred_idx = torch.argmax(preds, dim=1).item()
                pred_class_id = classes[pred_idx].item()
                pred_name = class_names[pred_class_id]

                last_inference_time = current_time
                print(f"Prediction: {pred_name}")

            cv2.putText(
                frame,
                f"Pred: {pred_name}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "ESC to quit",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.imshow("Few-Shot MobileNet ProtoNet", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()