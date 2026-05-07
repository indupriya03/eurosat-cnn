import io
import base64
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from torchvision import transforms
from src.model import EurosatCNN
from src.gradcam import GradCAM
from src.utils import load_model
from src.dataset import load_stats
from config import CLASSES, DEVICE, MODEL_PATH

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse



# ── Setup ────────────────────────────────────────────────
app = FastAPI(
    title="EuroSAT Satellite Classifier",
    description="Upload a satellite image → get land-use class + GradCAM explanation",
    version="2.0.0"
)

# after app = FastAPI(...)


@app.get("/ui", response_class=FileResponse)
def frontend():
    return "templates/index.html"

device    = torch.device(DEVICE)
MEAN, STD = load_stats()

# ── Load model once at startup ───────────────────────────
model   = EurosatCNN(num_classes=len(CLASSES)).to(device)
model   = load_model(model, MODEL_PATH, device)
gradcam = GradCAM(model, target_layer=model.block3[3])

# ── Preprocessing ────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ── Helper: fig → base64 ─────────────────────────────────
def fig_to_base64(fig) -> str:
    buf     = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded

# ── Helper: run inference ────────────────────────────────
def run_inference(image: Image.Image):
    tensor = preprocess(image).unsqueeze(0).to(device)  # outside no_grad → grad tracking ✓
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]
    pred_idx   = probs.argmax().item()
    confidence = probs[pred_idx].item()
    all_scores = {cls: round(probs[i].item(), 4) for i, cls in enumerate(CLASSES)}
    return pred_idx, confidence, all_scores, tensor      # tensor reused for GradCAM

# ── Routes ───────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "EuroSAT Classifier API is running"}

@app.get("/classes")
def get_classes():
    return {"classes": CLASSES}

# ── /predict ─────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid image file"})

    pred_idx, confidence, all_scores, _ = run_inference(image)
    return {
        "prediction": CLASSES[pred_idx],
        "confidence": round(confidence, 4),
        "all_scores": all_scores
    }


from fastapi.responses import StreamingResponse
import io

@app.post("/predict/explain/view")
async def predict_explain_view(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid image file"})

    pred_idx, confidence, all_scores, tensor = run_inference(image)

    # GradCAM
    cam, _ = gradcam.generate(tensor, class_idx=pred_idx)

    # build overlay
    img_array = np.array(image.resize((64, 64))) / 255.0
    cam_image = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((64, 64), Image.BILINEAR)
    ) / 255.0
    heatmap = plt.cm.jet(cam_image)[:, :, :3]
    overlay = 0.5 * img_array + 0.5 * heatmap

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(img_array);             axes[0].set_title("Original",                   fontsize=8); axes[0].axis("off")
    axes[1].imshow(cam_image, cmap="jet"); axes[1].set_title("GradCAM",                    fontsize=8); axes[1].axis("off")
    axes[2].imshow(overlay);               axes[2].set_title(f"Pred: {CLASSES[pred_idx]}", fontsize=8); axes[2].axis("off")
    plt.tight_layout()

    # return as PNG directly
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return StreamingResponse(buf, media_type="image/png")

# ── Run ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)