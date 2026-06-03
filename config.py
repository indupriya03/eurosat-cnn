# config.py
import os
import torch
import urllib.request

# project root
ROOT_DIR     = os.path.dirname(os.path.abspath(__file__))

# key paths
DATA_DIR     = os.path.join(ROOT_DIR, "data", "raw")
OUTPUTS_DIR  = os.path.join(ROOT_DIR, "outputs")
STATS_PATH   = os.path.join(OUTPUTS_DIR, "dataset_stats.json")
MODEL_PATH   = os.path.join(OUTPUTS_DIR, "best_model.pth")
GRADCAM_DIR  = os.path.join(OUTPUTS_DIR, "gradcam")

HF_MODEL_URL = "https://huggingface.co/indupriya03/eurosat-cnn/resolve/main/best_model.pth"

def ensure_model_downloaded():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        print("Downloading model from Hugging Face...")
        urllib.request.urlretrieve(HF_MODEL_URL, MODEL_PATH)
        print("Model downloaded.")

# model config
NUM_CLASSES  = 10
BATCH_SIZE   = 32
EPOCHS       = 30
LR           = 1e-3
WEIGHT_DECAY = 1e-4
SEED         = 42        

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]