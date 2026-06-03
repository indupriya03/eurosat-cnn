
# 🛰️ EuroSAT-CNN — Satellite Land-Use Classifier

[![EuroSAT CNN CI/CD](https://github.com/indupriya03/eurosat-cnn/actions/workflows/ci.yml/badge.svg)](https://github.com/indupriya03/eurosat-cnn/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-brightgreen)](https://eurosat-cnn-latest-2.onrender.com/ui)
[![Model on Hugging Face](https://img.shields.io/badge/Model-Hugging%20Face-yellow)](https://huggingface.co/indupriyachidambararaj/eurosat-cnn)
=======

> A production-grade deep learning pipeline that classifies satellite imagery into 10 land-use categories with **92.2% test accuracy**, built entirely from scratch — no pretrained weights.

---

## 🏆 Highlights

| | |
|---|---|
| 🎯 **92.2% Test Accuracy** | CNN trained from scratch on 27,000 satellite images |
| 🖥️ **Web UI** | Dark-themed drag & drop interface with GradCAM visualisation |
| 🔍 **Explainable AI** | GradCAM visualisations show *why* the model predicts each class |
| 🚀 **Production API** | FastAPI REST API with Swagger UI — deployable, not just a notebook |
| 📊 **No Data Leakage** | Mean/std computed on train set only, after stratified split |
| ⚖️ **Stratified Splits** | sklearn stratified 70/15/15 split — class balance preserved across all sets |
| 🧹 **Clean Pipeline** | Single `pipeline.py` entry point — runs everything end to end |
| ⚙️ **Config Driven** | All paths, hyperparameters, and constants centralised in `config.py` |
| 🔁 **CI/CD Pipeline** | GitHub Actions — auto lint (ruff), pytest (12 tests), Docker build & push to GHCR |

---

## 📁 Project Structure

```
eurosat-cnn/
├── config.py                   # single source of truth — paths, hyperparams, seed
├── pipeline.py                 # end to end runner
├── app.py                      # FastAPI REST API + UI server
├── conftest.py                 # pytest: patches load_model with random weights for CI
├── ruff.toml                   # linter config (line-length=100, excludes notebooks/)
├── Dockerfile                  # container image — python:3.10-slim + app
├── .dockerignore               # excludes data/, outputs/, notebooks/ from build context
├── requirements.txt
│
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD — lint + test → Docker build & push to GHCR
│
├── tests/
│   ├── test_api.py             # 7 FastAPI endpoint tests (health, predict, GradCAM)
│   └── test_model.py           # 5 model architecture tests (shape, logits, block3)
│
├── data/
│   └── raw/                    # EuroSAT dataset (10 class folders)
│       ├── AnnualCrop/
│       ├── Forest/
│       └── ...
│
├── outputs/
│   ├── dataset_stats.json      # mean/std computed on train set only
│   ├── best_model.pth          # best checkpoint saved during training
│   ├── training_curves.png     # loss and accuracy plots
│   ├── confusion_matrix.png    # per-class confusion matrix
│   └── gradcam/                # one GradCAM visualisation per class
│
├── notebooks/
│   └── 01_eda.ipynb            # exploratory data analysis
│
├── templates/
│   └── index.html              # web UI frontend
│
└── src/
    ├── dataset.py              # split + stats + transforms + dataloader
    ├── model.py                # EurosatCNN architecture
    ├── train.py                # training loop
    ├── evaluate.py             # metrics + confusion matrix
    ├── gradcam.py              # GradCAM implementation
    └── utils.py                # shared helpers
```

---

## 🧠 Model Architecture

Custom CNN built from scratch — no pretrained weights, no transfer learning.

```
Input  3×64×64
  ↓ Block 1 — Conv→BN→ReLU→Conv→BN→ReLU→MaxPool→Dropout2d(0.1)
 32×32×32
  ↓ Block 2 — Conv→BN→ReLU→Conv→BN→ReLU→MaxPool→Dropout2d(0.2)
 64×16×16
  ↓ Block 3 — Conv→BN→ReLU→Conv→BN→ReLU→AdaptiveAvgPool(4×4)
128×4×4
  ↓ Flatten → Linear(2048→256) → ReLU → Dropout(0.4) → Linear(256→10)
 10 class scores
```

**Design decisions:**
- **BatchNorm** after every conv — training stability, higher learning rate
- **Dropout2d** in conv blocks + **Dropout** in classifier — regularisation at two levels
- **AdaptiveAvgPool** in final block — fixed output size regardless of input resolution
- **Filters double** each block (32→64→128) — learns increasingly abstract features

---

## 📊 Results

```
Overall Test Accuracy: 92.2%

Class                   Precision  Recall   F1
─────────────────────────────────────────────
AnnualCrop              0.950      0.840    0.892
Forest                  0.867      0.998    0.928  ← near perfect recall
HerbaceousVegetation    0.900      0.782    0.837  ← hardest class
Highway                 0.957      0.960    0.959
Industrial              0.986      0.931    0.957
Pasture                 0.925      0.910    0.918
PermanentCrop           0.816      0.899    0.855
Residential             0.894      0.991    0.940
River                   0.954      0.949    0.952
SeaLake                 1.000      0.962    0.981  ← perfect precision
─────────────────────────────────────────────
macro avg               0.925      0.922    0.922
```

> HerbaceousVegetation and AnnualCrop are the hardest classes — even remote sensing experts struggle to distinguish these from RGB imagery alone. Multispectral bands (NIR, SWIR) would improve these.

---

## 🖥️ Web Interface

A production-grade dark-themed dashboard served at `http://localhost:8000/ui`

**Features:**
- Drag & drop satellite image upload
- Instant classification with animated confidence bar
- All 10 class scores visualised as a bar chart
- GradCAM explainability panel — Original | Heatmap | Overlay
- Fully self-contained — no external frontend dependencies

---

## 🔍 Explainable AI — GradCAM

Every prediction comes with a visual explanation showing *which pixels* the model focused on.

```
Original          GradCAM Heatmap       Overlay
┌──────────┐      ┌──────────┐         ┌──────────┐
│  🌾🌾🌾  │      │  🔵🔴🔵  │         │  🌾🔴🌾  │
│  🌾🌾🌾  │  →   │  🔴🔴🔴  │    →    │  🌾🔴🌾  │
│  🌾🌾🌾  │      │  🔵🔴🔵  │         │  🌾🔴🌾  │
└──────────┘      └──────────┘         └──────────┘
                   red = model          confirms model
                   focused here         learned correct
                                        features
```

---

## 🔁 CI/CD Pipeline

Every push to `main` triggers a two-job GitHub Actions workflow:

| Job | What it does |
|-----|-------------|
| **Lint & Test** | Runs `ruff check` on all source files, then `pytest` (12 tests) with a mocked model so no checkpoint is needed in CI |
| **Docker Build & Push** | Builds the Docker image and pushes to GitHub Container Registry (`ghcr.io/indupriya03/eurosat-cnn:latest`) — only runs if Lint & Test passes |

**Key design decisions:**
- Model loading moved into FastAPI `lifespan` startup — import-safe, patchable in tests
- `conftest.py` patches `load_model` with random weights so API tests run without `best_model.pth`
- Docker job only triggers on `main` branch pushes, not PRs

---

## 🌐 Live Demo

Try the deployed app — no setup needed:

| | |
|---|---|
| 🖥️ **Web UI** | https://eurosat-cnn-latest-2.onrender.com/ui |
| 📡 **API Docs** | https://eurosat-cnn-latest-2.onrender.com/docs |
| 🤗 **Model Weights** | https://huggingface.co/indupriyachidambararaj/eurosat-cnn |

> Note: Hosted on Render free tier — may take 30–60 seconds to wake up on first visit.

## 🚀 Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download EuroSAT dataset
Download from [EuroSAT GitHub](https://github.com/phelber/EuroSAT) and place in `data/raw/`

### 3. Run the full pipeline
```bash
# First run — trains from scratch
python pipeline.py

# After training — skip training, reuse checkpoint
python pipeline.py --skip-train
```

### 4. Start the API
```bash
python app.py
```

### 5. Open Web UI
```
http://localhost:8000/ui
```

### 6. Or use Swagger API docs
```
http://localhost:8000/docs
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/ui` | Web UI — drag & drop classifier |
| `GET` | `/classes` | List all 10 land-use classes |
| `POST` | `/predict` | Upload image → JSON prediction |
| `POST` | `/predict/explain/view` | Upload image → GradCAM PNG |

### Example — `/predict`
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@data/raw/Forest/Forest_00001.jpg"
```

```json
{
  "prediction": "Forest",
  "confidence": 0.9731,
  "all_scores": {
    "AnnualCrop": 0.0021,
    "Forest": 0.9731,
    "HerbaceousVegetation": 0.0083,
    "..."  : "..."
  }
}
```

### Example — `/predict/explain/view`
```bash
curl -X POST http://localhost:8000/predict/explain/view \
  -F "file=@data/raw/Industrial/Industrial_00001.jpg" \
  --output gradcam_result.png
```
Returns a PNG with three panels: Original | GradCAM Heatmap | Overlay

---

## ⚙️ Configuration

All settings in `config.py` — change once, reflects everywhere:

```python
SEED         = 42       # reproducibility
BATCH_SIZE   = 32       # dataloader batch size
EPOCHS       = 30       # training epochs
LR           = 1e-3     # learning rate
WEIGHT_DECAY = 1e-4     # L2 regularisation
NUM_CLASSES  = 10       # EuroSAT classes
```

---

## 🔬 Key ML Engineering Decisions

**No data leakage:**
```
raw images → stratified split → compute mean/std on train only
                                         ↓
                              save to dataset_stats.json
                                         ↓
                              load → Normalize all splits
```

**Stratified split over random split:**
```
random_split     → class distribution not guaranteed
stratified split → every split has same class proportions ✓
```

**Incremental mean/std computation:**
```
stack approach      → loads all images into RAM  ✗
incremental E[X²]   → single pass, constant memory
                       scales to any dataset size ✓
```

**Augmentation strategy:**
```
Spatial transforms   → RandomHorizontalFlip, RandomVerticalFlip,
                        RandomRotation(90), RandomResizedCrop
                        satellite has no natural orientation ✓

Pixel transforms     → ColorJitter, GaussianBlur (mild)
                        simulates lighting and atmospheric variation ✓

Avoided              → RandomGrayscale (loses spectral info)
                        RandomErasing (destroys land cover features)
                        AutoAugment (ImageNet policies, not satellite) ✗
```

---

## 🧰 Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch |
| Data Loading | torchvision ImageFolder |
| Splitting | scikit-learn StratifiedSplit |
| Explainability | GradCAM (custom implementation) |
| API | FastAPI + Uvicorn |
| Web UI | Vanilla HTML / CSS / JS |
| Visualisation | Matplotlib, Seaborn |
| Environment | Python 3.10+ |
| CI/CD | GitHub Actions — ruff, pytest, Docker |
| Container | Docker → GHCR (`ghcr.io/indupriya03/eurosat-cnn`) |

---

## 📈 Training Details

```
Optimiser   : Adam (lr=1e-3, weight_decay=1e-4)
Loss        : CrossEntropyLoss
Scheduler   : ReduceLROnPlateau (patience=3, factor=0.5)
Epochs      : 30
Batch size  : 32
Best model  : saved automatically on val accuracy improvement
```

---

## 🗂️ Dataset

[EuroSAT](https://github.com/phelber/EuroSAT) — RGB satellite images from Sentinel-2

| Property | Value |
|----------|-------|
| Total images | ~27,000 |
| Classes | 10 land-use types |
| Image size | 64×64 pixels |
| Source | Sentinel-2 satellite |
| Split | 70% train / 15% val / 15% test |

---

## 👤 Author

**Indupriya Chidambararaj**
- 🔗 [LinkedIn](https://www.linkedin.com/in/indupriyachidambararaj/)
- 🐙 [GitHub](https://github.com/indupriya03)
- 📧 indupriya.chidambararaj@gmail.com