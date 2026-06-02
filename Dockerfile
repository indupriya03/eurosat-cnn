# ── Base image ────────────────────────────────────────────────
# Python 3.10 on slim Linux — small and clean
FROM python:3.10-slim

# ── Working directory inside container ────────────────────────
# Creates /app folder inside the container
WORKDIR /app

# ── Install dependencies ──────────────────────────────────────
# Copy requirements first — Docker caches this layer
# So if requirements.txt doesn't change, pip install is skipped
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 300 -r requirements.txt

# ── Copy your project files ───────────────────────────────────
COPY src/ ./src/
COPY templates/ ./templates/
COPY app.py config.py ./
COPY outputs/ ./outputs/

# ── Create outputs folder ─────────────────────────────────────
# Your model weights go here at runtime (not committed to git)
RUN mkdir -p outputs

# ── Expose port ───────────────────────────────────────────────
# FastAPI runs on 8000
EXPOSE 8000

# ── Start the app ─────────────────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]