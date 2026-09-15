# Reinforcement Studio backend image — CPU by default; pass --gpus all at run time if
# the host has the NVIDIA Container Toolkit and you want AlphaZero/PPO on GPU.
FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /uvx /bin/

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY rl_core/requirements.txt /app/rl_core/requirements.txt
COPY backend/requirements.txt /app/backend/requirements.txt

RUN uv pip install --system --no-cache --torch-backend cpu "torch>=2.2" \
    && uv pip install --system --no-cache \
      -r /app/rl_core/requirements.txt \
      -r /app/backend/requirements.txt

COPY rl_core /app/rl_core
COPY backend /app/backend

ENV PYTHONUNBUFFERED=1 \
    RL_STUDIO_ROOT=/app/rl_core

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/system/health', timeout=2)" || exit 1

CMD ["python", "-m", "uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
