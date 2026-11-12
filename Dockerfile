# Web service for the premium-classification model, served with Waitress.
#
# Build:  docker build -t cattle-price .
# Run:    docker run -it --rm -p 9696:9696 cattle-price
#
# models/model.bin must exist before building: run `uv run python src/train.py` first.

FROM python:3.14-slim

# uv, copied from its official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 1. Dependencies first: this layer is cached and only rebuilt when they change.
#    --locked: use exactly the versions in uv.lock. --no-dev: skip Jupyter, plotting, XGBoost...
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project

# 2. The service and the trained model
COPY src/predict.py ./
COPY models/model.bin ./models/model.bin
ENV MODEL_PATH=/app/models/model.bin

# Use the virtual environment created by uv
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "predict:app"]
