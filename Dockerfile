# Web service for the premium-classification model, served with Waitress.
#
# Build:  docker build -t cattle-price .
# Run:    docker run -it --rm -p 9696:9696 cattle-price
#
# models/model.bin and models/form_options.json must exist before building:
# run `uv run python src/train.py` first.

FROM python:3.14-slim

# uv, copied from its official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 1. Dependencies first: this layer is cached and only rebuilt when they change.
#    --locked: use exactly the versions in uv.lock. --no-dev: skip Jupyter, plotting, XGBoost...
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project

# 2. The service (code, HTML templates, CSS) and the trained model with the form options
COPY src/predict.py ./
COPY src/templates ./templates
COPY src/static ./static
COPY models/model.bin models/form_options.json ./models/
ENV MODEL_PATH=/app/models/model.bin \
    OPTIONS_PATH=/app/models/form_options.json

# Use the virtual environment created by uv
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "predict:app"]
