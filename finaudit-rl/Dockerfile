FROM python:3.11-slim

WORKDIR /app

# The project is dependency-free at runtime.  Installing it in editable mode
# still requires no network access beyond the base image.
COPY pyproject.toml README.md LICENSE scripts/ scripts/
COPY finaudit ./finaudit
COPY tests ./tests
COPY docs ./docs
COPY episodes ./episodes
COPY trajectories ./trajectories

RUN python3 -m pip install --no-cache-dir -e .

CMD ["sh", "scripts/check.sh"]
