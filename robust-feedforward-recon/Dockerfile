FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
RUN pip install --no-cache-dir -e ".[dev]"
CMD ["bash", "scripts/run_all.sh"]
