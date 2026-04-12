# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Agentic Browser — API server container
#
# Includes a full Playwright/Chromium installation so the API server can
# control a headless browser inside the container.
#
# Build:
#   docker build -t agenticbrowser .
#
# Run:
#   docker run -p 8000:8000 agenticbrowser
# ---------------------------------------------------------------------------

FROM python:3.12-slim

# System packages required by Playwright/Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
        # Chromium runtime dependencies
        libnss3 \
        libnspr4 \
        libatk1.0-0 \
        libatk-bridge2.0-0 \
        libcups2 \
        libdrm2 \
        libdbus-1-3 \
        libxkbcommon0 \
        libxcomposite1 \
        libxdamage1 \
        libxfixes3 \
        libxrandr2 \
        libgbm1 \
        libasound2 \
        libpango-1.0-0 \
        libpangocairo-1.0-0 \
        libcairo2 \
        libatspi2.0-0 \
        libwayland-client0 \
        # Cleanup
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cached as long as requirements.txt
# and the Playwright version don't change).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && playwright install chromium \
    && playwright install-deps chromium

# Copy application source
COPY api_server.py \
     browser_agent.py \
     task_planner.py \
     system_tools.py \
     ./

# Workspace directory for file-system actions
RUN mkdir -p /app/workspace
ENV BROWSER_WORKSPACE=/app/workspace

# Expose the API port
EXPOSE 8000

# Run the API server (headless=True is the default)
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
