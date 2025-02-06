FROM mcr.microsoft.com/playwright/python:v1.40.0-jammy

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    redis-tools \
    chromium-browser \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright with specific version and verify
RUN playwright install --with-deps chromium && \
    playwright install chromium && \
    python3 -c "from playwright.sync_api import sync_playwright; playwright = sync_playwright().start(); browser = playwright.chromium.launch(); browser.close(); playwright.stop()"

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p /tmp/calendar-data && chmod 777 /tmp/calendar-data
RUN mkdir -p /tmp/screenshots && chmod 777 /tmp/screenshots
RUN mkdir -p /tmp/errors && chmod 777 /tmp/errors

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
ENV DEBUG=pw:api

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"] 