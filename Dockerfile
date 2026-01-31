# Use a Python image with built-in browser support or install it
FROM python:3.12-slim

# Install system dependencies for Chrome and Selenium
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    curl \
    # Dependencies for Chrome
    fonts-liberation \
    libappindicator3-1 \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libc6 \
    libcairo2 \
    libcups2 \
    libdbus-1-3 \
    libexpat1 \
    libfontconfig1 \
    libgbm1 \
    libgcc1 \
    libgconf-2-4 \
    libgdk-pixbuf2.0-0 \
    libglib2.0-0 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libstdc++6 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxrandr2 \
    libxrender1 \
    libxss1 \
    libxtst6 \
    ca-certificates \
    libnss3 \
    lsb-release \
    xdg-utils \
    --no-install-recommends

# Install Google Chrome
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Set display port to avoid crash
ENV DISPLAY=:99

WORKDIR /app

# Copy requirements and install
# (Creating a temporary requirements.txt for the build)
RUN pip install --no-cache-dir \
    selenium \
    webdriver-manager \
    python-dotenv \
    fastmcp \
    mcp \
    uvicorn

# Copy the server code
COPY mcp_scrape.py .

# Expose port for SSE
EXPOSE 8000

# Command to run the server in SSE mode
# We use uvicorn to serve the FastMCP app if we can get the app instance, 
# or use fastmcp CLI
CMD ["fastmcp", "run", "mcp_scrape.py", "--transport", "sse", "--port", "8000", "--host", "0.0.0.0"]
