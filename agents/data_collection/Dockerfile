FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHROME_BIN=/usr/bin/chromium \
    CHROMEDRIVER_BIN=/usr/bin/chromedriver \
    SE_CHROMEDRIVER=/usr/bin/chromedriver \
    SE_CHROME_BROWSER_PATH=/usr/bin/chromium

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    chromium \
    chromium-driver \
    fonts-liberation \
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
    libgcc-s1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libstdc++6 \
    libvulkan1 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    xdg-utils \
    xvfb \
    && ln -s /usr/bin/chromium /usr/bin/google-chrome \
    && ln -s /usr/bin/chromium /usr/bin/chromium-browser \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install \
        aiohttp \
        beautifulsoup4 \
        black \
        bs4 \
        click \
        colorama \
        ddgs \
        fake-useragent \
        fastapi \
        html5lib \
        ipykernel \
        ipython \
        jupyter_client \
        jupyter_kernel_gateway \
        loguru \
        lxml \
        markdownify \
        mypy \
        numpy \
        openpyxl \
        orjson \
        pandas \
        parsel \
        pdfplumber \
        pillow \
        playwright \
        pydantic \
        pypdf \
        pytest \
        pytest-asyncio \
        python-dateutil \
        python-dotenv \
        pytz \
        PyYAML \
        rapidfuzz \
        readability-lxml \
        regex \
        requests \
        rich \
        ruff \
        scrapy \
        selenium \
        selenium-stealth \
        selectolax \
        tenacity \
        trafilatura \
        typer \
        uvicorn \
        httpx \
        xlsxwriter

RUN playwright install chromium

EXPOSE 8888

CMD ["jupyter", "kernelgateway", "--KernelGatewayApp.ip=0.0.0.0", "--KernelGatewayApp.port=8888", "--KernelGatewayApp.allow_origin=*"]
