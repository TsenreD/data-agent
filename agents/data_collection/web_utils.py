from __future__ import annotations

from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


def fetch_html(url: str, timeout: int = 30, headers: dict[str, str] | None = None) -> dict[str, Any]:
    merged_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        )
    }
    if headers:
        merged_headers.update(headers)

    response = requests.get(url, timeout=timeout, headers=merged_headers)
    response.raise_for_status()
    return {
        "method": "requests",
        "url": response.url,
        "status_code": int(response.status_code),
        "html": response.text,
    }


def fetch_rendered_html_selenium(url: str, timeout: int = 30, wait_seconds: float = 3.0) -> dict[str, Any]:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1440,2000")

    driver = webdriver.Chrome(service=Service(), options=options)
    driver.set_page_load_timeout(timeout)
    try:
        driver.get(url)
        if wait_seconds > 0:
            driver.implicitly_wait(wait_seconds)
        return {
            "method": "selenium",
            "url": driver.current_url,
            "status_code": 200,
            "html": driver.page_source,
        }
    finally:
        driver.quit()


def extract_main_text(html: str, url: str | None = None) -> dict[str, Any]:
    trafilatura_result = _extract_with_trafilatura(html, url=url)
    if trafilatura_result is not None:
        return trafilatura_result

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
    return {"method": "bs4", "text": text}


def extract_links(
    html: str,
    base_url: str,
    *,
    same_domain_only: bool = False,
    selector: str | None = None,
) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    anchors = soup.select(selector) if selector else soup.find_all("a")
    base_domain = urlparse(base_url).netloc
    links: list[dict[str, str]] = []
    seen: set[str] = set()

    for anchor in anchors:
        href = (anchor.get("href") or "").strip()
        if not href:
            continue
        absolute_url = urljoin(base_url, href)
        if same_domain_only and urlparse(absolute_url).netloc not in {"", base_domain}:
            continue
        if absolute_url in seen:
            continue
        seen.add(absolute_url)
        links.append(
            {
                "url": absolute_url,
                "text": " ".join(anchor.get_text(" ", strip=True).split()),
            }
        )
    return links


def fetch_and_extract(url: str, timeout: int = 30, use_selenium: bool = False) -> dict[str, Any]:
    fetched = fetch_rendered_html_selenium(url, timeout=timeout) if use_selenium else fetch_html(url, timeout=timeout)
    extracted = extract_main_text(fetched["html"], url=fetched["url"])
    return {
        "fetch_method": fetched["method"],
        "extract_method": extracted["method"],
        "url": fetched["url"],
        "html": fetched["html"],
        "text": extracted["text"],
        "status_code": fetched["status_code"],
        "links": extract_links(fetched["html"], fetched["url"]),
    }


def _extract_with_trafilatura(html: str, url: str | None = None) -> dict[str, Any] | None:
    try:
        import trafilatura
    except Exception:
        return None

    try:
        text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            include_formatting=False,
        )
    except Exception:
        return None

    if not text:
        return None
    cleaned = text.strip()
    if len(cleaned) < 200:
        return None
    return {"method": "trafilatura", "text": cleaned}
