from __future__ import annotations

import io
from typing import Any

import fitz
import pdfplumber
import requests
from pdfminer.high_level import extract_text as pdfminer_extract_text


def extract_pdf_text_from_url(url: str, timeout: int = 30) -> dict[str, Any]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return extract_pdf_text_from_bytes(response.content)


def extract_pdf_text_from_bytes(pdf_bytes: bytes) -> dict[str, Any]:
    pymupdf_result = _extract_with_pymupdf(pdf_bytes)
    if pymupdf_result is not None:
        return pymupdf_result

    pdfplumber_result = _extract_with_pdfplumber(pdf_bytes)
    if pdfplumber_result is not None:
        return pdfplumber_result

    pdfminer_result = _extract_with_pdfminer(pdf_bytes)
    if pdfminer_result is not None:
        return pdfminer_result

    return {"method": "unparsed", "text": "", "page_texts": []}


def _extract_with_pymupdf(pdf_bytes: bytes) -> dict[str, Any] | None:
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_texts = [(page.get_text("text") or "").strip() for page in document]
        text = "\n\n".join(chunk for chunk in page_texts if chunk)
        if len(text) >= 200:
            return {"method": "pymupdf", "text": text, "page_texts": page_texts}
    except Exception:
        return None
    return None


def _extract_with_pdfplumber(pdf_bytes: bytes) -> dict[str, Any] | None:
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            page_texts = [(page.extract_text() or "").strip() for page in pdf.pages]
        text = "\n\n".join(chunk for chunk in page_texts if chunk)
        if len(text) >= 200:
            return {"method": "pdfplumber", "text": text, "page_texts": page_texts}
    except Exception:
        return None
    return None


def _extract_with_pdfminer(pdf_bytes: bytes) -> dict[str, Any] | None:
    try:
        text = (pdfminer_extract_text(io.BytesIO(pdf_bytes)) or "").strip()
        if len(text) >= 200:
            return {"method": "pdfminer", "text": text, "page_texts": [text]}
    except Exception:
        return None
    return None
