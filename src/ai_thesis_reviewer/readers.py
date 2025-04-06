from pathlib import Path
from typing import Literal, Optional, overload

import pymupdf  # type: ignore
import pymupdf4llm  # type: ignore


@overload
def pdf_to_markdown(
    pdf: str | Path | pymupdf.Document,
    page_chunks: Literal[True],
    pages: Optional[list[int]] = None,
) -> list[str]: ...


@overload
def pdf_to_markdown(
    pdf: str | Path | pymupdf.Document,
    page_chunks: Literal[False] = False,
    pages: Optional[list[int]] = None,
) -> str: ...


def pdf_to_markdown(
    pdf: str | Path | pymupdf.Document,
    page_chunks: bool = False,
    pages: Optional[list[int]] = None,
) -> str | list[str]:
    """Extract text content from a PDF file."""
    return pymupdf4llm.to_markdown(pdf, pages=pages, page_chunks=page_chunks)  # type: ignore


def png_from_page(page: pymupdf.Page, zoom_factor: float = 1.5) -> bytes:
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom_factor, zoom_factor))  # type: ignore
    return pixmap.tobytes("png")  # type: ignore
