import base64
import logging
import os
from pathlib import Path
from typing import cast

import anthropic
import pymupdf  # type: ignore
from pydantic import BaseModel

from .config import DEBUG_DIR
from .llm import Usage
from .readers import pdf_to_markdown, png_from_page

logger = logging.getLogger(__name__)


class Chapter(BaseModel):
    title: str
    content: str
    first_page: int
    last_page: int


def chunk_into_chapters(pdf: str | Path | pymupdf.Document) -> list[Chapter]:
    """Split text into chapters based on `#` Markdown headers."""
    if isinstance(pdf, Path):
        pdf = pymupdf.open(pdf)

    pages = pdf_to_markdown(pdf, page_chunks=True)
    chapters: list[Chapter] = []
    current_chapter: Chapter | None = None

    for i, page_text in enumerate(pages):
        for line in page_text.split("\n"):
            if line.startswith("# "):
                if current_chapter:
                    current_chapter.content = current_chapter.content.strip()
                    current_chapter.last_page = i - 1

                    chapters.append(current_chapter)
                current_chapter = Chapter(
                    title=line[2:], content="", first_page=i, last_page=i
                )
            elif current_chapter:
                current_chapter.content += line + "\n"
                current_chapter.last_page = i - 1
            else:
                logger.warning(f"⚠️ Chapter-less content: {line}")

    if current_chapter:
        chapters.append(current_chapter)

    if chapters:
        return chapters

    logger.warning(
        "⚠️ No chapters found in document:\n" + "\n".join(pages)[:500] + "..."
    )
    return [
        Chapter(
            title="",
            content="\n".join(pages),
            first_page=0,
            last_page=len(pages) - 1,
        )
    ]


def chunk_into_chapters_from_toc(
    pdf: str | Path | pymupdf.Document,
) -> tuple[list[Chapter], Usage]:
    """Split PDF into chapters based on table of contents using Claude Vision."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    if not isinstance(pdf, pymupdf.Document):
        pdf = pymupdf.open(pdf)

    # First, find the table of contents page
    page_pngs: list[bytes] = []
    for i in range(min(5, len(pdf))):  # Check first 10 pages
        # Render the TOC page as an image
        page = pdf[i]
        img_bytes = png_from_page(page)

        # Save the TOC image for debugging purposes
        debug_image_path = DEBUG_DIR / f"page_{i + 1}.png"
        debug_image_path.write_bytes(img_bytes)
        logger.debug(f"🔍 Saved page screenshot to {debug_image_path.as_posix()}")

        page_pngs.append(img_bytes)

    # Use Claude Vision to analyze the TOC
    system_prompt = """You are helping to extract the table of contents from a thesis. 
    Identify chapter titles and their corresponding page numbers.
    Return the information in a structured JSON format like:
    [
        {"title": "Introduction", "page": 1},
        {"title": "Literature Review", "page": 10},
        {"title": "Appendix", "page": 24, "exclude": true},
        ...
    ]
    Only include main chapters, not subsections. Page numbers should be integers.
    Set `exclude` for non-content chapters like appendices.

    If there is no ToC, just return an empty array `[]`."""

    response = client.messages.create(
        model="claude-3-7-sonnet-latest",
        max_tokens=1000,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here are the first 10 pages of the document:",
                    },
                    *[
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(png).decode("utf-8"),
                            },
                        }
                        for png in page_pngs
                    ],
                    {
                        "type": "text",
                        "text": "Extract the chapter titles and page numbers from the table of contents.",
                    },
                ],
            }
        ],
    )

    # Parse the response to get chapter information
    import json
    import re

    response_text = response.content[0].text
    usage = Usage(**response.usage.model_dump())
    # Extract JSON from the response if it's wrapped in text
    json_match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
    if json_match:
        chapters_info = json.loads(json_match.group(1))
    else:
        try:
            chapters_info = json.loads(response_text)
        except json.JSONDecodeError:
            logger.error(f"🚨 Failed to parse TOC response: {response_text}")
            return chunk_into_chapters(pdf), usage

    # Extract chapters based on page numbers
    chapters: list[Chapter] = []
    for i in range(len(chapters_info)):
        current = chapters_info[i]
        if current.get("exclude"):
            continue
        title: str = current["title"]
        start_page: int = current["page"] - 1  # Convert to 0-indexed

        # Determine end page
        if i < len(chapters_info) - 1:
            end_page = chapters_info[i + 1]["page"] - 1
        else:
            end_page = len(pdf)

        # Extract text from the chapter pages
        chapter_text = ""
        for page_num in range(start_page, end_page):
            if 0 <= page_num < len(pdf):
                chapter_text += cast(str, pdf[page_num].get_text())  # type: ignore

        chapters.append(
            Chapter(
                title=title,
                content=chapter_text,
                first_page=start_page,
                last_page=end_page - 1,
            )
        )

    if not chapters:
        logger.warning(
            "🚨 No chapters extracted from TOC. Falling back to simple chunking."
        )
        return chunk_into_chapters(pdf), usage

    return chapters, usage
