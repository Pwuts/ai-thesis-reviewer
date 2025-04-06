import logging
import os
import traceback
from typing import Iterator, Optional

import anthropic
import gradio as gr

from .chunkers import chunk_into_chapters_from_toc
from .config import LOG_FORMAT, LOG_LEVEL
from .llm import Usage
from .readers import pdf_to_markdown

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
FENCE = "=" * 64


def review_thesis_oneshot(
    main_pdf: str, rubric_pdf: Optional[str] = None, text_only: bool = False
) -> Iterator[tuple[str, str]]:
    """Process PDFs and generate comprehensive feedback."""
    logger.info("⏳ Loading submitted document...")
    pages = pdf_to_markdown(main_pdf, page_chunks=True)
    content = "\n".join(
        f"<-------[page {i}]------->\n{content}" for i, content in enumerate(pages, 1)
    )

    # Extract text from rubric if provided
    rubric_content = None
    if rubric_pdf:
        logger.info("⏳ Loading rubric...")
        rubric_content = pdf_to_markdown(rubric_pdf)

    logger.info("⏳ Generating feedback...")
    try:
        usage = None
        for feedback, usage in generate_feedback(content, rubric_content):
            yield feedback, format_usage(usage)
        assert usage
        logger.info(f"💲 Review usage: ${usage.cost:.3f} - {usage}")
    except Exception as e:
        yield format_error(e), ""
        raise


def review_thesis_per_chapter(
    main_pdf: str, rubric_pdf: Optional[str] = None, text_only: bool = False
) -> Iterator[tuple[str, str]]:
    """Process PDFs and generate comprehensive feedback."""
    # Extract text from rubric if provided
    rubric_content = None
    if rubric_pdf:
        rubric_content = pdf_to_markdown(rubric_pdf)

    # Split into chapters based on table of contents
    chapters, chunking_usage = chunk_into_chapters_from_toc(main_pdf)
    logger.info(
        "📖 Chapters:\n"
        + "\n".join(f"{i}. {c.title}" for i, c in enumerate(chapters, 1))
    )
    logger.info(f"💲 Chunking cost: ${chunking_usage.cost:.3f} from {chunking_usage}")

    # Generate feedback for each chapter
    feedback: list[str] = []
    total_review_usage = None
    for i, chapter in enumerate(chapters, 1):
        logger.info(
            f"⏳ Generating feedback for chapter {i} ({chapter.title}):\n"
            f"{FENCE}\n{chapter.content[:500]}...\n{FENCE}"
        )
        chapter_feedback = ""
        chapter_usage = None
        try:
            for _wip_feedback, _usage in generate_feedback(
                chapter.content, rubric_content
            ):
                chapter_usage = _usage
                _total_usage = (
                    total_review_usage + chapter_usage
                    if total_review_usage
                    else chapter_usage
                ) + chunking_usage
                # Show intermediate output
                yield (
                    "\n".join(feedback)
                    + f"\nChapter {i} - {chapter.title}:\n{_wip_feedback}\n",
                    format_usage(_total_usage),
                )
                chapter_feedback = _wip_feedback
        except Exception as e:
            yield format_error(e), ""
            raise

        assert chapter_usage
        logger.info(
            f"💲 Review cost for chapter {i}: ${chapter_usage.cost:.3f} "
            f"from {chapter_usage}"
        )
        total_review_usage = (total_review_usage or Usage.null()) + chapter_usage

        feedback.append(f"Chapter {i} - {chapter.title}:\n{chapter_feedback}\n")

    assert total_review_usage
    logger.info("✅ Finished review")
    total_usage = chunking_usage + total_review_usage
    logger.info(f"💲💲💲 Total review usage: ${total_usage.cost:.3f} - {total_usage}")


# Create Gradio interface
iface = gr.Interface(
    fn=review_thesis_oneshot,
    # fn=review_thesis_per_chapter,
    inputs=[
        gr.File(label="Upload Thesis Draft (PDF)", file_types=[".pdf"]),
        gr.File(label="Upload Rubric (PDF, Optional)", file_types=[".pdf"]),
        gr.Checkbox(label="Text only (cheaper)", value=False),
    ],
    outputs=[
        gr.Markdown(label="Feedback", height="50vh"),
        gr.Markdown(),
    ],
    title="Academic Writing Feedback Generator",
    description="Upload your thesis draft to receive detailed feedback. Optionally include a rubric for more specific evaluation.",
)


def generate_feedback(
    content: str, rubric_content: Optional[str] = None
) -> Iterator[tuple[str, Usage]]:
    """Generate feedback using Claude."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    system_prompt = """You are an experienced academic writing advisor. Analyze the text and provide clear, constructive feedback based on Purdue OWL writing guidelines. Your feedback should:
    1. Be specific and actionable
    2. Include praise only when genuinely deserved
    3. Offer encouragement coupled with constructive suggestions when appropriate
    4. Focus on both structural and content-related aspects
    5. Consider academic writing conventions"""

    if rubric_content:
        system_prompt += (
            "\nAlso evaluate the content against the provided rubric criteria."
        )

    message = f"""Here is the academic text to review:

{content}

"""
    if rubric_content:
        message += f"\nRubric criteria:\n{rubric_content}"

    usage = None
    with client.messages.stream(
        model="claude-3-7-sonnet-latest",
        max_tokens=4000,
        system=system_prompt,
        messages=[{"role": "user", "content": message}],
    ) as stream:
        text_so_far = ""
        for event in stream:
            if event.type == "text":
                text_so_far += event.text
            else:
                logger.debug(f"🤔 Unhandled stream event '{event.type}': {event}")
            usage = Usage(**stream.current_message_snapshot.usage.model_dump())
            yield text_so_far, usage


def format_error(exc: BaseException) -> str:
    return f"""## ⚠️ Something went wrong

    The review could not be completed due to an error (`{exc.__class__.__name__}`):
    ```
    {exc.body if isinstance(exc, anthropic.APIError) else traceback.format_exception(exc)}
    ```
    """


def format_usage(usage: Usage) -> str:
    """Formats a usage object into a Markdown section"""
    return f"### Inference cost: 💲{usage.cost:.2f}"


def main():
    iface.launch()
