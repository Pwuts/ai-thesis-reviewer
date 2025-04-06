import logging
import os
import traceback
from typing import Iterator, Optional

import anthropic
import gradio as gr
import pymupdf  # type: ignore

from .chunkers import chunk_into_chapters_from_toc
from .config import LOG_FORMAT, LOG_LEVEL
from .llm import Usage
from .readers import PDFInput, pdf_to_markdown
from .utils import join_with_page_fences

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
FENCE = "=" * 64


def main():
    # Create Gradio interface with tabs for initial review and follow-up chat
    with gr.Blocks(title="Academic Writing Feedback Generator") as iface:
        document_content = gr.State("")
        rubric_content = gr.State(None)
        session_usage = gr.State(Usage.null())

        with gr.Tab("Initial Review"):
            gr.Markdown("# Academic Writing Feedback Generator")
            gr.Markdown(
                "Upload your thesis draft to receive detailed feedback. Optionally include a rubric for more specific evaluation."
            )

            with gr.Row():
                with gr.Column():
                    main_pdf = gr.File(
                        label="Upload Thesis Draft (PDF)", file_types=[".pdf"]
                    )
                    rubric_pdf = gr.File(
                        label="Upload Rubric (PDF, Optional)", file_types=[".pdf"]
                    )
                    text_only = gr.Checkbox(label="Text only (cheaper)", value=True)
                    review_button = gr.Button("Generate Review")

                with gr.Column():
                    feedback_output = gr.Markdown(label="Feedback", height="75vh")
                    usage_output = gr.Markdown()

        with gr.Tab("Follow-up Discussion"):
            with gr.Row():
                with gr.Column(scale=2):
                    feedback_output_2 = gr.Markdown(
                        label="Feedback",
                        height="80vh",
                        value="👆🏼 Let me review your document first before we discuss it",
                    )

                with gr.Column(scale=3):
                    gr.Markdown("# Discuss Your Feedback")
                    gr.Markdown(
                        "After reviewing the initial feedback, you can ask questions and discuss improvements here."
                    )

                    chatbot = gr.Chatbot(type="messages", height=500, min_height="66vh")
                    user_msg = gr.Textbox(
                        label="Ask a question about your document or the feedback"
                    )
                    chat_button = gr.Button("Send")
                    chat_usage_output = gr.Markdown()

        def do_review(
            main_pdf: PDFInput,
            rubric_pdf: PDFInput,
            text_only: bool,
        ) -> Iterator[tuple[str, str, str]]:
            if not isinstance(main_pdf, pymupdf.Document):
                main_pdf = pymupdf.Document(main_pdf)
            if rubric_pdf and not isinstance(rubric_pdf, pymupdf.Document):
                rubric_pdf = pymupdf.Document(rubric_pdf)

            # Store document content for the chat feature
            document_content.value = join_with_page_fences(
                pdf_to_markdown(main_pdf, page_chunks=True)
            )
            rubric_content.value = pdf_to_markdown(rubric_pdf) if rubric_pdf else None

            streaming_feedback, usage = None, None
            for streaming_feedback, usage in review_thesis_oneshot(
                main_pdf, rubric_pdf, text_only
            ):
                yield (
                    streaming_feedback,
                    format_usage(usage + session_usage.value),
                    streaming_feedback,
                )

            assert usage
            session_usage.value += usage

        def do_chat_exchange(
            user_message: str,
            history: list[gr.MessageDict],
            document_content: str,
            rubric_content: Optional[str] = None,
        ) -> Iterator[tuple[list[gr.MessageDict], str, str]]:
            yield (
                history + [{"role": "user", "content": user_message}],
                format_usage(session_usage.value),
                "",  # clear input field
            )

            usage = None
            for updated_history, usage in chat_with_advisor(
                user_message=user_message,
                history=history,
                document_content=document_content,
                rubric_content=rubric_content,
            ):
                yield (
                    updated_history,
                    format_usage(usage + session_usage.value),
                    gr.skip(),  # type: ignore
                )

            session_usage.value += usage

        review_button.click(
            do_review,
            inputs=[main_pdf, rubric_pdf, text_only],
            outputs=[feedback_output, usage_output, feedback_output_2],
        )

        chat_button.click(
            do_chat_exchange,
            inputs=[user_msg, chatbot, document_content, rubric_content],
            outputs=[chatbot, chat_usage_output, user_msg],
        )
        user_msg.submit(
            do_chat_exchange,
            inputs=[user_msg, chatbot, document_content, rubric_content],
            outputs=[chatbot, chat_usage_output, user_msg],
        )

    iface.launch()


def review_thesis_oneshot(
    main_pdf: PDFInput, rubric_pdf: Optional[PDFInput] = None, text_only: bool = False
) -> Iterator[tuple[str, Usage]]:
    """Process PDFs and generate comprehensive feedback."""
    logger.info("⏳ Loading submitted document...")
    pages = pdf_to_markdown(main_pdf, page_chunks=True)
    content = join_with_page_fences(pages)

    # Extract text from rubric if provided
    rubric_content = None
    if rubric_pdf:
        logger.info("⏳ Loading rubric...")
        rubric_content = pdf_to_markdown(rubric_pdf)

    logger.info("⏳ Generating feedback...")
    try:
        usage = None
        for feedback, usage in generate_feedback(content, rubric_content):
            yield feedback, usage
        assert usage
        logger.info(f"💲 Review usage: ${usage.cost:.3f} - {usage}")
    except Exception as e:
        yield format_error(e), Usage.null()
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


def chat_with_advisor(
    user_message: str,
    history: list[gr.MessageDict],
    document_content: str,
    rubric_content: Optional[str] = None,
) -> Iterator[tuple[list[gr.MessageDict], Usage]]:
    """Allow users to chat with the advisor about their document after the initial review."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    system_prompt = """You are an experienced academic writing advisor. You've already provided feedback on the user's document, and now they want to discuss it further.
    Help them understand your feedback, brainstorm improvements, and answer any questions they have about academic writing.
    Base your responses on the document content you were provided, but you can also offer general academic writing advice.
    Be specific, constructive, and encouraging."""

    # Prepare conversation history for the API
    messages: list[anthropic.types.MessageParam] = []
    for message in history:
        messages.append({"role": message["role"], "content": message["content"]})

    # Add the current message
    messages.append({"role": "user", "content": user_message})

    # Add document context to the system prompt
    full_system_prompt = system_prompt + (
        "\n\nDocument content for reference:\n"
        f"<document>\n{document_content}\n</document>"
    )
    if rubric_content:
        full_system_prompt += (
            f"\n\nRubric criteria:\n<document>\n{rubric_content}\n</document>"
        )

    try:
        text_so_far = ""
        usage = None

        with client.messages.stream(
            model="claude-3-7-sonnet-latest",
            max_tokens=2000,
            system=full_system_prompt,
            messages=messages,
        ) as stream:
            for event in stream:
                if event.type == "text":
                    text_so_far += event.text
                usage = Usage(**stream.current_message_snapshot.usage.model_dump())

                yield (
                    history
                    + [
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": text_so_far},
                    ],
                    usage,
                )
    except Exception as e:
        error_message = format_error(e)
        return (
            history
            + [
                {"role": "user", "content": user_message},
                {"role": "system", "content": error_message},
            ],
            Usage.null(),
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
    return f"### Cost of this session: 💲{usage.cost:.2f}"
