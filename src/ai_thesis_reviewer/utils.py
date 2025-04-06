from typing import Sequence


def join_with_page_fences(pages: Sequence[str]) -> str:
    return "\n".join(
        f"<-------[page {i}]------->\n{content}" for i, content in enumerate(pages, 1)
    )
