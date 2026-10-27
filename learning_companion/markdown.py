"""Render assistant Markdown with raw HTML and remote images disabled."""

from markdown_it import MarkdownIt

renderer = MarkdownIt("js-default", {"breaks": True}).disable("image")


def render_markdown(content: str) -> str:
    return renderer.render(content)
