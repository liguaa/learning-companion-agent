from learning_companion.markdown import render_markdown


def test_learning_reply_formats_lists_emphasis_and_code():
    result = render_markdown(
        "**1. 计数版本**\n\n"
        "- `[1, 2, 3]`：外层走了 **2 轮**，交换 **0 次**。\n"
        "- `[5, 2, 9, 1, 5, 6]`：5 轮，6 次交换。\n\n"
        "```python\nprint('**literal**')\n```\n\n&#x20;"
    )
    assert "<strong>1. 计数版本</strong>" in result
    assert "<ul>" in result and result.count("<li>") == 2
    assert "<code>[1, 2, 3]</code>" in result
    assert "<strong>2 轮</strong>" in result
    assert '<pre><code class="language-python">' in result
    assert "**literal**" in result
    assert "&amp;#x20;" not in result


def test_untrusted_content_cannot_create_active_html_or_unsafe_links():
    result = render_markdown(
        '<script>alert(1)</script>\n\n<img src=x onerror="alert(1)">\n\n'
        "[click](javascript:alert%281%29)\n\n![tracker](https://example.com/pixel)"
    )
    assert "<script>" not in result
    assert "<img" not in result
    assert 'href="javascript:' not in result
    assert "&lt;script&gt;" in result
