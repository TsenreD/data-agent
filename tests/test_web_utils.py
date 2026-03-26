from agents.data_collection.web_utils import extract_links, extract_main_text


def test_extract_links_normalizes_relative_urls() -> None:
    html = """
    <html>
      <body>
        <a href="/alpha">Alpha</a>
        <a href="https://example.com/beta">Beta</a>
      </body>
    </html>
    """

    links = extract_links(html, "https://example.com/root")

    assert links == [
        {"url": "https://example.com/alpha", "text": "Alpha"},
        {"url": "https://example.com/beta", "text": "Beta"},
    ]


def test_extract_main_text_falls_back_to_bs4_when_trafilatura_is_unavailable() -> None:
    html = """
    <html>
      <body>
        <script>ignored()</script>
        <main>
          <h1>Title</h1>
          <p>Body text here.</p>
        </main>
      </body>
    </html>
    """

    result = extract_main_text(html)

    assert result["method"] in {"bs4", "trafilatura"}
    assert "Title" in result["text"]
    assert "Body text here." in result["text"]
