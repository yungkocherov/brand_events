"""End-to-end tests for _fetch_article_date.

These exercise the full priority chain (URL → site rules → JSON-LD →
meta → <time> → og:title → whole-doc site rules → marker text) against
realistic HTML fixtures. httpx is faked so no network is touched.
"""
import pytest

from app.services.event_search import _fetch_article_date


class FakeResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code


class FakeClient:
    """Minimal httpx.AsyncClient stand-in that serves a fixed HTML string."""

    def __init__(self, html: str, status_code: int = 200, raises: Exception | None = None):
        self._html = html
        self._status = status_code
        self._raises = raises

    async def get(self, url, **kwargs):
        if self._raises:
            raise self._raises
        return FakeResponse(self._html, self._status)


@pytest.mark.asyncio
class TestFetchArticleDate:
    async def test_url_date_wins_over_everything(self):
        # URL carries a YMD; body contains a different date in JSON-LD.
        # URL takes precedence — it's the most reliable signal.
        html = '<script type="application/ld+json">{"@type":"Article","datePublished":"2020-01-01"}</script>'
        client = FakeClient(html)
        d = await _fetch_article_date(client, "https://tass.ru/news/2023/07/19/foo")
        assert d == "2023-07-19"

    async def test_jsonld_article_datepublished(self):
        html = """
        <html><head></head><body>
        <script type="application/ld+json">
        {"@type": "NewsArticle", "datePublished": "2024-03-15T10:00:00Z"}
        </script>
        <h1>Title</h1>
        </body></html>
        """
        client = FakeClient(html)
        d = await _fetch_article_date(client, "https://example.com/news/foo")
        assert d == "2024-03-15"

    async def test_meta_article_published_time(self):
        html = """
        <html><head>
        <meta property="article:published_time" content="2023-11-20T08:30:00+03:00">
        </head><body><h1>x</h1></body></html>
        """
        client = FakeClient(html)
        d = await _fetch_article_date(client, "https://example.com/article")
        assert d == "2023-11-20"

    async def test_og_title_russian_fallback(self):
        # Simulates the insur-info.ru case: wrong h1, no JSON-LD, no
        # published_time meta, but og:title carries the date in Russian.
        html = """
        <html><head>
        <meta property="og:title" content="Headline | Forbes, 7 июля 2022 г." />
        </head><body>
        <h1>Section Title Not The Article</h1>
        <div>unrelated 15.01.2019 sidebar date</div>
        </body></html>
        """
        client = FakeClient(html)
        d = await _fetch_article_date(client, "https://oldsite.ru/press/1/")
        assert d == "2022-07-07"

    async def test_head_meta_beats_sidebar_time_tag(self):
        # The sidebar's <time datetime> (older date) must not win over the
        # canonical <head> meta. This exercises the rule ordering:
        # head meta comes before the <time>-in-window fallback.
        html = """
        <html><head>
        <meta property="article:published_time" content="2024-06-01T12:00:00+03:00">
        </head><body>
        <aside><time datetime="2010-10-10">old</time></aside>
        <article>
          <h1>News</h1>
          <p>Long article body padded out so the <article> branch of
          _article_window triggers instead of the h1 window heuristic.
          """ + ("x " * 200) + """</p>
        </article>
        </body></html>
        """
        client = FakeClient(html)
        d = await _fetch_article_date(client, "https://example.com/news/some-slug")
        assert d == "2024-06-01"

    async def test_time_tag_in_window(self):
        html = """
        <html><head></head><body>
        <article>
          <h1>Title</h1>
          <time datetime="2023-09-14">14 September</time>
          <p>body</p>
        </article>
        </body></html>
        """
        client = FakeClient(html)
        d = await _fetch_article_date(client, "https://example.com/post")
        assert d == "2023-09-14"

    async def test_http_error_returns_empty(self):
        client = FakeClient("", status_code=500)
        d = await _fetch_article_date(client, "https://example.com/post")
        assert d == ""

    async def test_network_exception_returns_empty(self):
        client = FakeClient("", raises=RuntimeError("connection reset"))
        d = await _fetch_article_date(client, "https://example.com/post")
        assert d == ""

    async def test_prefers_datepublished_over_datemodified(self):
        # If a page has both, we must never pick dateModified — it drifts
        # after edits and is semantically the wrong answer.
        html = """
        <script type="application/ld+json">
        {"@type": "Article",
         "dateModified": "2025-02-10",
         "datePublished": "2024-01-05"}
        </script>
        """
        client = FakeClient(html)
        d = await _fetch_article_date(client, "https://example.com/x")
        assert d == "2024-01-05"

    async def test_insur_info_grey_italic_rule(self):
        # Simulates the real insur-info.ru CMS pattern: <I CLASS="grey">
        # sits right above the article title in plain text.
        html = """
        <html><head>
        <meta property="og:title" content="Страхование | Источник, 5 мая 2023 г." />
        </head><body>
        <h1>Пресса о страховании</h1>
        <div>Sidebar content</div>
        <I CLASS="grey">5 мая 2023 г.</I>
        <span class="h3">Article Title Here</span>
        </body></html>
        """
        client = FakeClient(html)
        d = await _fetch_article_date(client, "https://www.insur-info.ru/press/999/")
        # Either og:title or <I class="grey"> rule can win — both yield the
        # same date, which is the whole point.
        assert d == "2023-05-05"
