"""Tests for the date-extraction pipeline in services/event_search.py.

These are the highest-leverage tests in the suite: the pipeline is pure
logic over strings, and bugs in it show up as silently wrong dates on the
timeline (the hardest class of bug to notice). Each test targets one of
the extractor layers so a regression points at the layer that broke.
"""
import pytest

from app.services.event_search import (
    _normalise_date,
    _date_from_url,
    _article_window,
    _date_from_jsonld,
    _site_rules_for,
    SITE_DATE_RULES,
)


# --- _normalise_date ---------------------------------------------------------

class TestNormaliseDate:
    def test_iso(self):
        assert _normalise_date("2024-03-15") == "2024-03-15"

    def test_iso_with_time_suffix(self):
        assert _normalise_date("2024-03-15T12:30:00Z") == "2024-03-15"

    def test_dotted_ru(self):
        assert _normalise_date("15.03.2024") == "2024-03-15"

    def test_dotted_ru_short_day(self):
        assert _normalise_date("5.3.2024") == "2024-03-05"

    def test_russian_text_full_month(self):
        assert _normalise_date("18 декабря 2023") == "2023-12-18"

    def test_russian_text_abbreviated_month(self):
        assert _normalise_date("7 июл 2022 г.") == "2022-07-07"

    def test_russian_text_may_edge_case(self):
        # "май"/"мае"/"мая" all share the "ма" prefix; must pick month 05
        assert _normalise_date("1 мая 2024") == "2024-05-01"

    def test_empty(self):
        assert _normalise_date("") == ""

    def test_garbage(self):
        assert _normalise_date("no date here") == ""

    def test_embedded_in_title(self):
        # og:title style string: "Заголовок | Forbes, 7 июля 2022 г."
        assert _normalise_date("Headline | Forbes, 7 июля 2022 г.") == "2022-07-07"


# --- _date_from_url ----------------------------------------------------------

class TestDateFromUrl:
    def test_ymd_path(self):
        assert _date_from_url("https://tass.ru/news/2023/07/19/article") == "2023-07-19"

    def test_ymd_path_short(self):
        assert _date_from_url("https://example.com/2023/7/9/foo") == "2023-07-09"

    def test_dmy_path_rbc_style(self):
        assert _date_from_url("https://rbc.ru/19/07/2023/abc") == "2023-07-19"

    def test_ymd_hyphen(self):
        assert _date_from_url("https://site.ru/post-2024-03-15-title") == "2024-03-15"

    def test_ym_only(self):
        assert _date_from_url("https://site.ru/2023/07/slug") == "2023-07-01"

    def test_no_date(self):
        assert _date_from_url("https://vc.ru/brand/12345-title") == ""


# --- _article_window ---------------------------------------------------------

class TestArticleWindow:
    def test_prefers_article_tag(self):
        html = (
            "<html><head></head><body>"
            "<div>sidebar noise DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD 10 октября 2015</div>"
            "<article>this is the real article content with enough length to pass the 200-char guard. " + "x" * 300 + "</article>"
            "</body></html>"
        )
        window = _article_window(html)
        assert "real article content" in window
        assert "sidebar noise" not in window

    def test_centers_on_h1(self):
        # 500 tokens of 8 chars = 4000 chars of noise before h1.
        # Window takes ~600 chars before h1, so ~75 tokens may leak in, but
        # the vast majority (the other ~425) must be cut off.
        prefix = "SIDEBAR " * 500
        html = f"<html><body>{prefix}<h1>Headline</h1>article body after h1</body></html>"
        window = _article_window(html)
        assert "Headline" in window
        assert "article body after h1" in window
        # At most ~10% of the original sidebar tokens should survive the crop
        assert window.count("SIDEBAR") < 120

    def test_h1_window_includes_byline_above(self):
        # Dates often sit just above <h1>; window must include ~600 chars before
        pre = "byline 15.03.2024 " + ("x" * 200)
        html = f"<html><body>{pre}<h1>Title</h1>body</body></html>"
        window = _article_window(html)
        assert "15.03.2024" in window

    def test_fallback_without_h1_or_article(self):
        html = "<html><body>" + "x" * 50000 + "</body></html>"
        window = _article_window(html)
        # Fallback caps at 25 KB
        assert len(window) <= 25000


# --- _date_from_jsonld -------------------------------------------------------

class TestDateFromJsonld:
    def test_prefers_article_typed_block(self):
        # A page with two JSON-LD blocks; the date we want lives in the
        # Article block, not the WebSite block.
        html = """
        <script type="application/ld+json">
        {"@type": "WebSite", "datePublished": "2010-01-01"}
        </script>
        <script type="application/ld+json">
        {"@type": "NewsArticle", "datePublished": "2024-03-15"}
        </script>
        """
        assert _date_from_jsonld(html) == "2024-03-15"

    def test_falls_back_to_any_block_if_no_article_typed(self):
        html = """
        <script type="application/ld+json">
        {"@type": "Organization", "datePublished": "2023-05-20"}
        </script>
        """
        assert _date_from_jsonld(html) == "2023-05-20"

    def test_ignores_date_modified(self):
        # We must pick datePublished, never dateModified
        html = """
        <script type="application/ld+json">
        {"@type": "Article", "dateModified": "2025-01-01", "datePublished": "2024-06-01"}
        </script>
        """
        assert _date_from_jsonld(html) == "2024-06-01"

    def test_no_jsonld(self):
        assert _date_from_jsonld("<html><body>no scripts</body></html>") == ""

    def test_blogposting_type_recognised(self):
        html = """
        <script type="application/ld+json">
        {"@type": "BlogPosting", "datePublished": "2023-11-03"}
        </script>
        """
        assert _date_from_jsonld(html) == "2023-11-03"


# --- _site_rules_for ---------------------------------------------------------

class TestSiteRulesFor:
    def test_exact_host(self):
        assert _site_rules_for("https://vc.ru/story/1") == SITE_DATE_RULES["vc.ru"]

    def test_www_prefix(self):
        assert _site_rules_for("https://www.insur-info.ru/press/1") == SITE_DATE_RULES["insur-info.ru"]

    def test_subdomain(self):
        # e.g. pro.rbc.ru — rules for rbc.ru should still apply via suffix match
        assert _site_rules_for("https://pro.rbc.ru/news/123") == SITE_DATE_RULES["rbc.ru"]

    def test_unknown_domain(self):
        assert _site_rules_for("https://unknown-site.com/article") == []
