"""Tests for the trusted-source filter in services/event_search.py."""
from app.services.event_search import _is_trusted, _get_trusted_domains, TRUSTED_GENERAL


class TestIsTrusted:
    def test_exact_host(self):
        assert _is_trusted("https://tass.ru/news/1", {"tass.ru"})

    def test_www_prefix_stripped(self):
        assert _is_trusted("https://www.tass.ru/news/1", {"tass.ru"})

    def test_subdomain_allowed(self):
        assert _is_trusted("https://pro.rbc.ru/news", {"rbc.ru"})

    def test_unrelated_domain_rejected(self):
        assert not _is_trusted("https://evil.com/tass.ru-fake", {"tass.ru"})

    def test_similar_suffix_rejected(self):
        # "faketass.ru" must NOT be trusted just because it ends in "tass.ru"
        assert not _is_trusted("https://faketass.ru/x", {"tass.ru"})

    def test_invalid_url(self):
        assert not _is_trusted("not-a-url", {"tass.ru"})

    def test_http_scheme(self):
        assert _is_trusted("http://kommersant.ru/x", {"kommersant.ru"})


class TestGetTrustedDomains:
    def test_general_domains_always_included(self):
        domains = _get_trusted_domains("")
        assert "rbc.ru" in domains
        assert "kommersant.ru" in domains
        assert domains >= TRUSTED_GENERAL

    def test_pharma_industry_adds_specific(self):
        domains = _get_trusted_domains("фармацевтика")
        assert "pharmvestnik.ru" in domains
        assert "vademec.ru" in domains
        # General ones still present
        assert "rbc.ru" in domains

    def test_auto_industry_substring_match(self):
        # "автомобили" should match the "авто" key in TRUSTED_BY_INDUSTRY
        domains = _get_trusted_domains("автомобили")
        assert "autonews.ru" in domains
        assert "zr.ru" in domains

    def test_case_insensitive_industry(self):
        upper = _get_trusted_domains("БАНКИ")
        lower = _get_trusted_domains("банки")
        assert upper == lower

    def test_unknown_industry_falls_back_to_general(self):
        domains = _get_trusted_domains("киберспорт")
        assert domains == TRUSTED_GENERAL

    def test_multiple_keys_match(self):
        # "фармацевтика и лекарства" should hit both "фарм" and "лекарств"
        domains = _get_trusted_domains("фармацевтика и лекарства")
        assert "pharmvestnik.ru" in domains
        assert "rlsnet.ru" in domains
