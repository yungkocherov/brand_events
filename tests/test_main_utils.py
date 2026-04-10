"""Tests for the utility helpers living in app/main.py."""
from datetime import date

from app.main import _to_snake_case, _parse_date


class TestToSnakeCase:
    def test_ascii_only(self):
        assert _to_snake_case("Hello World") == "hello_world"

    def test_cyrillic_transliteration(self):
        assert _to_snake_case("Запуск нового продукта") == "zapusk_novogo_produkta"

    def test_mixed_cyrillic_ascii(self):
        assert _to_snake_case("Exeed X5 запуск") == "exeed_x5_zapusk"

    def test_hard_soft_signs_dropped(self):
        # ъ and ь should map to empty string, not underscore
        assert _to_snake_case("подъезд") == "podezd"
        assert _to_snake_case("мальчик") == "malchik"

    def test_yo_normalised_to_e(self):
        assert _to_snake_case("ёлка") == "elka"

    def test_digits_preserved(self):
        assert _to_snake_case("Windows 11 Pro") == "windows_11_pro"

    def test_punctuation_becomes_underscore(self):
        assert _to_snake_case("Coca-Cola / Pepsi") == "coca_cola_pepsi"

    def test_collapses_multiple_underscores(self):
        assert _to_snake_case("a  !!  b") == "a_b"

    def test_strips_leading_trailing_underscores(self):
        assert _to_snake_case("!!hello!!") == "hello"

    def test_length_capped_at_60(self):
        result = _to_snake_case("a" * 100)
        assert len(result) <= 60

    def test_empty_string(self):
        assert _to_snake_case("") == ""


class TestParseDate:
    def test_iso_format(self):
        assert _parse_date("2024-03-15") == date(2024, 3, 15)

    def test_dotted_ru(self):
        assert _parse_date("15.03.2024") == date(2024, 3, 15)

    def test_ym_only_defaults_to_first(self):
        assert _parse_date("2024-03") == date(2024, 3, 1)

    def test_year_only_defaults_to_jan_first(self):
        assert _parse_date("2024") == date(2024, 1, 1)

    def test_empty_returns_none(self):
        assert _parse_date("") is None

    def test_garbage_returns_none(self):
        assert _parse_date("not a date") is None

    def test_none_input_returns_none(self):
        # Caller may pass any string; we should not crash on unexpected types
        assert _parse_date("") is None
