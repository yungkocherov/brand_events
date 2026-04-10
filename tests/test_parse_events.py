"""Tests for _parse_events — the Mistral JSON response parser.

Mistral is unreliable about formatting: it wraps JSON in ```json``` fences,
adds trailing prose, or returns dates inside phrases like "[дата
публикации: 2024-03-15]". _parse_events must survive all of that.
"""
from app.services.event_search import _parse_events


class TestParseEvents:
    def test_clean_json_array(self):
        text = """[
          {
            "event_name": "Launch",
            "event_date": "2024-03-15",
            "description": "desc",
            "impact_category": "new_product",
            "impact_score": 4,
            "sentiment": "positive",
            "source_url": "https://example.com/1",
            "source_title": "example.com"
          }
        ]"""
        events = _parse_events(text, "Brand")
        assert len(events) == 1
        assert events[0].brand == "Brand"
        assert events[0].event_name == "Launch"
        assert events[0].event_date == "2024-03-15"
        assert events[0].impact_score == 4

    def test_markdown_fenced_json(self):
        # Mistral loves to wrap output in ```json fences despite instructions
        text = '```json\n[{"event_name":"X","event_date":"2024-01-02","description":"d","impact_category":"other","impact_score":3,"sentiment":"neutral","source_url":"u","source_title":"t"}]\n```'
        events = _parse_events(text, "Brand")
        assert len(events) == 1
        assert events[0].event_date == "2024-01-02"

    def test_trailing_prose_stripped(self):
        # Extra commentary outside the array — bracket scan should handle it
        text = 'Вот события:\n[{"event_name":"X","event_date":"2023-05-01","description":"d","impact_category":"other","impact_score":3,"sentiment":"neutral","source_url":"u","source_title":"t"}]\nВсе остальное — шум.'
        events = _parse_events(text, "Brand")
        assert len(events) == 1

    def test_date_with_prefix_phrase(self):
        # Mistral sometimes returns "[дата публикации: 2024-03-15]"
        # instead of a clean ISO date — we must extract the YMD.
        text = '[{"event_name":"X","event_date":"[дата публикации: 2024-03-15]","description":"d","impact_category":"other","impact_score":3,"sentiment":"neutral","source_url":"u","source_title":"t"}]'
        events = _parse_events(text, "Brand")
        assert len(events) == 1
        assert events[0].event_date == "2024-03-15"

    def test_missing_date_becomes_empty(self):
        text = '[{"event_name":"X","event_date":"","description":"d","impact_category":"other","impact_score":3,"sentiment":"neutral","source_url":"u","source_title":"t"}]'
        events = _parse_events(text, "Brand")
        assert events[0].event_date == ""

    def test_empty_array(self):
        assert _parse_events("[]", "Brand") == []

    def test_no_json_at_all(self):
        assert _parse_events("Ничего не найдено.", "Brand") == []

    def test_malformed_json_returns_empty(self):
        # json.loads should fail → function returns [] rather than raising
        assert _parse_events("[{malformed", "Brand") == []

    def test_non_integer_impact_score_defaults_to_3(self):
        text = '[{"event_name":"X","event_date":"2024-01-01","description":"d","impact_category":"other","impact_score":"high","sentiment":"neutral","source_url":"u","source_title":"t"}]'
        events = _parse_events(text, "Brand")
        assert events[0].impact_score == 3

    def test_integer_impact_score_preserved(self):
        text = '[{"event_name":"X","event_date":"2024-01-01","description":"d","impact_category":"other","impact_score":5,"sentiment":"neutral","source_url":"u","source_title":"t"}]'
        events = _parse_events(text, "Brand")
        assert events[0].impact_score == 5

    def test_multiple_events(self):
        text = """[
          {"event_name":"A","event_date":"2024-01-01","description":"d","impact_category":"other","impact_score":3,"sentiment":"neutral","source_url":"u","source_title":"t"},
          {"event_name":"B","event_date":"2024-02-02","description":"d","impact_category":"scandal","impact_score":5,"sentiment":"negative","source_url":"u2","source_title":"t2"}
        ]"""
        events = _parse_events(text, "Brand")
        assert len(events) == 2
        assert events[0].event_name == "A"
        assert events[1].impact_category == "scandal"
