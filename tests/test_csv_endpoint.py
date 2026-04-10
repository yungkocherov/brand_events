"""Tests for the /api/csv endpoint — dummy-variable CSV generation.

The CSV logic is the user-facing output of the whole app: events are turned
into binary columns that flip from 0 to 1 on their event_date. Regressions
here silently corrupt every downstream model someone builds on top.
"""
import csv
import io

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def _parse_csv(body: bytes) -> tuple[list[str], list[list[str]]]:
    text = body.decode("utf-8-sig")  # endpoint writes BOM
    rows = list(csv.reader(io.StringIO(text)))
    return rows[0], rows[1:]


class TestCsvEndpoint:
    def test_single_event_monthly(self, client):
        payload = {
            "results": [
                {
                    "brand": "Nurofen",
                    "events": [
                        {
                            "brand": "Nurofen",
                            "event_name": "Price hike",
                            "event_date": "2024-02-15",
                            "description": "",
                            "impact_category": "price_change",
                            "impact_score": 3,
                            "sentiment": "neutral",
                            "source_url": "https://example.com/1",
                            "source_title": "example.com",
                        }
                    ],
                }
            ],
            "start_date": "2024-01-01",
            "end_date": "2024-04-01",
            "freq": "M",
        }
        resp = client.post("/api/csv", json=payload)
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/csv")

        header, rows = _parse_csv(resp.content)
        assert header[0] == "date"
        assert len(header) == 2
        assert header[1].startswith("Nurofen__")

        by_date = {r[0]: r[1] for r in rows}
        # January: before the event — 0
        assert by_date["2024-01-01"] == "0"
        # February, March, April: event has occurred (2024-02-15) — 1
        assert by_date["2024-02-01"] == "0"  # before 02-15
        assert by_date["2024-03-01"] == "1"
        assert by_date["2024-04-01"] == "1"

    def test_missing_event_date_column_is_blank(self, client):
        # Events without a date must still produce a column, but the cells
        # are blank (not 0/1) — the model consumer decides what to do.
        payload = {
            "results": [
                {
                    "brand": "Brand",
                    "events": [
                        {
                            "brand": "Brand",
                            "event_name": "Undated",
                            "event_date": "",
                            "description": "",
                            "impact_category": "other",
                            "impact_score": 3,
                            "sentiment": "neutral",
                            "source_url": "u",
                            "source_title": "t",
                        }
                    ],
                }
            ],
            "start_date": "2024-01-01",
            "end_date": "2024-01-03",
            "freq": "D",
        }
        resp = client.post("/api/csv", json=payload)
        assert resp.status_code == 200
        _, rows = _parse_csv(resp.content)
        # All cells for the undated column are empty strings
        assert all(r[1] == "" for r in rows)

    def test_snake_case_column_names(self, client):
        payload = {
            "results": [
                {
                    "brand": "Добрый",
                    "events": [
                        {
                            "brand": "Добрый",
                            "event_name": "Запуск новой линейки",
                            "event_date": "2024-06-01",
                            "description": "",
                            "impact_category": "new_product",
                            "impact_score": 3,
                            "sentiment": "positive",
                            "source_url": "u",
                            "source_title": "t",
                        }
                    ],
                }
            ],
            "start_date": "2024-05-01",
            "end_date": "2024-07-01",
            "freq": "M",
        }
        resp = client.post("/api/csv", json=payload)
        header, _ = _parse_csv(resp.content)
        # Cyrillic must be transliterated to ASCII snake_case
        assert header[1] == "Добрый__zapusk_novoy_lineyki"

    def test_duplicate_columns_deduplicated(self, client):
        # Two events with the same brand + same name collapse to a single
        # column (current behaviour: first occurrence wins).
        ev = {
            "brand": "Brand",
            "event_name": "Launch",
            "event_date": "2024-03-01",
            "description": "",
            "impact_category": "other",
            "impact_score": 3,
            "sentiment": "neutral",
            "source_url": "u",
            "source_title": "t",
        }
        payload = {
            "results": [{"brand": "Brand", "events": [ev, {**ev, "event_date": "2024-08-01"}]}],
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "freq": "D",
        }
        resp = client.post("/api/csv", json=payload)
        header, _ = _parse_csv(resp.content)
        # Only one data column (plus "date")
        assert len(header) == 2

    def test_weekly_frequency(self, client):
        payload = {
            "results": [{"brand": "B", "events": []}],
            "start_date": "2024-01-01",
            "end_date": "2024-01-22",
            "freq": "W",
        }
        resp = client.post("/api/csv", json=payload)
        _, rows = _parse_csv(resp.content)
        dates = [r[0] for r in rows]
        assert dates == ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"]

    def test_response_has_attachment_header(self, client):
        payload = {
            "results": [],
            "start_date": "2024-01-01",
            "end_date": "2024-01-01",
            "freq": "D",
        }
        resp = client.post("/api/csv", json=payload)
        assert "attachment" in resp.headers["content-disposition"]
        assert "brand_events.csv" in resp.headers["content-disposition"]
