from pydantic import BaseModel
from datetime import date


class BrandRequest(BaseModel):
    brands: list[str]
    industry: str = ""
    event_types: list[str] = []
    year_from: int = 2022
    year_to: int = 2025
    api_key: str = ""


class BrandEvent(BaseModel):
    brand: str
    event_name: str
    event_date: str
    description: str
    impact_category: str  # revenue, awareness, reputation, operations, legal
    sentiment: str = "neutral"  # positive, negative, neutral
    source_url: str
    source_title: str


class BrandEventsResponse(BaseModel):
    brand: str
    events: list[BrandEvent]


class SearchResponse(BaseModel):
    results: list[BrandEventsResponse]


class CsvRequest(BaseModel):
    results: list[BrandEventsResponse]
    start_date: date
    end_date: date
    freq: str = "D"  # D=daily, W=weekly, M=monthly
