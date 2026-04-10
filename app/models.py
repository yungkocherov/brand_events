from pydantic import BaseModel
from datetime import date


class BrandRequest(BaseModel):
    brands: list[str]
    industry: str = ""
    api_key: str = ""
    provider: str = "mistral"
    model: str = ""


class CheckKeyRequest(BaseModel):
    api_key: str
    provider: str = "mistral"
    model: str = ""


class BrandEvent(BaseModel):
    brand: str
    event_name: str
    event_date: str
    description: str
    # One of: market_exit, rebrand, new_product, supply, ad_campaign,
    # scandal, sanctions, price_change, management, merger, other
    impact_category: str = "other"
    impact_score: int = 3  # 1-5: estimated impact on business metrics
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
