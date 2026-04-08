import asyncio
import csv
import io
import re
from datetime import date, timedelta

from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

from app.models import BrandRequest, SearchResponse, CsvRequest
from app.services.event_search import search_brand_events, EVENT_TYPES

app = FastAPI(title="Brand Events Finder")


@app.get("/")
async def index():
    return FileResponse("app/static/index.html")


@app.get("/api/event-types")
async def get_event_types():
    return {k: v["label"] for k, v in EVENT_TYPES.items()}


class CheckKeyRequest(BaseModel):
    api_key: str


@app.post("/api/check-key")
async def check_key(request: CheckKeyRequest):
    from mistralai.client import Mistral
    try:
        client = Mistral(api_key=request.api_key)
        client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/search", response_model=SearchResponse)
async def search_events(request: BrandRequest):
    tasks = [
        search_brand_events(brand, request.event_types, request.year_from, request.year_to, request.api_key, request.industry)
        for brand in request.brands
    ]
    results = await asyncio.gather(*tasks)
    return SearchResponse(results=results)


@app.post("/api/csv")
async def export_csv(request: CsvRequest):
    start = request.start_date
    end = request.end_date

    dates: list[date] = []
    current = start
    while current <= end:
        dates.append(current)
        if request.freq == "W":
            current += timedelta(weeks=1)
        elif request.freq == "M":
            month = current.month + 1
            year = current.year
            if month > 12:
                month = 1
                year += 1
            try:
                current = current.replace(year=year, month=month)
            except ValueError:
                current = current.replace(year=year, month=month, day=28)
        else:
            current += timedelta(days=1)

    all_events: list[tuple[str, str, date | None]] = []
    for brand_result in request.results:
        for ev in brand_result.events:
            event_date = _parse_date(ev.event_date)
            col_name = f"{ev.brand} | {ev.event_name}"
            all_events.append((col_name, ev.event_date, event_date))

    output = io.StringIO()
    writer = csv.writer(output)

    header = ["date"] + [e[0] for e in all_events]
    writer.writerow(header)

    for d in dates:
        row = [d.isoformat()]
        for _, _, event_date in all_events:
            if event_date is None:
                row.append("")
            else:
                row.append(1 if d >= event_date else 0)
        writer.writerow(row)

    csv_bytes = output.getvalue().encode("utf-8-sig")
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=brand_events.csv"},
    )


def _parse_date(date_str: str) -> date | None:
    """Parse date string in various formats: YYYY-MM-DD, DD.MM.YYYY, YYYY-MM, YYYY."""
    try:
        return date.fromisoformat(date_str)
    except (ValueError, TypeError):
        pass

    m = re.match(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", date_str)
    if m:
        return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))

    m = re.match(r"(\d{4})-(\d{1,2})", date_str)
    if m:
        return date(int(m.group(1)), int(m.group(2)), 1)

    m = re.match(r"(\d{4})", date_str)
    if m:
        return date(int(m.group(1)), 1, 1)

    return None


app.mount("/static", StaticFiles(directory="app/static"), name="static")
