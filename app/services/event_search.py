import asyncio
import json
from functools import partial

from duckduckgo_search import DDGS

from app.models import BrandEvent, BrandEventsResponse


SEARCH_QUERIES = [
    "{brand} скандал суд штраф",
    "{brand} ребрендинг новый продукт запуск",
    "{brand} выручка санкции проблемы",
    "{brand} партнёрство слияние поглощение",
]

ANALYSIS_PROMPT = """\
Ты — бизнес-аналитик. Ниже результаты поиска по бренду «{brand}».
Проанализируй их и выдели значимые события, которые могли повлиять на бизнес-метрики:
выручку, узнаваемость, репутацию, операционную деятельность.

Примеры событий: ребрендинг, рекламные кампании, суды/штрафы, утечки данных,
перебои поставок, новые продукты, смена руководства, слияния, санкции, партнёрства.

Результаты поиска:
{search_results}

Верни от 3 до 8 наиболее значимых РЕАЛЬНЫХ событий.
ВАЖНО: ответ СТРОГО в формате JSON-массива, без markdown, без ```json```:
[
  {{
    "event_name": "краткое название",
    "event_date": "YYYY-MM-DD",
    "description": "2-3 предложения",
    "impact_category": "revenue|awareness|reputation|operations|legal",
    "source_url": "https://...",
    "source_title": "название источника"
  }}
]

Используй только факты из результатов поиска. Не придумывай события.
Если точная дата неизвестна, укажи хотя бы месяц (YYYY-MM-01).
"""


def _search_ddg(brand: str) -> list[dict]:
    """Run multiple DuckDuckGo searches for a brand and collect results."""
    all_results = []
    with DDGS() as ddgs:
        for query_template in SEARCH_QUERIES:
            query = query_template.format(brand=brand)
            try:
                results = list(ddgs.text(query, max_results=5))
                all_results.extend(results)
            except Exception:
                continue
    return all_results


def _analyze_with_chat(brand: str, search_results: list[dict]) -> str:
    """Use DuckDuckGo AI chat to analyze search results."""
    # Format search results for the prompt
    formatted = []
    for i, r in enumerate(search_results, 1):
        formatted.append(
            f"{i}. [{r.get('title', '')}]({r.get('href', '')})\n"
            f"   {r.get('body', '')}"
        )
    search_text = "\n\n".join(formatted)

    prompt = ANALYSIS_PROMPT.format(brand=brand, search_results=search_text)

    with DDGS() as ddgs:
        response = ddgs.chat(prompt, model="gpt-4o-mini")
    return response


async def search_brand_events(brand: str) -> BrandEventsResponse:
    loop = asyncio.get_event_loop()

    # Step 1: search the web
    search_results = await loop.run_in_executor(
        None, partial(_search_ddg, brand)
    )

    if not search_results:
        return BrandEventsResponse(brand=brand, events=[])

    # Step 2: analyze with AI chat
    ai_response = await loop.run_in_executor(
        None, partial(_analyze_with_chat, brand, search_results)
    )

    # Step 3: parse
    events = _parse_events(ai_response, brand, search_results)
    return BrandEventsResponse(brand=brand, events=events)


def _parse_events(
    text: str, brand: str, search_results: list[dict]
) -> list[BrandEvent]:
    """Parse events JSON from AI response."""
    text = text.strip()

    # Remove markdown code fences if present
    if "```" in text:
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    # Find JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []

    json_str = text[start : end + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return []

    # Build a lookup of real URLs from search results
    real_urls = {r.get("href", ""): r.get("title", "") for r in search_results}

    events = []
    for item in data:
        source_url = item.get("source_url", "")
        # If the AI hallucinated a URL, try to find the closest real one
        if source_url and source_url not in real_urls:
            for real_url in real_urls:
                if real_url and any(
                    kw in real_url
                    for kw in item.get("event_name", "").lower().split()[:2]
                ):
                    source_url = real_url
                    break
            else:
                # Fallback: use first search result URL
                if search_results:
                    source_url = search_results[0].get("href", source_url)

        try:
            events.append(
                BrandEvent(
                    brand=brand,
                    event_name=item.get("event_name", ""),
                    event_date=item.get("event_date", ""),
                    description=item.get("description", ""),
                    impact_category=item.get("impact_category", "other"),
                    source_url=source_url,
                    source_title=item.get("source_title", ""),
                )
            )
        except Exception:
            continue

    return events
