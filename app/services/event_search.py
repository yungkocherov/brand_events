import asyncio
import json
import logging
import os
from functools import partial

from ddgs import DDGS
from mistralai.client import Mistral

from app.models import BrandEvent, BrandEventsResponse

logger = logging.getLogger(__name__)

SITES = [
    "rbc.ru", "kommersant.ru", "vedomosti.ru", "forbes.ru",
    "tass.ru", "ria.ru", "interfax.ru", "lenta.ru",
    "gazeta.ru", "iz.ru", "vc.ru", "banki.ru",
    "sostav.ru", "adindex.ru", "retail.ru",
]

SITE_FILTER = " OR ".join(f"site:{s}" for s in SITES)

SEARCH_QUERIES = [
    "{brand} скандал суд штраф {year}",
    "{brand} ребрендинг новый продукт запуск {year}",
    "{brand} выручка санкции кризис {year}",
    "{brand} партнёрство слияние сделка {year}",
]

SYSTEM_PROMPT = """\
Ты — бизнес-аналитик. Проанализируй результаты поиска и выдели \
значимые события, которые могли повлиять на бизнес-метрики компании: \
выручку, узнаваемость, репутацию, операционную деятельность.

Примеры событий: ребрендинг, рекламные кампании, суды/штрафы, утечки данных, \
перебои поставок, новые продукты, смена руководства, слияния, санкции, партнёрства.

Верни от 5 до 15 наиболее значимых РЕАЛЬНЫХ событий. Чем больше — тем лучше.
Каждое событие должно быть уникальным — не дублируй.
ВАЖНО: ответ СТРОГО в формате JSON-массива, без markdown, без ```json```:
[
  {
    "event_name": "краткое название",
    "event_date": "YYYY-MM-DD",
    "description": "2-3 предложения",
    "impact_category": "revenue|awareness|reputation|operations|legal",
    "source_url": "https://...",
    "source_title": "название источника"
  }
]

Используй только факты из результатов поиска. Не придумывай события.
Если точная дата неизвестна, укажи хотя бы месяц (YYYY-MM-01)."""

USER_PROMPT = """\
Проанализируй результаты поиска по бренду «{brand}» за {year_from}-{year_to} годы \
и выдели значимые события:

{search_results}"""


def _search_ddg(brand: str, year_from: int, year_to: int) -> list[dict]:
    """Search DuckDuckGo for brand events on Russian news sites."""
    all_results = []
    years = range(year_from, year_to + 1)

    with DDGS() as ddgs:
        for year in years:
            for q_template in SEARCH_QUERIES:
                query = f"{q_template.format(brand=brand, year=year)} ({SITE_FILTER})"
                try:
                    results = list(ddgs.text(query, max_results=5, region="ru-ru"))
                    all_results.extend(results)
                    logger.info(f"DDG: {len(results)} results for '{brand} {year}'")
                except Exception as e:
                    logger.error(f"DDG search failed: {e}")
                    continue

    return all_results


def _analyze_with_mistral(
    brand: str, search_results: list[dict], year_from: int, year_to: int
) -> str:
    """Use Mistral to analyze search results."""
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    formatted = []
    for i, r in enumerate(search_results, 1):
        formatted.append(
            f"{i}. {r.get('title', '')}\n"
            f"   URL: {r.get('href', '')}\n"
            f"   {r.get('body', '')}"
        )
    search_text = "\n\n".join(formatted)
    user_text = USER_PROMPT.format(
        brand=brand, search_results=search_text,
        year_from=year_from, year_to=year_to,
    )

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        temperature=0.3,
        max_tokens=8000,
    )

    text = response.choices[0].message.content or ""
    logger.info(f"Mistral response length for '{brand}': {len(text)}")
    return text


async def search_brand_events(
    brand: str, year_from: int = 2022, year_to: int = 2025
) -> BrandEventsResponse:
    loop = asyncio.get_event_loop()

    # Step 1: search DuckDuckGo
    logger.info(f"Searching '{brand}' ({year_from}-{year_to})")
    all_results = await loop.run_in_executor(
        None, partial(_search_ddg, brand, year_from, year_to)
    )

    logger.info(f"Brand '{brand}': {len(all_results)} total search results")
    if not all_results:
        return BrandEventsResponse(brand=brand, events=[])

    # Deduplicate by URL
    seen_urls = set()
    unique_results = []
    for r in all_results:
        url = r.get("href", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(r)

    logger.info(f"Brand '{brand}': {len(unique_results)} unique results")

    # Step 2: analyze with Mistral
    try:
        ai_response = await loop.run_in_executor(
            None, partial(
                _analyze_with_mistral, brand, unique_results, year_from, year_to
            )
        )
    except Exception as e:
        logger.error(f"Mistral failed for '{brand}': {e}")
        return BrandEventsResponse(brand=brand, events=[])

    # Step 3: parse
    events = _parse_events(ai_response, brand, unique_results)
    return BrandEventsResponse(brand=brand, events=events)


def _parse_events(
    text: str, brand: str, search_results: list[dict]
) -> list[BrandEvent]:
    text = text.strip()

    if "```" in text:
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []

    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []

    real_urls = {r.get("href", "") for r in search_results}
    events = []
    for item in data:
        source_url = item.get("source_url", "")
        if source_url not in real_urls:
            for r in search_results:
                href = r.get("href", "")
                name_words = item.get("event_name", "").lower().split()[:2]
                if href and any(w in href.lower() for w in name_words if len(w) > 3):
                    source_url = href
                    break

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
