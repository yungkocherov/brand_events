import asyncio
import json
import logging
import re
import time
from functools import partial

from ddgs import DDGS
from mistralai.client import Mistral

from app.models import BrandEvent, BrandEventsResponse

logger = logging.getLogger(__name__)

EVENT_TYPES = {
    "market_exit": {
        "label": "Уход / приход на рынок",
        "query": "{brand} уход с рынка закрытие выход {year}",
    },
    "rebrand": {
        "label": "Ребрендинг / смена названия",
        "query": "{brand} ребрендинг смена названия логотип {year}",
    },
    "sanctions": {
        "label": "Санкции / ограничения",
        "query": "{brand} санкции ограничения запрет блокировка {year}",
    },
    "scandal": {
        "label": "Крупный скандал / суд",
        "query": "{brand} скандал суд штраф иск {year}",
    },
    "new_product": {
        "label": "Запуск нового продукта",
        "query": "{brand} запуск новый продукт сервис релиз {year}",
    },
    "management": {
        "label": "Смена собственника / руководства",
        "query": "{brand} смена CEO директор руководство назначение {year}",
    },
    "ad_campaign": {
        "label": "Крупная рекламная кампания",
        "query": "{brand} рекламная кампания спонсорство амбассадор {year}",
    },
    "supply": {
        "label": "Перебои с поставками / дефицит",
        "query": "{brand} дефицит перебои поставки нехватка {year}",
    },
    "price_change": {
        "label": "Изменение цен",
        "query": "{brand} повышение цен подорожание скидки {year}",
    },
    "merger": {
        "label": "Слияние / поглощение",
        "query": "{brand} слияние поглощение покупка сделка {year}",
    },
}

SYSTEM_PROMPT = """\
Ты — бизнес-аналитик. Тебе даны результаты поиска по бренду «{brand}».
Отфильтруй и оставь ТОЛЬКО те результаты, которые описывают реальные значимые \
события, непосредственно связанные с брендом «{brand}».

Отсей:
- Статьи, не связанные с брендом
- Общие новости рынка/отрасли без упоминания бренда
- Дубликаты одного и того же события
- Незначительные события

Для каждого оставшегося события укажи:
- event_name: краткое название события
- event_date: дата в формате YYYY-MM-DD (если неизвестна — YYYY-MM-01)
- description: 1-2 предложения
- impact_category: категория из списка результатов
- source_url: URL из результатов поиска
- source_title: домен источника

ВАЖНО: ответ СТРОГО в формате JSON-массива, без markdown, без ```json```:
[
  {{
    "event_name": "...",
    "event_date": "YYYY-MM-DD",
    "description": "...",
    "impact_category": "...",
    "source_url": "https://...",
    "source_title": "..."
  }}
]

Если ни один результат не подходит — верни пустой массив []."""


def _search_ddg(
    brand: str, event_types: list[str], year_from: int, year_to: int
) -> list[dict]:
    """Search DuckDuckGo for selected event types."""
    all_results = []
    seen_urls = set()

    queries = []
    for year in range(year_from, year_to + 1):
        for et in event_types:
            cfg = EVENT_TYPES.get(et)
            if not cfg:
                continue
            queries.append((cfg["query"].format(brand=brand, year=year), et))

    with DDGS() as ddgs:
        for query, category in queries:
            try:
                results = list(ddgs.text(query, max_results=5, region="ru-ru"))
            except Exception as e:
                logger.error(f"DDG search failed: {e}")
                results = []

            for r in results:
                url = r.get("href", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                all_results.append({
                    "title": r.get("title", "").strip(),
                    "href": url,
                    "body": r.get("body", "").strip(),
                    "category": category,
                })

            time.sleep(0.3)

    return all_results


def _analyze_with_mistral(
    api_key: str, brand: str, search_results: list[dict]
) -> str:
    """Use Mistral to filter and structure search results."""
    client = Mistral(api_key=api_key)

    formatted = []
    for i, r in enumerate(search_results, 1):
        cat_label = EVENT_TYPES.get(r["category"], {}).get("label", r["category"])
        formatted.append(
            f"{i}. [{cat_label}] {r['title']}\n"
            f"   URL: {r['href']}\n"
            f"   {r['body']}"
        )
    search_text = "\n\n".join(formatted)

    prompt = SYSTEM_PROMPT.format(brand=brand)

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": search_text},
        ],
        temperature=0.2,
        max_tokens=8000,
    )

    text = response.choices[0].message.content or ""
    logger.info(f"Mistral response length for '{brand}': {len(text)}")
    return text


def _parse_events(text: str, brand: str) -> list[BrandEvent]:
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

    events = []
    for item in data:
        try:
            events.append(BrandEvent(
                brand=brand,
                event_name=item.get("event_name", ""),
                event_date=item.get("event_date", ""),
                description=item.get("description", ""),
                impact_category=item.get("impact_category", "other"),
                sentiment="neutral",
                source_url=item.get("source_url", ""),
                source_title=item.get("source_title", ""),
            ))
        except Exception:
            continue

    return events


async def search_brand_events(
    brand: str,
    event_types: list[str] | None = None,
    year_from: int = 2022,
    year_to: int = 2025,
    api_key: str = "",
) -> BrandEventsResponse:
    if not event_types:
        event_types = list(EVENT_TYPES.keys())

    loop = asyncio.get_event_loop()
    logger.info(f"Searching '{brand}' ({year_from}-{year_to}), types: {event_types}")

    # Step 1: DDG search
    search_results = await loop.run_in_executor(
        None, partial(_search_ddg, brand, event_types, year_from, year_to)
    )

    logger.info(f"Brand '{brand}': {len(search_results)} raw results")
    if not search_results:
        return BrandEventsResponse(brand=brand, events=[])

    # Step 2: filter with Mistral (if API key provided)
    if api_key:
        try:
            ai_response = await loop.run_in_executor(
                None, partial(_analyze_with_mistral, api_key, brand, search_results)
            )
            events = _parse_events(ai_response, brand)
        except Exception as e:
            logger.error(f"Mistral failed for '{brand}': {e}")
            events = _raw_to_events(brand, search_results)
    else:
        events = _raw_to_events(brand, search_results)

    events.sort(key=lambda e: e.event_date)
    logger.info(f"Brand '{brand}': {len(events)} events after filtering")
    return BrandEventsResponse(brand=brand, events=events)


def _raw_to_events(brand: str, results: list[dict]) -> list[BrandEvent]:
    """Fallback: convert raw search results to events without AI."""
    events = []
    for r in results:
        date_str = _extract_date(f"{r['title']} {r['body']}")
        events.append(BrandEvent(
            brand=brand,
            event_name=r["title"],
            event_date=date_str,
            description=r["body"],
            impact_category=r["category"],
            sentiment="neutral",
            source_url=r["href"],
            source_title=_domain(r["href"]),
        ))
    return events


MONTHS_RU = {
    "январ": "01", "феврал": "02", "март": "03", "апрел": "04",
    "мая": "05", "мае": "05", "май": "05", "июн": "06",
    "июл": "07", "август": "08", "сентябр": "09",
    "октябр": "10", "ноябр": "11", "декабр": "12",
}


def _extract_date(text: str) -> str:
    m = re.search(r"(\d{1,2})[./](\d{1,2})[./](20\d{2})", text)
    if m:
        return f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"
    m = re.search(r"(20\d{2})-(\d{2})-(\d{2})", text)
    if m:
        return m.group(0)
    for month_prefix, month_num in MONTHS_RU.items():
        pattern = rf"(\d{{1,2}})\s+{month_prefix}\S*\s+(20\d{{2}})"
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return f"{m.group(2)}-{month_num}-{m.group(1).zfill(2)}"
    for month_prefix, month_num in MONTHS_RU.items():
        pattern = rf"{month_prefix}\S*\s+(20\d{{2}})"
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return f"{m.group(1)}-{month_num}-01"
    m = re.search(r"(20\d{2})", text)
    if m:
        return f"{m.group(1)}-01-01"
    return ""


def _domain(url: str) -> str:
    m = re.search(r"https?://(?:www\.)?([^/]+)", url)
    return m.group(1) if m else ""
