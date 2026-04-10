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
        "keywords": "уход с рынка закрытие выход приход",
    },
    "rebrand": {
        "label": "Ребрендинг / смена названия",
        "keywords": "ребрендинг смена названия логотип",
    },
    "new_product": {
        "label": "Запуск нового продукта",
        "keywords": "запуск новый продукт релиз линейка",
    },
    "supply": {
        "label": "Перебои с поставками / дефицит",
        "keywords": "дефицит перебои поставки нехватка",
    },
    "ad_campaign": {
        "label": "Крупная рекламная кампания",
        "keywords": "рекламная кампания спонсорство амбассадор",
    },
    "scandal": {
        "label": "Крупный скандал / суд",
        "keywords": "скандал суд штраф иск",
    },
    "sanctions": {
        "label": "Санкции / ограничения",
        "keywords": "санкции ограничения запрет блокировка",
    },
    "price_change": {
        "label": "Изменение цен",
        "keywords": "повышение цен подорожание скидки",
    },
    "management": {
        "label": "Смена собственника / руководства",
        "keywords": "смена CEO директор руководство назначение",
    },
    "merger": {
        "label": "Слияние / поглощение",
        "keywords": "слияние поглощение покупка сделка",
    },
    "pharma_registration": {
        "label": "Регистрация / отзыв препарата",
        "keywords": "регистрация отзыв препарат Минздрав",
    },
    "pharma_clinical": {
        "label": "Клинические исследования",
        "keywords": "клинические исследования испытания эффективность",
    },
    "pharma_safety": {
        "label": "Безопасность / побочные эффекты",
        "keywords": "побочные эффекты безопасность отзыв партии Росздравнадзор",
    },
}

SYSTEM_PROMPT = """\
Ты — бизнес-аналитик. Тебе даны результаты поиска по бренду «{brand}»{industry_note}.
Отфильтруй и оставь ТОЛЬКО те результаты, которые описывают реальные значимые \
события, непосредственно связанные с брендом «{brand}»{industry_note}.

Отсей:
- Статьи, не связанные с брендом «{brand}»
- Статьи про другие компании/продукты с похожим названием, но из ДРУГОЙ отрасли
- Общие новости рынка/отрасли без упоминания бренда
- Дубликаты одного и того же события
- Незначительные события

Для каждого оставшегося события укажи:
- event_name: краткое название события
- event_date: дата в формате YYYY-MM-DD (если неизвестна — YYYY-MM-01)
- description: 1-2 предложения
- impact_category: СТРОГО одно из: market_exit, rebrand, new_product, supply, ad_campaign, scandal, sanctions, price_change, management, merger, pharma_registration, pharma_clinical, pharma_safety (или custom_N если событие связано с пользовательской темой)
- impact_score: ЦЕЛОЕ число от 1 до 5, насколько событие повлияло на бизнес-метрики бренда (выручку, продажи, узнаваемость). 1=минимальное влияние, 5=критическое (уход с рынка, крупный скандал, ребрендинг)
- sentiment: СТРОГО одно из: positive, negative, neutral
- source_url: URL из результатов поиска
- source_title: домен источника

ВАЖНО: ответ СТРОГО в формате JSON-массива, без markdown, без ```json```:
[
  {{
    "event_name": "...",
    "event_date": "YYYY-MM-DD",
    "description": "...",
    "impact_category": "market_exit|rebrand|new_product|supply|ad_campaign|scandal|sanctions|price_change|management|merger|pharma_registration|pharma_clinical|pharma_safety|custom",
    "impact_score": 1-5,
    "sentiment": "positive|negative|neutral",
    "source_url": "https://...",
    "source_title": "..."
  }}
]

Если ни один результат не подходит — верни пустой массив []."""


def _search_ddg(
    brand: str, event_types: list[str], industry: str = "",
    custom_queries: list[str] | None = None,
) -> list[dict]:
    """Search DuckDuckGo with one query per event type + custom queries."""
    all_results = []
    seen_urls = set()
    industry_suffix = f" {industry}" if industry else ""

    # Build query list: standard types + custom
    queries = []
    for et in event_types:
        cfg = EVENT_TYPES.get(et)
        if cfg:
            queries.append((f'"{brand}" {cfg["keywords"]}{industry_suffix}', et))
    for i, cq in enumerate(custom_queries or []):
        queries.append((f'"{brand}" {cq}{industry_suffix}', f"custom_{i}"))

    with DDGS() as ddgs:
        for query, category in queries:
            try:
                results = list(ddgs.text(query, max_results=10, region="ru-ru"))
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

            logger.info(f"DDG: {len(results)} results for '{brand}' [{et}]")
            time.sleep(0.5)

    return all_results


def _analyze_with_mistral(
    api_key: str, brand: str, search_results: list[dict],
    industry: str = "", model: str = "open-mistral-nemo",
    custom_queries: list[str] | None = None,
) -> str:
    """Use Mistral to filter and structure search results."""
    client = Mistral(api_key=api_key)

    custom_queries = custom_queries or []
    formatted = []
    for i, r in enumerate(search_results, 1):
        cat = r["category"]
        if cat.startswith("custom_"):
            idx = int(cat.split("_")[1])
            cat_label = custom_queries[idx] if idx < len(custom_queries) else cat
        else:
            cat_label = EVENT_TYPES.get(cat, {}).get("label", cat)
        formatted.append(
            f"{i}. [{cat_label}] {r['title']}\n"
            f"   URL: {r['href']}\n"
            f"   {r['body']}"
        )
    search_text = "\n\n".join(formatted)

    industry_note = f" (отрасль: {industry})" if industry else ""

    # Build extended category list including custom ones
    custom_cats = ", ".join(f"custom_{i}" for i in range(len(custom_queries)))
    custom_note = ""
    if custom_queries:
        custom_list = ", ".join(f'custom_{i}={q!r}' for i, q in enumerate(custom_queries))
        custom_note = (
            f"\n\nДОПОЛНИТЕЛЬНО: используй категории {custom_cats} для событий, "
            f"связанных с темами: {custom_list}. "
            f"События по этим темам ОБЯЗАТЕЛЬНО включи в результат."
        )

    prompt = SYSTEM_PROMPT.format(brand=brand, industry_note=industry_note) + custom_note

    response = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": search_text},
        ],
        temperature=0,
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
                impact_score=int(item.get("impact_score", 3)) if str(item.get("impact_score", "")).isdigit() else 3,
                sentiment=item.get("sentiment", "neutral"),
                source_url=item.get("source_url", ""),
                source_title=item.get("source_title", ""),
            ))
        except Exception:
            continue

    return events


async def search_brand_events(
    brand: str,
    event_types: list[str] | None = None,
    custom_queries: list[str] | None = None,
    api_key: str = "",
    industry: str = "",
    model: str = "open-mistral-nemo",
) -> BrandEventsResponse:
    if not event_types:
        event_types = list(EVENT_TYPES.keys())

    loop = asyncio.get_event_loop()
    logger.info(f"Searching '{brand}', industry='{industry}', types: {event_types}")

    # Step 1: DDG search
    search_results = await loop.run_in_executor(
        None, partial(_search_ddg, brand, event_types, industry, custom_queries or [])
    )

    logger.info(f"Brand '{brand}': {len(search_results)} raw results")
    if not search_results:
        return BrandEventsResponse(brand=brand, events=[])

    # Step 2: filter with Mistral (if API key provided)
    # Limit input to ~30 results to avoid overloading Mistral context
    mistral_input = search_results[:30]
    if api_key:
        events = []
        for attempt in range(3):
            try:
                ai_response = await loop.run_in_executor(
                    None, partial(_analyze_with_mistral, api_key, brand, mistral_input, industry, model, custom_queries or [])
                )
                events = _parse_events(ai_response, brand)
                if events:
                    break
                logger.warning(f"Mistral returned 0 events for '{brand}', attempt {attempt + 1}/3")
            except Exception as e:
                logger.error(f"Mistral attempt {attempt + 1}/3 failed for '{brand}': {e}")
            if attempt < 2:
                await asyncio.sleep(2)
        if not events:
            logger.warning(f"All Mistral attempts failed for '{brand}', using raw results")
            events = _raw_to_events(brand, search_results)
    else:
        events = _raw_to_events(brand, search_results)

    events.sort(key=lambda e: (-e.impact_score, e.event_date))
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
