import asyncio
import json
import logging
import re
import time
from functools import partial

import httpx
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
- event_date: дата в формате YYYY-MM-DD. Используй [дата публикации: ...] если она есть в источнике. Если её нет — оставь пустую строку. НЕ ВЫДУМЫВАЙ даты!
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
        # Prefer fetched date from page > date from URL
        date = r.get("fetched_date") or _date_from_url(r["href"])
        date_hint = f" [дата публикации: {date}]" if date else ""
        formatted.append(
            f"{i}. [{cat_label}] {r['title']}{date_hint}\n"
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

    # Step 2: enrich with article publication dates (parallel page fetch)
    mistral_input = search_results[:30]
    await _enrich_with_dates(mistral_input)
    logger.info(f"Brand '{brand}': enriched dates for {sum(1 for r in mistral_input if r.get('fetched_date'))}/{len(mistral_input)} articles")

    # Step 3: filter with Mistral (if API key provided)
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


async def _fetch_article_date(client: httpx.AsyncClient, url: str) -> str:
    """Fetch article HTML and extract publication date."""
    try:
        resp = await client.get(url, timeout=3, follow_redirects=True,
                                headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return ""
        html = resp.text[:50000]  # first 50KB
    except Exception:
        return ""

    # 1. JSON-LD datePublished
    m = re.search(r'"datePublished"\s*:\s*"(\d{4}-\d{2}-\d{2})', html)
    if m:
        return m.group(1)

    # 2. <meta property="article:published_time" content="2025-12-23...">
    m = re.search(
        r'<meta[^>]+(?:property|name)=["\'](?:article:published_time|pubdate|publishdate|date)["\'][^>]+content=["\']([^"\']+)',
        html, re.IGNORECASE,
    )
    if not m:
        m = re.search(
            r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\'](?:article:published_time|pubdate|publishdate|date)["\']',
            html, re.IGNORECASE,
        )
    if m:
        date_str = m.group(1)
        d = re.search(r"(\d{4})-(\d{2})-(\d{2})", date_str)
        if d:
            return d.group(0)

    # 3. <time datetime="2025-12-23">
    m = re.search(r'<time[^>]+datetime=["\'](\d{4}-\d{2}-\d{2})', html)
    if m:
        return m.group(1)

    # 4. Russian date in text: "23 декабря 2025"
    for prefix, num in MONTHS_RU.items():
        match = re.search(rf"(\d{{1,2}})\s+{prefix}\S*\s+(20\d{{2}})", html, re.IGNORECASE)
        if match:
            return f"{match.group(2)}-{num}-{match.group(1).zfill(2)}"

    # 5. DD.MM.YYYY
    m = re.search(r"(\d{1,2})\.(\d{1,2})\.(20\d{2})", html)
    if m:
        return f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"

    return ""


async def _enrich_with_dates(results: list[dict]) -> None:
    """Fetch all article pages in parallel and add 'fetched_date' field."""
    async with httpx.AsyncClient() as client:
        tasks = [_fetch_article_date(client, r["href"]) for r in results]
        dates = await asyncio.gather(*tasks, return_exceptions=True)
        for r, d in zip(results, dates):
            r["fetched_date"] = d if isinstance(d, str) else ""


def _date_from_url(url: str) -> str:
    """Extract YYYY-MM-DD from URL patterns."""
    # /YYYY/MM/DD/
    m = re.search(r"/(20\d{2})/(\d{1,2})/(\d{1,2})(?:/|\b)", url)
    if m:
        return f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"
    # /DD/MM/YYYY/ (e.g. РБК)
    m = re.search(r"/(\d{1,2})/(\d{1,2})/(20\d{2})(?:/|\b)", url)
    if m:
        return f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"
    # YYYY-MM-DD
    m = re.search(r"(20\d{2})-(\d{2})-(\d{2})", url)
    if m:
        return m.group(0)
    # /YYYY/MM/
    m = re.search(r"/(20\d{2})/(\d{1,2})(?:/|\b)", url)
    if m:
        return f"{m.group(1)}-{m.group(2).zfill(2)}-01"
    return ""
