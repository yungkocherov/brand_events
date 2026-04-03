import asyncio
import json
import os
import xml.etree.ElementTree as ET

import httpx
from duckduckgo_search import DDGS

from app.models import BrandEvent, BrandEventsResponse


YANDEX_SEARCH_URL = "https://searchapi.api.cloud.yandex.net/v2/web/searchAsync"
YANDEX_OPERATIONS_URL = "https://searchapi.api.cloud.yandex.net/v2/operations"

SEARCH_QUERIES = [
    "{brand} скандал суд штраф новости",
    "{brand} ребрендинг новый продукт запуск",
    "{brand} выручка санкции проблемы кризис",
    "{brand} партнёрство слияние поглощение сделка",
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


async def _yandex_search(client: httpx.AsyncClient, query: str) -> list[dict]:
    """Run a single Yandex search query and return parsed results."""
    api_key = os.environ["YANDEX_SEARCH_API_KEY"]
    folder_id = os.environ["YANDEX_FOLDER_ID"]

    body = {
        "query": {
            "searchType": "SEARCH_TYPE_RU",
            "queryText": query,
            "page": "0",
        },
        "groupSpec": {
            "groupMode": "GROUP_MODE_FLAT",
            "groupsOnPage": "10",
            "docsInGroup": "1",
        },
        "folderId": folder_id,
        "responseFormat": "FORMAT_XML",
    }

    # Start async search
    resp = await client.post(
        YANDEX_SEARCH_URL,
        headers={"Authorization": f"Api-Key {api_key}"},
        json=body,
    )
    if resp.status_code != 200:
        return []

    operation = resp.json()
    operation_id = operation.get("id")
    if not operation_id:
        return []

    # Poll for results
    for _ in range(15):
        await asyncio.sleep(1)
        poll_resp = await client.get(
            f"{YANDEX_OPERATIONS_URL}/{operation_id}",
            headers={"Authorization": f"Api-Key {api_key}"},
        )
        if poll_resp.status_code != 200:
            continue
        poll_data = poll_resp.json()
        if poll_data.get("done"):
            return _parse_yandex_xml(poll_data)

    return []


def _parse_yandex_xml(operation_data: dict) -> list[dict]:
    """Parse Yandex XML response from operation result."""
    response_data = operation_data.get("response", {})
    raw_xml = response_data.get("rawData", "")
    if not raw_xml:
        return []

    results = []
    try:
        root = ET.fromstring(raw_xml)
        # Yandex XML uses namespace sometimes; search without namespace too
        for group in root.iter("group"):
            doc = group.find(".//doc")
            if doc is None:
                continue
            url_el = doc.find("url")
            title_el = doc.find("title")
            snippet_el = doc.find("passages")

            url = url_el.text if url_el is not None else ""
            title = _xml_text(title_el) if title_el is not None else ""
            snippet = _xml_text(snippet_el) if snippet_el is not None else ""

            if url:
                results.append({
                    "title": title,
                    "href": url,
                    "body": snippet,
                })
    except ET.ParseError:
        pass

    return results


def _xml_text(element) -> str:
    """Extract all text from an XML element and its children."""
    return "".join(element.itertext()).strip()


def _analyze_with_chat(brand: str, search_results: list[dict]) -> str:
    """Use DuckDuckGo AI chat to analyze search results."""
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
    # Step 1: search Yandex
    all_results = []
    async with httpx.AsyncClient(timeout=30) as client:
        tasks = [
            _yandex_search(client, q.format(brand=brand))
            for q in SEARCH_QUERIES
        ]
        results_lists = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results_lists:
            if isinstance(res, list):
                all_results.extend(res)

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

    # Step 2: analyze with AI chat
    loop = asyncio.get_event_loop()
    ai_response = await loop.run_in_executor(
        None, _analyze_with_chat, brand, unique_results
    )

    # Step 3: parse
    events = _parse_events(ai_response, brand, unique_results)
    return BrandEventsResponse(brand=brand, events=events)


def _parse_events(
    text: str, brand: str, search_results: list[dict]
) -> list[BrandEvent]:
    """Parse events JSON from AI response."""
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
            # Try to find a matching real URL
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
