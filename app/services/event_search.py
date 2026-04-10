import asyncio
import json
import logging
import re
import time
from functools import partial

import httpx
from ddgs import DDGS

from app.models import BrandEvent, BrandEventsResponse
from app.services.llm import complete as llm_complete

logger = logging.getLogger(__name__)

# Trusted news sources used for filtering DDG results.
# General-purpose Russian news outlets are always included.
TRUSTED_GENERAL = {
    "rbc.ru", "kommersant.ru", "vedomosti.ru", "forbes.ru", "tass.ru",
    "ria.ru", "interfax.ru", "lenta.ru", "gazeta.ru", "iz.ru",
    "rg.ru", "novayagazeta.ru", "expert.ru", "1prime.ru", "bfm.ru",
    "thebell.io", "rbc.ua", "finmarket.ru", "finanz.ru", "bcs-express.ru",
    "frankmedia.ru", "frankrg.com", "vc.ru", "cnews.ru", "tadviser.ru",
    "secretmag.ru", "rusbase.com", "sostav.ru", "adindex.ru",
    "reuters.com", "bloomberg.com", "ft.com",
}

# Industry-specific trusted sources.
# Keys are lowercase substrings; matched against user-entered industry text.
TRUSTED_BY_INDUSTRY = {
    # Auto market
    "автомоб": {"autonews.ru", "autoreview.ru", "zr.ru", "drive.ru",
                "auto.ru", "kolesa.ru", "5koleso.ru", "motor.ru",
                "quto.ru", "autostat.ru"},
    "авто": {"autonews.ru", "autoreview.ru", "zr.ru", "drive.ru",
             "auto.ru", "kolesa.ru", "5koleso.ru", "motor.ru",
             "quto.ru", "autostat.ru"},
    # Banks / finance
    "банк": {"banki.ru", "frankmedia.ru", "frankrg.com", "bfm.ru",
             "finanz.ru", "finmarket.ru", "1prime.ru", "bcs-express.ru",
             "thebell.io", "pro.rbc.ru"},
    "финанс": {"banki.ru", "frankmedia.ru", "frankrg.com", "finmarket.ru",
               "1prime.ru", "bcs-express.ru", "thebell.io", "pro.rbc.ru",
               "finanz.ru"},
    # Insurance
    "страхов": {"asn-news.ru", "insur-info.ru", "wiki-insurance.ru",
                "insrev.ru", "banki.ru", "asn-news.ru", "insurancesummit.ru"},
    # Food / FMCG / retail
    "продукт": {"retail.ru", "new-retail.ru", "foodretail.ru", "sfera.fm",
                "foodnewsweek.ru", "foodbay.com", "souzmoloko.ru",
                "foodmarkets.ru", "milknews.ru", "agrobook.ru"},
    "питан": {"retail.ru", "new-retail.ru", "foodretail.ru", "sfera.fm",
              "foodnewsweek.ru", "foodbay.com", "souzmoloko.ru",
              "foodmarkets.ru", "milknews.ru", "agrobook.ru"},
    "напит": {"retail.ru", "new-retail.ru", "foodretail.ru", "sfera.fm",
              "foodnewsweek.ru", "foodbay.com", "souzmoloko.ru",
              "foodmarkets.ru", "milknews.ru", "agrobook.ru"},
    "fmcg": {"retail.ru", "new-retail.ru", "foodretail.ru", "sfera.fm",
             "foodmarkets.ru", "sostav.ru", "adindex.ru"},
    # Pharma
    "фарм": {"pharmvestnik.ru", "pharmprom.ru", "remedium.ru", "vademec.ru",
             "apteka.ru", "pharmacopoeia.ru", "pharmacology.kz",
             "garant.ru", "rosminzdrav.ru", "rspchm.ru", "rlsnet.ru"},
    "лекарств": {"pharmvestnik.ru", "pharmprom.ru", "remedium.ru", "vademec.ru",
                 "apteka.ru", "rlsnet.ru", "garant.ru", "rosminzdrav.ru"},
    # Telecom / IT
    "телеком": {"comnews.ru", "cnews.ru", "tadviser.ru", "iksmedia.ru",
                "roem.ru", "habr.com"},
    "it": {"cnews.ru", "tadviser.ru", "comnews.ru", "iksmedia.ru",
           "roem.ru", "habr.com", "rusbase.com", "vc.ru"},
    # Real estate / construction
    "недвиж": {"cian.ru", "mirkvartir.ru", "bn.ru", "vsenovostroyki.ru",
               "domclick.ru"},
    "строит": {"stroygaz.ru", "vestnikstroy.ru", "construction.ru"},
    # Energy
    "энерг": {"oilcapital.ru", "neftegaz.ru", "oil.gov.ru", "energyland.info",
              "rusenergetika.ru"},
    "нефт": {"oilcapital.ru", "neftegaz.ru", "rupec.ru", "oilru.com"},
    # Marketing / advertising
    "реклам": {"sostav.ru", "adindex.ru", "advertology.ru", "marketing.by"},
    "маркетинг": {"sostav.ru", "adindex.ru", "marketing.hse.ru", "advertology.ru"},
}


def _get_trusted_domains(industry: str) -> set[str]:
    """Return general + industry-specific trusted domains."""
    domains = set(TRUSTED_GENERAL)
    if industry:
        ind_lower = industry.lower()
        for key, sites in TRUSTED_BY_INDUSTRY.items():
            if key in ind_lower:
                domains |= sites
    return domains


def _is_trusted(url: str, trusted: set[str]) -> bool:
    """Check if URL belongs to a trusted domain."""
    m = re.search(r"https?://(?:www\.)?([^/]+)", url)
    if not m:
        return False
    host = m.group(1).lower()
    return any(host == d or host.endswith("." + d) for d in trusted)


CATEGORY_LABELS = {
    "market_exit": "Уход / приход на рынок",
    "rebrand": "Ребрендинг / смена названия",
    "new_product": "Запуск нового продукта",
    "supply": "Перебои с поставками / дефицит",
    "ad_campaign": "Рекламная кампания",
    "scandal": "Скандал / суд",
    "sanctions": "Санкции / ограничения",
    "price_change": "Изменение цен",
    "management": "Смена руководства",
    "merger": "Слияние / поглощение",
    "other": "Другое",
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
- event_date: ТОЛЬКО дата в формате YYYY-MM-DD (например "2024-03-15"). БЕЗ скобок, БЕЗ слов "дата публикации". Если есть [дата публикации: X] в источнике — извлеки оттуда X. Если даты нет — оставь "". НЕ ВЫДУМЫВАЙ даты!
- description: 1-2 предложения
- impact_category: СТРОГО одно из: market_exit (уход/приход на рынок), rebrand (ребрендинг), new_product (новый продукт), supply (перебои поставок/дефицит), ad_campaign (реклама), scandal (скандал/суд), sanctions (санкции), price_change (изменение цен), management (смена руководства), merger (слияние), other (другое)
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
    "impact_category": "market_exit|rebrand|new_product|supply|ad_campaign|scandal|sanctions|price_change|management|merger|other",
    "impact_score": 1-5,
    "sentiment": "positive|negative|neutral",
    "source_url": "https://...",
    "source_title": "..."
  }}
]

Если ни один результат не подходит — верни пустой массив []."""


SEARCH_QUERY_TEMPLATES = [
    '"{brand}" {industry} новости события',
    '"{brand}" {industry} скандал суд кризис санкции',
    '"{brand}" {industry} запуск ребрендинг сделка',
    '"{brand}" {industry} уход с рынка приход выход закрытие',
    '"{brand}" {industry} цены подорожание дефицит перебои поставки',
]


def _search_ddg(brand: str, industry: str = "") -> list[dict]:
    """Search DuckDuckGo with broad queries, filter by trusted news sites."""
    all_results = []
    seen_urls = set()
    industry_part = industry if industry else ""
    trusted = _get_trusted_domains(industry)

    queries = [t.format(brand=brand, industry=industry_part).strip() for t in SEARCH_QUERY_TEMPLATES]

    with DDGS() as ddgs:
        for query in queries:
            try:
                # Request more results since we filter many out
                results = list(ddgs.text(query, max_results=25, region="ru-ru"))
            except Exception as e:
                logger.error(f"DDG search failed: {e}")
                results = []

            kept = 0
            for r in results:
                url = r.get("href", "")
                if not url or url in seen_urls:
                    continue
                if not _is_trusted(url, trusted):
                    continue
                seen_urls.add(url)
                all_results.append({
                    "title": r.get("title", "").strip(),
                    "href": url,
                    "body": r.get("body", "").strip(),
                })
                kept += 1

            logger.info(f"DDG: {kept}/{len(results)} kept (trusted) for '{query}'")
            time.sleep(0.5)

    return all_results


async def _analyze_with_llm(
    provider: str, api_key: str, model: str,
    brand: str, search_results: list[dict], industry: str = "",
) -> str:
    """Ask the selected LLM to filter and structure search results."""
    formatted = []
    for i, r in enumerate(search_results, 1):
        date = r.get("fetched_date") or _date_from_url(r["href"])
        date_hint = f" [дата публикации: {date}]" if date else ""
        formatted.append(
            f"{i}. {r['title']}{date_hint}\n"
            f"   URL: {r['href']}\n"
            f"   {r['body']}"
        )
    search_text = "\n\n".join(formatted)

    industry_note = f" (отрасль: {industry})" if industry else ""
    prompt = SYSTEM_PROMPT.format(brand=brand, industry_note=industry_note)

    text = await llm_complete(
        provider, api_key, model,
        system=prompt, user=search_text,
        max_tokens=8000, temperature=0,
    )
    logger.info(f"LLM response length for '{brand}' ({provider}/{model}): {len(text)}")
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
            # Extract clean YYYY-MM-DD from date field (Mistral sometimes wraps it)
            raw_date = str(item.get("event_date", ""))
            date_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", raw_date)
            clean_date = date_match.group(0) if date_match else ""
            events.append(BrandEvent(
                brand=brand,
                event_name=item.get("event_name", ""),
                event_date=clean_date,
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
    api_key: str = "",
    industry: str = "",
    model: str = "",
    provider: str = "mistral",
) -> BrandEventsResponse:
    loop = asyncio.get_event_loop()
    logger.info(f"Searching '{brand}', industry='{industry}', provider='{provider}', model='{model}'")

    # Step 1: DDG search
    search_results = await loop.run_in_executor(
        None, partial(_search_ddg, brand, industry)
    )

    logger.info(f"Brand '{brand}': {len(search_results)} raw results")
    if not search_results:
        return BrandEventsResponse(brand=brand, events=[])

    # Step 2: enrich with article publication dates (parallel page fetch)
    llm_input = search_results[:30]
    await _enrich_with_dates(llm_input)
    logger.info(f"Brand '{brand}': enriched dates for {sum(1 for r in llm_input if r.get('fetched_date'))}/{len(llm_input)} articles")

    # Step 3: filter with the selected LLM
    if api_key and model:
        events = []
        for attempt in range(3):
            try:
                ai_response = await _analyze_with_llm(
                    provider, api_key, model, brand, llm_input, industry,
                )
                events = _parse_events(ai_response, brand)
                if events:
                    break
                logger.warning(f"LLM returned 0 events for '{brand}', attempt {attempt + 1}/3")
            except Exception as e:
                logger.error(f"LLM attempt {attempt + 1}/3 failed for '{brand}': {e}")
            if attempt < 2:
                await asyncio.sleep(2)
        if not events:
            logger.warning(f"All LLM attempts failed for '{brand}', using raw results")
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
        date_str = r.get("fetched_date") or _date_from_url(r["href"]) \
            or _extract_date(f"{r['title']} {r['body']}")
        events.append(BrandEvent(
            brand=brand,
            event_name=r["title"],
            event_date=date_str,
            description=r["body"],
            impact_category="other",
            impact_score=3,
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


# Per-domain HTML date extraction rules.
# Each rule is a regex; the first capture group must yield a parseable date
# (YYYY-MM-DD, DD.MM.YYYY, or "18 декабря 2023" — all normalised by _normalise_date).
#
# IMPORTANT: rules are applied to a NARROW article-header window (a slice of
# HTML around the first <h1>), not to the whole page. This guarantees we pick
# the article's own publication date instead of a date from the sidebar,
# "related news" block, footer copyright, or comments. If the window-scoped
# search finds nothing, we fall back to the whole document only for the
# strictest patterns (JSON-LD, head meta).
#
# Therefore broad regexes like `\d{1,2}\.\d{1,2}\.20\d{2}` ARE safe here:
# their search area is bounded.
SITE_DATE_RULES: dict[str, list[str]] = {
    "vc.ru": [
        r'"date_publish_iso"\s*:\s*"(\d{4}-\d{2}-\d{2})',
        r'data-date=["\'](\d{4}-\d{2}-\d{2})',
    ],
    "insur-info.ru": [
        # CMS-specific: <I CLASS="grey">7 июля 2022 г.</I> sits directly above
        # the article title on every press page.
        r'<I\s+CLASS=["\']grey["\'][^>]*>\s*(\d{1,2}\s+(?:январ|феврал|март|апрел|ма[яй]|июн|июл|август|сентябр|октябр|ноябр|декабр)\S*\s+20\d{2})',
        r'press[_-]?date[^>]*>\s*(\d{1,2}\.\d{1,2}\.20\d{2})',
        r'class=["\'][^"\']*date[^"\']*["\'][^>]*>\s*(\d{1,2}\.\d{1,2}\.20\d{2})',
    ],
    "asn-news.ru": [
        r'class=["\'][^"\']*date[^"\']*["\'][^>]*>\s*(\d{1,2}\.\d{1,2}\.20\d{2})',
        r'pubdate[^>]*>\s*(\d{1,2}\.\d{1,2}\.20\d{2})',
        r'(\d{1,2}\s+(?:январ|феврал|март|апрел|ма[яй]|июн|июл|август|сентябр|октябр|ноябр|декабр)\S*\s+20\d{2})',
    ],
    "tass.ru": [
        r'data-time=["\'](\d{4}-\d{2}-\d{2})',
    ],
    "rbc.ru": [
        r'content=["\'](\d{4}-\d{2}-\d{2})T[^"\']*["\'][^>]+article:published_time',
    ],
    "kommersant.ru": [
        r'datetime=["\'](\d{4}-\d{2}-\d{2})',
    ],
    "banki.ru": [
        r'(\d{1,2}\.\d{1,2}\.20\d{2})',
    ],
    "frankmedia.ru": [
        r'datetime=["\'](\d{4}-\d{2}-\d{2})',
    ],
    # Forum/wiki-style site: the topic post itself sits in `.comment.newstopic`
    # and stamps its date in <small>DD.MM.YYYY...</small>. Comments below use
    # <span> instead, so the <small> form is unique to the topic-creator post.
    "foodmarkets.ru": [
        r'class=["\']comment\s+newstopic["\'][\s\S]{0,8000}?<small>\s*(\d{1,2}\.\d{1,2}\.20\d{2})',
        r'<small>\s*(\d{1,2}\.\d{1,2}\.20\d{2})',
    ],
}


# JSON-LD types that mark "this is THE article" — used to disambiguate when a
# page has several JSON-LD blocks (e.g. WebSite + NewsArticle + BreadcrumbList).
_LD_ARTICLE_TYPES_RE = re.compile(
    r'"@type"\s*:\s*"(?:NewsArticle|Article|BlogPosting|Report|ReportageNewsArticle)"',
    re.IGNORECASE,
)


def _normalise_date(s: str) -> str:
    """Convert any of YYYY-MM-DD, DD.MM.YYYY, '18 декабря 2023' to YYYY-MM-DD."""
    if not s:
        return ""
    s = s.strip()
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return m.group(0)
    m = re.search(r"(\d{1,2})\.(\d{1,2})\.(20\d{2})", s)
    if m:
        return f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"
    for prefix, num in MONTHS_RU.items():
        m = re.search(rf"(\d{{1,2}})\s+{prefix}\S*\s+(20\d{{2}})", s, re.IGNORECASE)
        if m:
            return f"{m.group(2)}-{num}-{m.group(1).zfill(2)}"
    return ""


def _site_rules_for(url: str) -> list[str]:
    """Return per-site regex rules matching this URL's domain."""
    host = _domain(url).lower()
    for d, rules in SITE_DATE_RULES.items():
        if host == d or host.endswith("." + d):
            return rules
    return []


def _article_window(html: str) -> str:
    """Return a slice of HTML that brackets the article header.

    Publication dates on news sites almost always sit immediately above or
    below the <h1>. Cropping to that area kills false positives from sidebar
    widgets ("most read", "related news"), header navigation, footer
    copyright, and comments.

    Strategy:
      1. Prefer the inside of <article>...</article> if present (capped).
      2. Otherwise, take a window around the first <h1>: 600 chars before
         and 2500 after — enough room for byline + date + lead paragraph.
      3. Fallback: first 25 KB of the document.
    """
    # 1. <article> tag content
    m = re.search(r"<article\b[^>]*>(.*?)</article>", html, re.DOTALL | re.IGNORECASE)
    if m and len(m.group(1)) > 200:
        return m.group(1)[:30000]

    # 2. Window around first <h1>
    h1 = re.search(r"<h1\b", html, re.IGNORECASE)
    if h1:
        start = max(0, h1.start() - 600)
        end = min(len(html), h1.start() + 2500)
        return html[start:end]

    # 3. Fallback
    return html[:25000]


def _date_from_jsonld(html: str) -> str:
    """Extract datePublished from a JSON-LD script that describes the article.

    A page may host several <script type="application/ld+json"> blocks
    (Article + WebSite + Organization + BreadcrumbList). Only the
    Article-typed block carries the real publication date. We iterate the
    blocks and prefer the one whose @type is an Article variant; if none
    qualifies, fall back to the first datePublished anywhere.

    Also explicitly looks for `datePublished` only (NOT `dateModified` —
    those drift after edits and we want the original publication time).
    """
    blocks = re.findall(
        r'<script[^>]+application/ld\+json[^>]*>(.*?)</script>',
        html, re.DOTALL | re.IGNORECASE,
    )
    # First pass: Article-typed blocks
    for block in blocks:
        if not _LD_ARTICLE_TYPES_RE.search(block):
            continue
        m = re.search(r'"datePublished"\s*:\s*"(\d{4}-\d{2}-\d{2})', block)
        if m:
            return m.group(1)
    # Second pass: any block with datePublished
    for block in blocks:
        m = re.search(r'"datePublished"\s*:\s*"(\d{4}-\d{2}-\d{2})', block)
        if m:
            return m.group(1)
    return ""


async def _fetch_article_date(client: httpx.AsyncClient, url: str) -> str:
    """Fetch article HTML and extract publication date.

    Multiple dates often appear on a page (sidebar, related news, comments,
    "updated" timestamps). We resolve this by:
      - searching a NARROW article-header window first for site rules and
        <time> tags;
      - preferring JSON-LD blocks of @type Article over generic ones;
      - using `article:published_time` over generic meta dates;
      - explicitly NOT looking at `dateModified`.

    Priority order:
      1. Date encoded in the URL path (most reliable when present).
      2. Site rule applied to the article-header window.
      3. JSON-LD datePublished from an Article-typed block.
      4. <head> meta `article:published_time` (or other publish meta).
      5. First <time datetime="..."> inside the article window.
      6. Site rule applied to the WHOLE document (last-resort widening).
      7. Russian-text date near a "опубликовано" / "дата" marker in window.
    """
    # 1. URL date
    url_date = _date_from_url(url)
    if url_date:
        return url_date

    try:
        resp = await client.get(url, timeout=3, follow_redirects=True,
                                headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return ""
        html = resp.text
    except Exception:
        return ""

    window = _article_window(html)
    site_rules = _site_rules_for(url)

    # 2. Site rules — only inside the article window (avoids sidebar/footer dates)
    for pattern in site_rules:
        m = re.search(pattern, window, re.IGNORECASE)
        if m:
            d = _normalise_date(m.group(1))
            if d:
                return d

    # 3. JSON-LD datePublished from an Article-typed block
    d = _date_from_jsonld(html)
    if d:
        return d

    # 4. <head> meta tags — prefer article:published_time, then generic
    head_end = html.lower().find("</head>")
    head = html[:head_end] if head_end > 0 else html[:15000]
    for prop_re in (
        r"article:published_time",
        r"(?:pubdate|publishdate|og:pubdate|dc\.date(?:\.issued)?|date)",
    ):
        m = re.search(
            rf'<meta[^>]+(?:property|name)=["\']{prop_re}["\'][^>]+content=["\']([^"\']+)',
            head, re.IGNORECASE,
        )
        if not m:
            m = re.search(
                rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\']{prop_re}["\']',
                head, re.IGNORECASE,
            )
        if m:
            d = _normalise_date(m.group(1))
            if d:
                return d

    # 5. First <time datetime="..."> inside the article window
    m = re.search(r'<time[^>]+datetime=["\'](\d{4}-\d{2}-\d{2})', window)
    if m:
        return m.group(1)

    # 5b. og:title — many old-school CMS sites (insur-info.ru and similar)
    # have a wrong <h1> (it's the section title, not the article title), so the
    # window heuristic misses. But they ALWAYS encode the date in og:title:
    #   <meta property="og:title" content="... | Source, 7 июля 2022 г." />
    # This is distinctive enough to apply to the whole document.
    og = re.search(
        r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)',
        head, re.IGNORECASE,
    )
    if og:
        d = _normalise_date(og.group(1))
        if d:
            return d

    # 6. Site rules — last-resort widen to whole document
    for pattern in site_rules:
        m = re.search(pattern, html, re.IGNORECASE)
        if m:
            d = _normalise_date(m.group(1))
            if d:
                return d

    # 7. Russian-text date near a publication marker, restricted to window
    marker = re.search(
        r'(?:опубликован|публикация|дата[^а-я]{0,8})[^<]{0,80}'
        r'(\d{1,2}[\s.]+(?:январ|феврал|март|апрел|ма[яй]|июн|июл|август|сентябр|октябр|ноябр|декабр)\S*[\s.]+20\d{2}'
        r'|\d{1,2}\.\d{1,2}\.20\d{2})',
        window, re.IGNORECASE,
    )
    if marker:
        d = _normalise_date(marker.group(1))
        if d:
            return d

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
