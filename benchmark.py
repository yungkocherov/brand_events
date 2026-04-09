"""
Benchmark: search pipeline timing and raw vs filtered results.
Usage: python benchmark.py
"""
import asyncio
import json
import os
import time

from dotenv import load_dotenv
load_dotenv()

from app.services.event_search import (
    _search_ddg, _analyze_with_mistral, _parse_events, _raw_to_events, EVENT_TYPES
)

BRANDS = ["Нурофен", "Миг", "Спазмалгон", "Спазган"]
INDUSTRY = "фармацевтика"
YEAR_FROM = 2019
YEAR_TO = 2024
EVENT_TYPE_KEYS = list(EVENT_TYPES.keys())
API_KEY = os.environ.get("MISTRAL_API_KEY", "")

OUTPUT_FILE = "benchmark_results.json"


def run():
    report = {
        "params": {
            "brands": BRANDS,
            "industry": INDUSTRY,
            "years": f"{YEAR_FROM}-{YEAR_TO}",
            "event_types": EVENT_TYPE_KEYS,
            "api_key_provided": bool(API_KEY),
        },
        "brands": [],
    }

    total_start = time.time()

    for brand in BRANDS:
        print(f"\n{'='*60}")
        print(f"  Brand: {brand}")
        print(f"{'='*60}")

        brand_report = {"brand": brand, "steps": {}}

        # Step 1: DDG search
        t0 = time.time()
        raw_results = _search_ddg(brand, EVENT_TYPE_KEYS, YEAR_FROM, YEAR_TO, INDUSTRY)
        t_search = time.time() - t0

        brand_report["steps"]["1_ddg_search"] = {
            "time_sec": round(t_search, 1),
            "raw_count": len(raw_results),
            "raw_results": [
                {
                    "title": r["title"],
                    "url": r["href"],
                    "snippet": r["body"][:150],
                    "category": r["category"],
                }
                for r in raw_results
            ],
        }
        print(f"  [1] DDG search: {len(raw_results)} results in {t_search:.1f}s")

        # Step 2: Raw fallback (without AI)
        t0 = time.time()
        raw_events = _raw_to_events(brand, raw_results)
        t_raw = time.time() - t0

        brand_report["steps"]["2_raw_events"] = {
            "time_sec": round(t_raw, 3),
            "count": len(raw_events),
        }
        print(f"  [2] Raw events (no AI): {len(raw_events)} in {t_raw:.3f}s")

        # Step 3: Mistral filtering
        if API_KEY:
            t0 = time.time()
            try:
                ai_response = _analyze_with_mistral(API_KEY, brand, raw_results, INDUSTRY)
                t_ai = time.time() - t0
                filtered_events = _parse_events(ai_response, brand)

                brand_report["steps"]["3_mistral_filter"] = {
                    "time_sec": round(t_ai, 1),
                    "filtered_count": len(filtered_events),
                    "removed_count": len(raw_results) - len(filtered_events),
                    "events": [
                        {
                            "name": e.event_name,
                            "date": e.event_date,
                            "category": e.impact_category,
                            "sentiment": e.sentiment,
                            "description": e.description[:120],
                            "source": e.source_url,
                        }
                        for e in filtered_events
                    ],
                }
                print(f"  [3] Mistral filter: {len(raw_results)} -> {len(filtered_events)} in {t_ai:.1f}s")
            except Exception as e:
                t_ai = time.time() - t0
                brand_report["steps"]["3_mistral_filter"] = {
                    "time_sec": round(t_ai, 1),
                    "error": str(e),
                }
                print(f"  [3] Mistral FAILED: {e}")
        else:
            brand_report["steps"]["3_mistral_filter"] = {"skipped": True}
            print(f"  [3] Mistral skipped (no API key)")

        report["brands"].append(brand_report)

    total_time = time.time() - total_start
    report["total_time_sec"] = round(total_time, 1)

    # Summary
    print(f"\n{'='*60}")
    print(f"  TOTAL: {total_time:.1f}s")
    print(f"{'='*60}")
    for br in report["brands"]:
        s = br["steps"]
        search_t = s["1_ddg_search"]["time_sec"]
        raw_n = s["1_ddg_search"]["raw_count"]
        ai_step = s.get("3_mistral_filter", {})
        ai_t = ai_step.get("time_sec", 0)
        filtered_n = ai_step.get("filtered_count", "N/A")
        print(f"  {br['brand']}: search={search_t}s ({raw_n} raw) -> AI={ai_t}s ({filtered_n} filtered)")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
