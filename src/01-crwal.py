"""Download, consolidate, and assess Alpaca SIP minute bars."""

from __future__ import annotations

from config import CALENDAR_FILE, REPORTS_DIR, ensure_project_directories
from reporting.acquisition import print_calendar_summary, print_final_report
from utils.acquisition import (
    build_daily_quality,
    build_market_calendar,
    build_quality_summary,
    consolidate_all_symbols,
    create_clients,
    download_chunks,
)


def main() -> None:
    """Run the complete download, consolidation, and reporting pipeline."""

    ensure_project_directories()
    data_client, trading_client = create_clients()
    calendar = build_market_calendar(trading_client)
    calendar.to_csv(CALENDAR_FILE, index=False)
    print_calendar_summary(calendar)
    download_chunks(data_client, calendar)
    consolidate_all_symbols()

    print("\n=== QUALITY REPORT ===")
    daily_quality = build_daily_quality(calendar)
    quality_summary = build_quality_summary(daily_quality)
    daily_quality.to_csv(REPORTS_DIR / "daily_data_quality.csv", index=False)
    quality_summary.to_csv(REPORTS_DIR / "data_quality_summary.csv")
    print_final_report(quality_summary)


if __name__ == "__main__":
    main()
