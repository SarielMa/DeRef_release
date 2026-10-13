"""FinTrace: scrape the public FMP stable API docs pages (network; no API key, no API calls).

Writes <out_dir>/_index.json and <out_dir>/<slug>.json (one per endpoint page; existing pages are skipped). The pages
are live, so a new scrape is not guaranteed to reproduce the cached pages used for the released data (scraped
2026-09-30); use the cached pages for an exact rebuild.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.build import fintrace  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, help="directory for the cached docs pages")
    a = ap.parse_args()
    fintrace.fetch_fmp_docs(Path(a.out_dir))


if __name__ == "__main__":
    main()
