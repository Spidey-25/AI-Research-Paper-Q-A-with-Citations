#!/usr/bin/env python3
"""Download the corpus described by ``corpus/manifest.json`` from arXiv.

The PDFs are not committed to this repository. arXiv papers carry a mix of
licences and redistributing the files would be a grey area, so the repository
ships the *pointers* and this downloader instead. Run it once after cloning::

    python scripts/fetch_corpus.py

Files land in ``corpus/pdfs/`` (gitignored). Re-running is cheap: files that are
already present and non-trivially sized are skipped unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rag import config  # noqa: E402
from rag.corpus import ManifestError, Paper, load_manifest  # noqa: E402

#: arXiv asks automated clients to identify themselves and to rate-limit.
USER_AGENT = "rag-document-qa/1.0 (+https://github.com/vk4868/rag-document-qa; academic use; contact via repo issues)"
MIN_PDF_BYTES = 20_000
REQUEST_TIMEOUT = 60


def _looks_like_pdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < MIN_PDF_BYTES:
        return False
    try:
        with path.open("rb") as fh:
            return fh.read(5) == b"%PDF-"
    except OSError:
        return False


def download(paper: Paper, dest_dir: Path, force: bool, delay: float) -> tuple[str, str]:
    """Fetch one paper. Returns ``(status, detail)``."""
    try:
        import requests
    except ImportError:  # pragma: no cover - import guard
        return "error", "the 'requests' package is not installed (pip install -r requirements.txt)"

    target = dest_dir / paper.filename
    if _looks_like_pdf(target) and not force:
        return "skipped", f"already present ({target.stat().st_size:,} bytes)"

    url = paper.pdf_url
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
            stream=True,
        )
        response.raise_for_status()
    except Exception as exc:
        return "error", f"{type(exc).__name__} fetching {url}: {exc}"

    content_type = response.headers.get("Content-Type", "")
    if "pdf" not in content_type.lower() and "octet-stream" not in content_type.lower():
        return "error", f"{url} returned Content-Type {content_type!r}, expected a PDF"

    tmp = target.with_suffix(".part")
    try:
        with tmp.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if chunk:
                    fh.write(chunk)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        return "error", f"could not write {tmp}: {exc}"

    if not _looks_like_pdf(tmp):
        size = tmp.stat().st_size if tmp.exists() else 0
        tmp.unlink(missing_ok=True)
        return "error", f"downloaded file was not a usable PDF ({size:,} bytes)"

    tmp.replace(target)
    time.sleep(delay)  # be a good citizen towards arXiv
    return "downloaded", f"{target.stat().st_size:,} bytes"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download the arXiv corpus listed in corpus/manifest.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=config.MANIFEST_PATH)
    parser.add_argument("--out-dir", type=Path, default=config.PDF_DIR)
    parser.add_argument("--force", action="store_true", help="re-download files already present")
    parser.add_argument("--delay", type=float, default=3.0, help="seconds to pause between downloads")
    parser.add_argument("--only", nargs="*", metavar="ARXIV_ID", help="fetch only these arXiv ids")
    args = parser.parse_args(argv)

    try:
        papers = load_manifest(args.manifest)
    except ManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.only:
        wanted = set(args.only)
        papers = [p for p in papers if p.arxiv_id in wanted]
        if not papers:
            print(f"ERROR: no manifest entries matched {sorted(wanted)}", file=sys.stderr)
            return 2

    try:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"ERROR: cannot create {args.out_dir}: {exc}", file=sys.stderr)
        return 2

    print(f"Fetching {len(papers)} paper(s) into {args.out_dir}\n")
    failures = 0
    for paper in papers:
        status, detail = download(paper, args.out_dir, args.force, args.delay)
        symbol = {"downloaded": "OK  ", "skipped": "SKIP", "error": "FAIL"}[status]
        print(f"  [{symbol}] {paper.arxiv_id}  {paper.title[:58]}")
        print(f"          {detail}")
        print(f"          licence: {paper.licence}")
        if status == "error":
            failures += 1

    print()
    if failures:
        print(
            f"{failures} of {len(papers)} paper(s) failed. arXiv rate-limits aggressive "
            "clients; wait a minute and re-run - completed files are skipped.",
            file=sys.stderr,
        )
        return 1

    print("Corpus ready. Next: python -m rag.ingest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
