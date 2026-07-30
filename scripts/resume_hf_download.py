#!/usr/bin/env python3
"""Resilient segmented downloader for a single Hugging Face LFS file.

Downloads a file in chunks using HTTP Range requests, retrying failed chunks
instead of restarting the whole file.  Designed for large model shards on
unstable links.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 100 * 1024 * 1024  # 100 MB
DEFAULT_TIMEOUT = 120


def download_file(
    url: str,
    dest: Path,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = 10,
    headers: dict[str, str] | None = None,
) -> None:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Determine current progress.
    current_size = dest.stat().st_size if dest.exists() else 0
    if expected_size and current_size >= expected_size:
        logger.info("File already complete: %s", dest)
        return

    # Get total size if not provided.
    if expected_size is None:
        r = requests.head(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        expected_size = int(r.headers.get("Content-Length", 0))
        if expected_size == 0:
            raise ValueError("Could not determine file size")
        logger.info("Detected file size: %d bytes", expected_size)

    logger.info(
        "Resuming %s from %d / %d bytes (%.1f%%)",
        dest,
        current_size,
        expected_size,
        100 * current_size / expected_size,
    )

    # Download remaining chunks sequentially.
    start_byte = current_size
    while start_byte < expected_size:
        end_byte = min(start_byte + chunk_size - 1, expected_size - 1)
        chunk_headers = {"Range": f"bytes={start_byte}-{end_byte}"}
        if headers:
            chunk_headers.update(headers)

        for attempt in range(max_retries):
            try:
                logger.info(
                    "Downloading bytes %d-%d (attempt %d/%d)",
                    start_byte,
                    end_byte,
                    attempt + 1,
                    max_retries,
                )
                with requests.get(
                    url, headers=chunk_headers, stream=True, timeout=timeout
                ) as r:
                    r.raise_for_status()
                    # Open in append mode and seek to start_byte to avoid gaps.
                    with open(dest, "r+b" if dest.exists() else "wb") as f:
                        f.seek(start_byte)
                        downloaded = 0
                        for data in r.iter_content(chunk_size=1024 * 1024):
                            if data:
                                f.write(data)
                                downloaded += len(data)
                        logger.info(
                            "Chunk complete: %d bytes (total %.1f%%)",
                            downloaded,
                            100 * (start_byte + downloaded) / expected_size,
                        )
                        start_byte += downloaded
                        break
            except Exception as exc:
                logger.warning("Chunk %d-%d attempt failed: %s", start_byte, end_byte, exc)
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Failed to download bytes {start_byte}-{end_byte} after {max_retries} attempts"
                    ) from exc
                time.sleep(2 ** attempt)

    # Verify.
    if expected_size and dest.stat().st_size != expected_size:
        raise RuntimeError(
            f"Size mismatch: expected {expected_size}, got {dest.stat().st_size}"
        )
    if expected_sha256:
        logger.info("Verifying sha256...")
        sha = hashlib.sha256()
        with open(dest, "rb") as f:
            while True:
                data = f.read(16 * 1024 * 1024)
                if not data:
                    break
                sha.update(data)
        if sha.hexdigest() != expected_sha256.lower():
            raise RuntimeError(
                f"sha256 mismatch: expected {expected_sha256}, got {sha.hexdigest()}"
            )
        logger.info("sha256 verified")
    logger.info("Download complete: %s", dest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resilient HF file downloader")
    parser.add_argument("url", help="Direct download URL")
    parser.add_argument("dest", type=Path, help="Destination path")
    parser.add_argument("--size", type=int, help="Expected file size in bytes")
    parser.add_argument("--sha256", help="Expected sha256 hex digest")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--token", help="HF token for private repos")
    args = parser.parse_args()

    headers = {}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"

    download_file(
        url=args.url,
        dest=args.dest,
        expected_size=args.size,
        expected_sha256=args.sha256,
        chunk_size=args.chunk_size,
        timeout=args.timeout,
        headers=headers,
    )


if __name__ == "__main__":
    main()
