"""Fetch selected CSVs from CSIRO's public Annaburroo ZIP by HTTP ranges.

CSIRO collection csiro:58746, version 1, file 64873323 (2023 archive).
Only the two analysis tables are downloaded; the 593 MB image archive is not
stored locally. Requires curl with HTTPS and HTTP Range support.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import struct
import subprocess
from tempfile import TemporaryDirectory
import zlib


ARCHIVE_URL = "https://data.csiro.au/dap/ws/v2/collections/58746/data/64873323"
ARCHIVE_SIZE = 593_231_549
MEMBERS = {
    "Plot_Average_Data.csv": "Annaburroo experimental fire data/Plot_Average_Data.csv",
    "Burn_Period_Data.csv": "Annaburroo experimental fire data/Burn_Period_Data.csv",
}


def _range(specification: str) -> tuple[bytes, int]:
    with TemporaryDirectory() as temporary:
        output = Path(temporary) / "content"
        headers = Path(temporary) / "headers"
        command = ["curl", "--location", "--fail", "--silent", "--show-error",
                   "--header", f"Range: bytes={specification}", "--dump-header", str(headers),
                   "--output", str(output), ARCHIVE_URL]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError("CSIRO ranged download failed; verify public source access")
        matches = re.findall(r"Content-Range:\s*bytes (\d+)-(\d+)/(\d+)", headers.read_text(), re.IGNORECASE)
        if not matches:
            raise ValueError("CSIRO server did not honor the HTTP Range request")
        start, end, total = map(int, matches[-1])
        content = output.read_bytes()
        if total != ARCHIVE_SIZE or len(content) != end - start + 1:
            raise ValueError("CSIRO archive size or response length changed; review the source version")
        return content, start


def _central_members() -> dict[str, tuple[int, int, int, int]]:
    tail, tail_start = _range("-65536")
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise ValueError("ZIP end-of-central-directory not found")
    record = struct.unpack_from("<4s4H2LH", tail, eocd)
    _, disk, central_disk, entries_disk, entries_total, central_size, central_offset, comment_size = record
    if disk or central_disk or entries_disk != entries_total or comment_size:
        raise ValueError("unexpected multi-disk or commented ZIP")
    central, start = _range(f"{central_offset}-{central_offset + central_size - 1}")
    if start != central_offset:
        raise ValueError("central directory offset mismatch")
    members = {}
    cursor = 0
    for _ in range(entries_total):
        fields = struct.unpack_from("<4s6H3L5H2L", central, cursor)
        if fields[0] != b"PK\x01\x02":
            raise ValueError("invalid ZIP central directory")
        name_length, extra_length, comment_length = fields[10:13]
        name = central[cursor + 46:cursor + 46 + name_length].decode("utf-8")
        members[name] = (fields[4], fields[8], fields[9], fields[-1])
        cursor += 46 + name_length + extra_length + comment_length
    if cursor != len(central):
        raise ValueError("central directory length mismatch")
    return members


def fetch_tables(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    members = _central_members()
    records = []
    for output_name, archive_name in MEMBERS.items():
        method, compressed_size, uncompressed_size, local_offset = members[archive_name]
        local, start = _range(f"{local_offset}-{local_offset + 29}")
        if start != local_offset or local[:4] != b"PK\x03\x04":
            raise ValueError(f"{archive_name}: invalid local ZIP header")
        local_fields = struct.unpack_from("<4s5H3L2H", local)
        if local_fields[3] != method:
            raise ValueError(f"{archive_name}: compression method mismatch")
        name_length, extra_length = local_fields[-2:]
        data_offset = local_offset + 30 + name_length + extra_length
        compressed, start = _range(f"{data_offset}-{data_offset + compressed_size - 1}")
        if start != data_offset:
            raise ValueError(f"{archive_name}: compressed data offset mismatch")
        if method == 8:
            content = zlib.decompress(compressed, -15)
        elif method == 0:
            content = compressed
        else:
            raise ValueError(f"{archive_name}: unsupported ZIP compression method {method}")
        if len(content) != uncompressed_size or not content.startswith(b"FireID"):
            raise ValueError(f"{archive_name}: unexpected table content")
        target = directory / output_name
        target.write_bytes(content)
        records.append({"filename": output_name, "archive_member": archive_name,
                        "sha256": sha256(content).hexdigest(), "size_bytes": len(content)})
    return {"schema_version": "annaburroo-source-0.1", "collection": "csiro:58746",
            "collection_version": 1, "archive_file_id": 64873323, "archive_size_bytes": ARCHIVE_SIZE,
            "source_url": ARCHIVE_URL, "licence": "CC BY 4.0", "tables": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("data/raw/annaburroo"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = fetch_tables(args.directory)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
