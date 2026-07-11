from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen

DATA_URL = (
    "https://raw.githubusercontent.com/WhartonSABI/rugby-ep/"
    "main/data/phase_2018-19.csv"
)
GIT_BLOB_SHA = "6cc793ef8d21b5ec08dc7a21a37533ad4e7d6e35"


def fetch_phase_data(destination: str | Path, overwrite: bool = False) -> Path:
    path = Path(destination)
    if path.exists() and not overwrite:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(DATA_URL, headers={"User-Agent": "rugby-possession-value/0.1"})
    with urlopen(request, timeout=120) as response:  # noqa: S310 - fixed trusted URL
        path.write_bytes(response.read())
    return path
