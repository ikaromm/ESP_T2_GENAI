#!/usr/bin/env python3
"""Baixa o dataset FINDSum da pasta publica do Google Drive.

O FINDSum (Liu et al., 2022, licenca ODC-BY) e distribuido apenas via Google
Drive. A listagem de pastas publicas nao esta na API sem credencial, mas o HTML
da pagina da pasta embute um payload JSON (`window['_DRIVE_ivd']`) com os filhos
diretos. Este script percorre essa arvore, grava um manifesto e baixa os
arquivos com retomada, de modo que reexecucoes nao rebaixem nada.

Uso:
    python scripts/fetch_findsum.py                # lista + baixa em data/raw/findsum
    python scripts/fetch_findsum.py --manifest-only
    python scripts/fetch_findsum.py --refresh      # ignora o cache de listagem
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

FOLDER_ID = "1O8HwUOp0Uxepc-SF9Oq2alxWHz03FEUE"
FOLDER_MIME = "application/vnd.google-apps.folder"
UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO_ROOT / "data" / "raw" / "findsum"
CHUNK = 1 << 20  # 1 MiB


@dataclass
class DriveEntry:
    """Um no da arvore do Drive (arquivo ou pasta)."""

    id: str
    name: str
    mime: str
    path: str
    size: int | None = None

    @property
    def is_folder(self) -> bool:
        return self.mime == FOLDER_MIME


def _open(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=timeout)


def _decode_ivd(raw: str) -> list:
    """Converte o literal JS de `_DRIVE_ivd` em estrutura Python.

    `unicode_escape` e usado apenas para resolver as sequencias `\\xNN`; ele
    opera em latin-1, entao o resultado e reinterpretado como UTF-8 para
    preservar nomes acentuados.
    """
    unescaped = raw.encode("utf-8", "surrogatepass").decode("unicode_escape")
    text = unescaped.encode("latin-1", "backslashreplace").decode("utf-8", "replace")
    return json.loads(text)


def list_folder(folder_id: str, timeout: int = 60) -> list[DriveEntry]:
    """Lista os filhos diretos de uma pasta publica do Drive."""
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    html = _open(url, timeout=timeout).read().decode("utf-8", "replace")
    match = re.search(r"_DRIVE_ivd'\]\s*=\s*'([^']*)'", html, re.S)
    if not match:
        raise RuntimeError(
            f"payload _DRIVE_ivd nao encontrado para {folder_id}; "
            "a pasta pode estar privada ou o layout do Drive mudou"
        )
    payload = _decode_ivd(match.group(1))
    entries: list[DriveEntry] = []
    for item in payload[0] or []:
        size = None
        if len(item) > 13 and item[13] is not None:
            try:
                size = int(item[13])
            except (TypeError, ValueError):
                size = None
        entries.append(
            DriveEntry(id=item[0], name=item[2], mime=item[3], path="", size=size)
        )
    return entries


def crawl(folder_id: str, max_depth: int = 6, pause: float = 0.4) -> list[DriveEntry]:
    """Percorre a arvore em largura e devolve apenas os arquivos encontrados."""
    files: list[DriveEntry] = []
    queue: list[tuple[str, str, int]] = [(folder_id, "", 0)]
    while queue:
        fid, prefix, depth = queue.pop(0)
        print(f"  listando {prefix or '/'} ...", flush=True)
        for entry in list_folder(fid):
            entry.path = f"{prefix}/{entry.name}".lstrip("/")
            if entry.is_folder:
                if depth < max_depth:
                    queue.append((entry.id, entry.path, depth + 1))
            else:
                files.append(entry)
        time.sleep(pause)
    return files


def _confirm_token(html: str) -> str | None:
    match = re.search(r'name="confirm"\s+value="([^"]+)"', html)
    return match.group(1) if match else None


def download(entry: DriveEntry, dest_root: Path, retries: int = 3) -> str:
    """Baixa um arquivo, pulando o que ja esta completo. Devolve o status."""
    target = dest_root / entry.path
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        local = target.stat().st_size
        if entry.size is None or local == entry.size:
            return "cache"
        print(f"    tamanho divergente ({local} != {entry.size}), rebaixando", flush=True)

    base = "https://drive.usercontent.google.com/download"
    params = {"id": entry.id, "export": "download"}
    tmp = target.with_suffix(target.suffix + ".part")

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            url = f"{base}?{urllib.parse.urlencode(params)}"
            with _open(url, timeout=120) as response:
                ctype = response.headers.get("Content-Type", "")
                if "text/html" in ctype:
                    # Interstitial de aviso de virus para arquivos grandes.
                    html = response.read().decode("utf-8", "replace")
                    token = _confirm_token(html)
                    if not token:
                        raise RuntimeError("interstitial do Drive sem token confirm")
                    params["confirm"] = token
                    continue
                with tmp.open("wb") as fh:
                    while chunk := response.read(CHUNK):
                        fh.write(chunk)
            tmp.replace(target)
            return "ok"
        except Exception as exc:
            last_error = exc
            print(f"    tentativa {attempt}/{retries} falhou: {exc}", flush=True)
            time.sleep(2 * attempt)

    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"falha ao baixar {entry.path}: {last_error}")


def human(size: int | None) -> str:
    if size is None:
        return "?"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--folder-id", default=FOLDER_ID)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="reconstroi o manifesto")
    args = parser.parse_args(argv)

    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "manifest.json"

    if manifest_path.exists() and not args.refresh:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = [DriveEntry(**item) for item in payload["files"]]
        print(f"manifesto em cache: {len(files)} arquivos ({manifest_path})")
    else:
        print(f"percorrendo a pasta {args.folder_id} ...")
        files = crawl(args.folder_id)
        files.sort(key=lambda e: e.path)
        manifest_path.write_text(
            json.dumps(
                {"folder_id": args.folder_id, "files": [asdict(f) for f in files]},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"manifesto salvo: {len(files)} arquivos -> {manifest_path}")

    total = sum(f.size or 0 for f in files)
    print(f"\n{len(files)} arquivos, {human(total)} no total\n")
    for f in files:
        print(f"  {f.path:<60} {human(f.size):>10}")

    if args.manifest_only:
        return 0

    print()
    counts = {"ok": 0, "cache": 0}
    for index, entry in enumerate(files, 1):
        print(f"[{index}/{len(files)}] {entry.path} ({human(entry.size)})", flush=True)
        status = download(entry, dest)
        counts[status] += 1
        if status == "cache":
            print("    ja baixado, pulando", flush=True)

    print(f"\nconcluido: {counts['ok']} baixados, {counts['cache']} em cache")
    print(f"destino: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
