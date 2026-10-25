"""File/URL loaders. Each returns a :class:`LoadedDocument` (normalized text + metadata)."""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

__all__ = [
    "LoadedDocument",
    "load_path",
    "load_url",
    "load_text",
    "load_markdown",
    "load_html",
    "load_pdf",
    "load_code",
    "load_json",
    "load_csv",
    "load_packets",
    "parse_front_matter",
    "html_to_text",
    "CODE_EXTS",
    "SUPPORTED_EXTS",
]


@dataclass
class LoadedDocument:
    text: str
    title: str = ""
    kind: str = "text"
    uri: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


CODE_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".c", ".h", ".cpp", ".hpp", ".cs", ".rb",
    ".php", ".swift", ".kt", ".scala", ".sh", ".sql", ".r", ".lua", ".m", ".toml", ".yaml", ".yml", ".ini",
}
TEXT_EXTS = {".txt", ".text", ".rst", ".log", ""}
MD_EXTS = {".md", ".markdown", ".mdx"}
HTML_EXTS = {".html", ".htm", ".xhtml"}
SUPPORTED_EXTS = CODE_EXTS | TEXT_EXTS | MD_EXTS | HTML_EXTS | {".pdf", ".json", ".csv", ".tsv", ".jsonl"}


# ----------------------------------------------------------------------------- helpers
_FM_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)


def parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Parse a simple ``---``-delimited ``key: value`` header. Returns ``(meta, body)``."""
    m = _FM_RE.match(text)
    if not m:
        return {}, text
    meta: dict[str, Any] = {}
    for line in m.group(1).splitlines():
        if ":" not in line or line.strip().startswith("#"):
            continue
        k, v = line.split(":", 1)
        v = v.strip().strip('"').strip("'")
        if re.fullmatch(r"-?\d+(\.\d+)?", v):
            meta[k.strip()] = float(v) if "." in v else int(v)
        elif v.startswith("[") and v.endswith("]"):
            meta[k.strip()] = [x.strip().strip("'\"") for x in v[1:-1].split(",") if x.strip()]
        else:
            meta[k.strip()] = v
    return meta, text[m.end():]


def _read(path: Path) -> str:
    data = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ----------------------------------------------------------------------------- loaders
def load_text(path: Path, raw: str | None = None) -> LoadedDocument:
    raw = _read(path) if raw is None else raw
    meta, body = parse_front_matter(raw)
    return LoadedDocument(_normalize(body), str(meta.get("title") or path.stem), "text", str(path), meta)


def load_markdown(path: Path, raw: str | None = None) -> LoadedDocument:
    raw = _read(path) if raw is None else raw
    meta, body = parse_front_matter(raw)
    title = meta.get("title")
    if not title:
        m = re.search(r"^#\s+(.+)$", body, re.M)
        title = m.group(1).strip() if m else path.stem
    return LoadedDocument(_normalize(body), str(title), "markdown", str(path), meta)


def html_to_text(html: str) -> tuple[str, str]:
    """Return ``(title, text)``. Uses BeautifulSoup when installed, else a regex fallback."""
    try:
        from bs4 import BeautifulSoup  # type: ignore[import-not-found]

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg"]):
            tag.decompose()
        title = soup.title.get_text(strip=True) if soup.title else ""
        blocks = []
        for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "blockquote", "pre", "td"]):
            t = el.get_text(" ", strip=True)
            if t:
                blocks.append(t)
        text = "\n\n".join(blocks) if blocks else soup.get_text("\n", strip=True)
        return title, text
    except ImportError:
        m = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
        title = re.sub(r"\s+", " ", m.group(1)).strip() if m else ""
        body = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
        body = re.sub(r"(?i)</(p|div|h[1-6]|li|tr|br|blockquote)>|<br\s*/?>", "\n\n", body)
        body = re.sub(r"<[^>]+>", " ", body)
        import html as _html

        body = _html.unescape(body)
        body = re.sub(r"[ \t]+", " ", body)
        return _html.unescape(title), body


def load_html(path: Path, raw: str | None = None, uri: str | None = None) -> LoadedDocument:
    raw = _read(path) if raw is None else raw
    title, text = html_to_text(raw)
    return LoadedDocument(_normalize(text), title or path.stem, "html", uri or str(path), {})


def load_pdf(path: Path) -> LoadedDocument:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - depends on env
        raise ImportError("PDF support needs pypdf: pip install 'noesis[pdf]'") from e
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        t = page.extract_text() or ""
        if t.strip():
            pages.append(t)
        del i
    title = ""
    try:
        title = (reader.metadata.title or "") if reader.metadata else ""
    except Exception:  # noqa: BLE001 - malformed metadata is common
        title = ""
    return LoadedDocument(_normalize("\n\n".join(pages)), title or path.stem, "pdf", str(path), {"pages": len(reader.pages)})


def load_code(path: Path, raw: str | None = None) -> LoadedDocument:
    raw = _read(path) if raw is None else raw
    lang = path.suffix.lstrip(".")
    return LoadedDocument(
        raw.replace("\r\n", "\n"), path.name, "code", str(path), {"language": lang, "source_type": "code"}
    )


def _flatten(obj: Any, prefix: str = "") -> list[str]:
    lines: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            lines += _flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:5000]):
            lines += _flatten(v, f"{prefix}[{i}]")
    else:
        lines.append(f"{prefix}: {obj}")
    return lines


def load_json(path: Path, raw: str | None = None) -> LoadedDocument:
    raw = _read(path) if raw is None else raw
    data = json.loads(raw)
    lines = _flatten(data)
    # group into paragraphs of 20 lines so the chunker has boundaries
    text = "\n\n".join("\n".join(lines[i : i + 20]) for i in range(0, len(lines), 20))
    return LoadedDocument(text, path.stem, "json", str(path), {})


def load_csv(path: Path, raw: str | None = None) -> LoadedDocument:
    raw = _read(path) if raw is None else raw
    dialect = "excel-tab" if path.suffix.lower() == ".tsv" else "excel"
    rows = list(csv.DictReader(io.StringIO(raw), dialect=dialect))
    lines = [f"Row {i + 1}: " + "; ".join(f"{k}={v}" for k, v in r.items() if k) + "." for i, r in enumerate(rows)]
    text = "\n\n".join("\n".join(lines[i : i + 25]) for i in range(0, len(lines), 25))
    return LoadedDocument(text, path.stem, "csv", str(path), {"rows": len(rows), "columns": list(rows[0]) if rows else []})


# EmotionLens behavior packets -------------------------------------------------
_TIME_KEYS = ("t", "time", "timestamp", "ts", "t_start", "start", "frame_time", "sec")
_WHO_KEYS = ("person", "person_id", "speaker", "subject", "track_id", "actor", "who", "character")
_LABEL_KEYS = ("emotion", "label", "predicted_emotion", "state", "affect", "behavior", "expression")
_CONF_KEYS = ("confidence", "score", "prob", "probability")
_CUE_KEYS = ("cues", "body_language", "gestures", "action_units", "aus", "features", "signals", "posture", "gaze")


def _first(d: dict, keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, "", []):
            return d[k]
    return None


def packet_to_observation(p: dict[str, Any]) -> str:
    """Render one behavior packet as a natural-language observation sentence."""
    t = _first(p, _TIME_KEYS)
    who = _first(p, _WHO_KEYS) or "the subject"
    label = _first(p, _LABEL_KEYS)
    conf = _first(p, _CONF_KEYS)
    cues = _first(p, _CUE_KEYS)
    parts = []
    when = f"At {float(t):.1f}s, " if isinstance(t, (int, float)) else (f"At {t}, " if t else "")
    if label is not None:
        if isinstance(label, dict):  # distribution
            top = sorted(label.items(), key=lambda kv: -float(kv[1]) if isinstance(kv[1], (int, float)) else 0)[:2]
            label_s = " / ".join(f"{k} ({float(v):.2f})" for k, v in top if isinstance(v, (int, float)))
        else:
            label_s = str(label)
        s = f"{when}{who} showed {label_s}"
        if isinstance(conf, (int, float)):
            s += f" (confidence {float(conf):.2f})"
        parts.append(s)
    else:
        parts.append(f"{when}{who} was observed")
    if cues:
        if isinstance(cues, dict):
            cue_s = ", ".join(f"{k}={v}" for k, v in list(cues.items())[:8])
        elif isinstance(cues, list):
            cue_s = ", ".join(str(c) for c in cues[:8])
        else:
            cue_s = str(cues)
        parts.append(f"cues: {cue_s}")
    known = set(_TIME_KEYS + _WHO_KEYS + _LABEL_KEYS + _CONF_KEYS + _CUE_KEYS)
    extra = {k: v for k, v in p.items() if k not in known and isinstance(v, (str, int, float, bool))}
    if extra:
        parts.append("; ".join(f"{k}={v}" for k, v in list(extra.items())[:6]))
    return "; ".join(parts) + "."


def load_packets(path: Path, raw: str | None = None) -> LoadedDocument:
    """EmotionLens ``packets.jsonl``: one JSON behavior packet per line -> observations.

    Field names are matched flexibly (time/t/timestamp, person/speaker, emotion/label,
    confidence/score, cues/action_units/...). Observations are grouped into paragraphs of
    10 so chunks cover short time windows.
    """
    raw = _read(path) if raw is None else raw
    obs = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            p = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(p, dict):
            obs.append(packet_to_observation(p))
    text = "\n\n".join("\n".join(obs[i : i + 10]) for i in range(0, len(obs), 10))
    return LoadedDocument(text, path.stem, "packets", str(path), {"packets": len(obs), "source_type": "observation"})


def _looks_like_packets(path: Path) -> bool:
    if "packet" in path.name.lower():
        return True
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            first = f.readline()
        p = json.loads(first)
        return isinstance(p, dict) and _first(p, _LABEL_KEYS) is not None and _first(p, _TIME_KEYS) is not None
    except (OSError, json.JSONDecodeError):
        return False


def load_path(path: str | Path) -> LoadedDocument:
    """Dispatch on file extension."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in MD_EXTS:
        return load_markdown(p)
    if ext in HTML_EXTS:
        return load_html(p)
    if ext == ".pdf":
        return load_pdf(p)
    if ext == ".json":
        return load_json(p)
    if ext in (".csv", ".tsv"):
        return load_csv(p)
    if ext == ".jsonl":
        if _looks_like_packets(p):
            return load_packets(p)
        raw = _read(p)
        lines = [json.dumps(json.loads(x)) for x in raw.splitlines() if x.strip()]
        return LoadedDocument("\n\n".join(lines), p.stem, "json", str(p), {})
    if ext in CODE_EXTS:
        return load_code(p)
    return load_text(p)


def load_url(url: str, *, timeout: float = 30.0, transport: httpx.BaseTransport | None = None) -> LoadedDocument:
    """Fetch a URL (HTML, plain text, JSON or PDF) and normalize it."""
    headers = {"user-agent": "noesis/0.1 (+https://github.com/asharan29-prog/chatterbox2)"}
    with httpx.Client(timeout=timeout, follow_redirects=True, transport=transport, headers=headers) as client:
        resp = client.get(url)
        resp.raise_for_status()
    ctype = resp.headers.get("content-type", "").lower()
    if "pdf" in ctype:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(resp.content)
            tmp = Path(f.name)
        try:
            doc = load_pdf(tmp)
        finally:
            tmp.unlink(missing_ok=True)
        doc.uri, doc.kind = url, "url"
        return doc
    if "json" in ctype:
        doc = load_json(Path(url.rstrip("/").split("/")[-1] or "data"), resp.text)
    elif "html" in ctype or resp.text.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
        title, text = html_to_text(resp.text)
        doc = LoadedDocument(_normalize(text), title or url, "url", url, {})
    else:
        doc = LoadedDocument(_normalize(resp.text), url, "url", url, {})
    doc.uri = url
    doc.kind = "url"
    doc.metadata["content_type"] = ctype
    return doc
