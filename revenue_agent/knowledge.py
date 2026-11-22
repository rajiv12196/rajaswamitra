"""Local knowledge base: the office's own Acts, rules, circulars and SOPs.

Drop .txt / .md / .pdf / .docx files into knowledge_base/<domain>/ (or the
root). Retrieval uses TF-IDF, which needs no model download and runs on any
office PC. Nothing leaves the machine.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_DIR = Path(__file__).resolve().parents[1] / "knowledge_base"
SUPPORTED = {".txt", ".md", ".pdf", ".docx"}


@dataclass
class Chunk:
    text: str
    source: str
    domain: str
    page: int | None = None


def _read_file(path: Path) -> list[tuple[str, int | None]]:
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return [(path.read_text(encoding="utf-8", errors="ignore"), None)]
    if ext == ".pdf":
        from pypdf import PdfReader
        return [(p.extract_text() or "", i + 1) for i, p in enumerate(PdfReader(str(path)).pages)]
    if ext == ".docx":
        import docx
        return [("\n".join(p.text for p in docx.Document(str(path)).paragraphs), None)]
    return []


def _split(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text)
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 1 <= size:
            cur = f"{cur}\n{p}" if cur else p
            continue
        if cur:
            chunks.append(cur)
        while len(p) > size:
            chunks.append(p[:size])
            p = p[size - overlap:]
        cur = p
    if cur:
        chunks.append(cur)
    return chunks


class KnowledgeBase:
    def __init__(self, kb_dir: Path = KB_DIR):
        self.kb_dir = Path(kb_dir)
        self.chunks: list[Chunk] = []
        self._vec = None
        self._matrix = None
        self.reload()

    def files(self) -> list[Path]:
        return sorted(p for p in self.kb_dir.rglob("*") if p.suffix.lower() in SUPPORTED and p.is_file())

    def reload(self) -> int:
        self.chunks = []
        for path in self.files():
            rel = path.relative_to(self.kb_dir)
            domain = rel.parts[0] if len(rel.parts) > 1 else "general"
            try:
                for text, page in _read_file(path):
                    for c in _split(text):
                        self.chunks.append(Chunk(c, str(rel), domain, page))
            except Exception as e:  # a bad file must not break the agent
                print(f"[knowledge] skipped {rel}: {e}")
        if self.chunks:
            self._vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
            self._matrix = self._vec.fit_transform([c.text for c in self.chunks])
        else:
            self._vec = self._matrix = None
        return len(self.chunks)

    def search(self, query: str, k: int = 4, domain: str | None = None, min_score: float = 0.05) -> list[dict]:
        if not self.chunks or not query.strip():
            return []
        scores = cosine_similarity(self._vec.transform([query]), self._matrix)[0]
        order = scores.argsort()[::-1]
        out = []
        for i in order:
            c = self.chunks[i]
            if scores[i] < min_score:
                break
            if domain and domain not in ("general", "other") and c.domain not in (domain, "general"):
                continue
            out.append({"source": c.source + (f" p.{c.page}" if c.page else ""), "domain": c.domain,
                        "score": round(float(scores[i]), 3), "text": c.text})
            if len(out) >= k:
                break
        return out
