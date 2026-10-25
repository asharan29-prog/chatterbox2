"""SQLite knowledge store: one ``.noesis.db`` file per topic/workspace.

Every model from :mod:`noesis.types` is stored as a JSON blob in its own table, with a few
indexed columns for fast filtering. Chunks and claims are mirrored into FTS5 tables for
keyword search; vectors live in a ``vectors`` table (float32 blobs) searched with numpy
cosine similarity.

Typical use::

    from noesis.store import KnowledgeStore
    with KnowledgeStore("coffee.noesis.db") as store:
        store.set_meta("topic", "Is coffee good for health?")
        src = store.add_source(Source(uri="notes.md", text="..."))
        hits = store.search_text("blood pressure", kind="chunk", k=5)

Thread-safety: one store per thread (sqlite3 connection is created with
``check_same_thread=False`` but writes are not locked; the MCP server should serialize).
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from .types import (
    Belief,
    BeliefChange,
    Chunk,
    Claim,
    Concept,
    Evidence,
    NoesisModel,
    Question,
    ReasoningRun,
    Source,
    Summary,
    Thought,
    utcnow,
)

__all__ = ["KnowledgeStore", "SCHEMA_VERSION"]

SCHEMA_VERSION = 1
M = TypeVar("M", bound=NoesisModel)

# model -> (table, {indexed column: attribute})
_TABLES: dict[type[NoesisModel], tuple[str, dict[str, str]]] = {
    Source: ("sources", {"content_hash": "content_hash", "uri": "uri"}),
    Chunk: ("chunks", {"source_id": "source_id", "idx": "index"}),
    Concept: ("concepts", {"name": "name", "parent_id": "parent_id"}),
    Claim: ("claims", {"origin": "origin"}),
    Evidence: ("evidence", {"claim_id": "claim_id", "chunk_id": "chunk_id", "source_id": "source_id", "stance": "stance"}),
    Belief: ("beliefs", {"claim_id": "claim_id", "status": "status", "confidence": "confidence"}),
    Question: ("questions", {"status": "status", "origin": "origin", "priority": "priority"}),
    Thought: ("thoughts", {"run_id": "run_id", "step": "step", "ts": "timestamp"}),
    ReasoningRun: ("runs", {"status": "status"}),
    Summary: ("summaries", {"level": "level", "scope": "scope"}),
}
_BY_TABLE = {t: cls for cls, (t, _) in _TABLES.items()}
_EXPORT_ORDER = [Source, Chunk, Concept, Claim, Evidence, Belief, Question, ReasoningRun, Thought, Summary]


class KnowledgeStore:
    """CRUD + search over a Noesis knowledge base.

    Generic API (works for every model type): :meth:`put`, :meth:`put_many`, :meth:`get`,
    :meth:`list`, :meth:`delete`, :meth:`count`. Typed convenience wrappers exist for the
    common operations (``add_source``, ``list_evidence(claim_id=...)``, ...).
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL" if self.path != ":memory:" else "PRAGMA journal_mode=MEMORY")
        self.conn.execute("PRAGMA foreign_keys=OFF")
        self._init_schema()

    # ------------------------------------------------------------------ lifecycle
    def _init_schema(self) -> None:
        c = self.conn
        c.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        for _cls, (table, cols) in _TABLES.items():
            col_sql = "".join(f", {col}" for col in cols)
            c.execute(f"CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY, data TEXT NOT NULL{col_sql}, seq INTEGER)")
            for col in cols:
                c.execute(f"CREATE INDEX IF NOT EXISTS ix_{table}_{col} ON {table}({col})")
        c.execute(
            "CREATE TABLE IF NOT EXISTS vectors (namespace TEXT, item_id TEXT, dim INTEGER, vec BLOB, "
            "PRIMARY KEY(namespace, item_id))"
        )
        c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(item_id UNINDEXED, kind UNINDEXED, text)")
        if self.get_meta("schema_version") is None:
            self.set_meta("schema_version", str(SCHEMA_VERSION))
            self.set_meta("created_at", utcnow().isoformat())
        c.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    def __enter__(self) -> KnowledgeStore:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def commit(self) -> None:
        self.conn.commit()

    # ------------------------------------------------------------------ meta
    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return row[0]

    def set_meta(self, key: str, value: Any) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, json.dumps(value, default=str)))
        self.conn.commit()

    @property
    def topic(self) -> str | None:
        return self.get_meta("topic")

    # ------------------------------------------------------------------ generic CRUD
    @staticmethod
    def _table(cls: type[NoesisModel]) -> tuple[str, dict[str, str]]:
        try:
            return _TABLES[cls]
        except KeyError as e:
            raise TypeError(f"{cls.__name__} is not a storable Noesis model") from e

    def put(self, obj: M, *, commit: bool = True) -> M:
        """Insert or replace ``obj`` (by id). Returns it."""
        table, cols = self._table(type(obj))
        values = [getattr(obj, attr) for attr in cols.values()]
        values = [v.isoformat() if hasattr(v, "isoformat") else v for v in values]
        col_names = ", ".join(["id", "data", *cols.keys(), "seq"])
        ph = ", ".join(["?"] * (len(cols) + 3))
        existing = self.conn.execute(f"SELECT seq FROM {table} WHERE id=?", (obj.id,)).fetchone()
        if existing is not None:
            seq = existing[0]
        else:
            seq = (self.conn.execute(f"SELECT COALESCE(MAX(seq), 0) + 1 FROM {table}").fetchone()[0])
        self.conn.execute(
            f"INSERT OR REPLACE INTO {table} ({col_names}) VALUES ({ph})",
            [obj.id, obj.model_dump_json(), *values, seq],
        )
        if isinstance(obj, Chunk):
            self._fts_upsert(obj.id, "chunk", obj.text)
        elif isinstance(obj, Claim):
            self._fts_upsert(obj.id, "claim", " ".join([obj.text, *obj.variants]))
        if commit:
            self.conn.commit()
        return obj

    def put_many(self, objs: Iterable[M]) -> list[M]:
        out = [self.put(o, commit=False) for o in objs]
        self.conn.commit()
        return out

    def get(self, cls: type[M], id: str) -> M | None:
        table, _ = self._table(cls)
        row = self.conn.execute(f"SELECT data FROM {table} WHERE id=?", (id,)).fetchone()
        return cls.model_validate_json(row[0]) if row else None

    def require(self, cls: type[M], id: str) -> M:
        obj = self.get(cls, id)
        if obj is None:
            raise KeyError(f"{cls.__name__} {id!r} not found")
        return obj

    def list(
        self,
        cls: type[M],
        *,
        limit: int | None = None,
        offset: int = 0,
        order_by: str | None = None,
        desc: bool = False,
        **filters: Any,
    ) -> list[M]:
        """List objects filtered by indexed columns (``None`` filter values are ignored).

        A filter value may be a list/tuple for ``IN`` matching. ``order_by`` must be an indexed
        column name; default is insertion order.
        """
        table, cols = self._table(cls)
        where, params = [], []
        attr_to_col = {a: c for c, a in cols.items()}
        for key, val in filters.items():
            if val is None:
                continue
            col = attr_to_col.get(key, key)
            if col not in cols:
                raise ValueError(f"cannot filter {cls.__name__} by {key!r}; indexed: {sorted(cols.values())}")
            if isinstance(val, (list, tuple, set)):
                val = list(val)
                if not val:
                    return []
                where.append(f"{col} IN ({', '.join('?' * len(val))})")
                params += val
            else:
                where.append(f"{col} = ?")
                params.append(val)
        sql = f"SELECT data FROM {table}"
        if where:
            sql += " WHERE " + " AND ".join(where)
        ocol = "seq"
        if order_by:
            ocol = attr_to_col.get(order_by, order_by)
            if ocol not in cols and ocol != "seq":
                raise ValueError(f"cannot order by {order_by!r}")
        sql += f" ORDER BY {ocol} {'DESC' if desc else 'ASC'}"
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params += [limit, offset]
        return [cls.model_validate_json(r[0]) for r in self.conn.execute(sql, params)]

    def count(self, cls: type[NoesisModel], **filters: Any) -> int:
        table, cols = self._table(cls)
        attr_to_col = {a: c for c, a in cols.items()}
        where, params = [], []
        for k, v in filters.items():
            if v is not None:
                where.append(f"{attr_to_col.get(k, k)} = ?")
                params.append(v)
        sql = f"SELECT COUNT(*) FROM {table}" + (" WHERE " + " AND ".join(where) if where else "")
        return int(self.conn.execute(sql, params).fetchone()[0])

    def delete(self, cls: type[NoesisModel], id: str) -> bool:
        table, _ = self._table(cls)
        cur = self.conn.execute(f"DELETE FROM {table} WHERE id=?", (id,))
        self.conn.execute("DELETE FROM fts WHERE item_id=?", (id,))
        self.conn.execute("DELETE FROM vectors WHERE item_id=?", (id,))
        self.conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------ typed helpers
    # Sources
    def add_source(self, source: Source) -> Source:
        return self.put(source)

    def get_source(self, id: str) -> Source | None:
        return self.get(Source, id)

    def list_sources(self) -> list[Source]:
        return self.list(Source)

    def find_source_by_hash(self, content_hash: str) -> Source | None:
        res = self.list(Source, content_hash=content_hash, limit=1)
        return res[0] if res else None

    def set_source_reliability(self, source_id: str, reliability: float, reason: str = "user override") -> Source:
        src = self.require(Source, source_id)
        src.reliability = float(reliability)
        src.reliability_reason = reason
        return self.put(src)

    # Chunks
    def add_chunks(self, chunks: Sequence[Chunk]) -> list[Chunk]:
        return self.put_many(chunks)

    def get_chunk(self, id: str) -> Chunk | None:
        return self.get(Chunk, id)

    def list_chunks(self, source_id: str | None = None) -> list[Chunk]:
        return self.list(Chunk, source_id=source_id, order_by="seq")

    # Concepts
    def add_concept(self, concept: Concept) -> Concept:
        return self.put(concept)

    def list_concepts(self, parent_id: str | None = None) -> list[Concept]:
        return self.list(Concept, parent_id=parent_id)

    def find_concept(self, name: str) -> Concept | None:
        """Case-insensitive lookup by name or alias."""
        low = name.strip().lower()
        row = self.conn.execute("SELECT data FROM concepts WHERE lower(name)=? LIMIT 1", (low,)).fetchone()
        if row:
            return Concept.model_validate_json(row[0])
        for c in self.list(Concept):
            if low in (a.lower() for a in c.aliases):
                return c
        return None

    # Claims
    def add_claim(self, claim: Claim) -> Claim:
        return self.put(claim)

    def get_claim(self, id: str) -> Claim | None:
        return self.get(Claim, id)

    def list_claims(self, origin: str | None = None) -> list[Claim]:
        return self.list(Claim, origin=origin)

    # Evidence
    def add_evidence(self, evidence: Evidence) -> Evidence:
        return self.put(evidence)

    def list_evidence(
        self,
        claim_id: str | Sequence[str] | None = None,
        *,
        chunk_id: str | None = None,
        source_id: str | None = None,
        stance: str | None = None,
    ) -> list[Evidence]:
        return self.list(Evidence, claim_id=claim_id, chunk_id=chunk_id, source_id=source_id, stance=stance)

    # Beliefs
    def get_belief(self, id: str) -> Belief | None:
        return self.get(Belief, id)

    def get_belief_for_claim(self, claim_id: str) -> Belief | None:
        res = self.list(Belief, claim_id=claim_id, limit=1)
        return res[0] if res else None

    def save_belief(self, belief: Belief) -> Belief:
        belief.updated_at = utcnow()
        return self.put(belief)

    def list_beliefs(self, status: str | None = None, *, order_by_confidence: bool = False) -> list[Belief]:
        return self.list(Belief, status=status, order_by="confidence" if order_by_confidence else None, desc=order_by_confidence)

    def belief_history(self, belief_or_claim_id: str) -> list[BeliefChange]:
        b = self.get_belief(belief_or_claim_id) or self.get_belief_for_claim(belief_or_claim_id)
        return list(b.history) if b else []

    # Questions
    def add_question(self, question: Question) -> Question:
        return self.put(question)

    def update_question(self, question: Question) -> Question:
        return self.put(question)

    def list_questions(self, status: str | None = None, origin: str | None = None) -> list[Question]:
        return self.list(Question, status=status, origin=origin, order_by="priority", desc=True)

    def find_question(self, text: str) -> Question | None:
        norm = re.sub(r"\W+", " ", text.lower()).strip()
        for q in self.list(Question):
            if re.sub(r"\W+", " ", q.text.lower()).strip() == norm:
                return q
        return None

    # Thoughts / runs / summaries
    def add_thought(self, thought: Thought) -> Thought:
        return self.put(thought)

    def list_thoughts(self, run_id: str | None = None) -> list[Thought]:
        return self.list(Thought, run_id=run_id)

    def add_run(self, run: ReasoningRun) -> ReasoningRun:
        return self.put(run)

    def update_run(self, run: ReasoningRun) -> ReasoningRun:
        return self.put(run)

    def get_run(self, id: str) -> ReasoningRun | None:
        return self.get(ReasoningRun, id)

    def list_runs(self) -> list[ReasoningRun]:
        return self.list(ReasoningRun)

    def add_summary(self, summary: Summary) -> Summary:
        return self.put(summary)

    def list_summaries(self, level: int | None = None, scope: str | None = None) -> list[Summary]:
        return self.list(Summary, level=level, scope=scope)

    def topic_summary(self) -> Summary | None:
        res = self.list(Summary, scope="topic", order_by="seq", desc=True, limit=1)
        return res[0] if res else None

    # ------------------------------------------------------------------ full-text search
    def _fts_upsert(self, item_id: str, kind: str, text: str) -> None:
        self.conn.execute("DELETE FROM fts WHERE item_id=?", (item_id,))
        self.conn.execute("INSERT INTO fts(item_id, kind, text) VALUES (?, ?, ?)", (item_id, kind, text))

    @staticmethod
    def _fts_query(q: str, mode: str = "or") -> str:
        terms = [t for t in re.findall(r"[A-Za-z0-9]+", q) if len(t) > 1]
        from .nlp import STOPWORDS

        terms = [t for t in terms if t.lower() not in STOPWORDS] or terms
        if not terms:
            return ""
        return f" {mode.upper()} ".join(f'"{t}"*' if len(t) > 3 else f'"{t}"' for t in terms)

    def search_text(self, query: str, *, kind: str = "chunk", k: int = 10) -> list[tuple[str, float]]:
        """FTS5 BM25 keyword search. ``kind`` is ``"chunk"`` or ``"claim"``.

        Returns ``[(item_id, score)]`` with higher score = better. Query syntax is sanitized
        (terms OR-ed, prefix-matched), so arbitrary user text is safe.
        """
        fq = self._fts_query(query)
        if not fq:
            return []
        rows = self.conn.execute(
            "SELECT item_id, bm25(fts) AS s FROM fts WHERE fts MATCH ? AND kind = ? ORDER BY s LIMIT ?",
            (fq, kind, k),
        ).fetchall()
        return [(r[0], float(-r[1])) for r in rows]

    # ------------------------------------------------------------------ vectors
    def set_vectors(self, namespace: str, ids: Sequence[str], vectors: np.ndarray) -> None:
        """Store vectors under ``namespace`` (e.g. ``"chunk:hashing-512"``)."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors length mismatch")
        self.conn.executemany(
            "INSERT OR REPLACE INTO vectors(namespace, item_id, dim, vec) VALUES (?, ?, ?, ?)",
            [(namespace, i, int(v.shape[0]), v.tobytes()) for i, v in zip(ids, vectors)],
        )
        self.conn.commit()

    def get_vectors(self, namespace: str, ids: Sequence[str] | None = None) -> tuple[list[str], np.ndarray]:
        if ids is not None:
            ids = list(ids)
            if not ids:
                return [], np.zeros((0, 0), dtype=np.float32)
            q = f"SELECT item_id, dim, vec FROM vectors WHERE namespace=? AND item_id IN ({','.join('?' * len(ids))})"
            rows = self.conn.execute(q, [namespace, *ids]).fetchall()
        else:
            rows = self.conn.execute("SELECT item_id, dim, vec FROM vectors WHERE namespace=?", (namespace,)).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        out_ids = [r[0] for r in rows]
        mat = np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
        return out_ids, mat

    def has_vector(self, namespace: str, item_id: str) -> bool:
        return (
            self.conn.execute("SELECT 1 FROM vectors WHERE namespace=? AND item_id=?", (namespace, item_id)).fetchone()
            is not None
        )

    def vector_search(
        self, namespace: str, query: np.ndarray, k: int = 10, *, restrict_to: Sequence[str] | None = None
    ) -> list[tuple[str, float]]:
        """Cosine-similarity search. Returns ``[(item_id, similarity)]`` best first."""
        ids, mat = self.get_vectors(namespace, restrict_to)
        if not ids:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != mat.shape[1]:
            raise ValueError(f"query dim {q.shape[0]} != stored dim {mat.shape[1]} in namespace {namespace!r}")
        qn = q / max(float(np.linalg.norm(q)), 1e-12)
        norms = np.clip(np.linalg.norm(mat, axis=1), 1e-12, None)
        sims = (mat @ qn) / norms
        order = np.argsort(-sims)[:k]
        return [(ids[i], float(sims[i])) for i in order]

    # ------------------------------------------------------------------ export / import
    def export_json(self) -> dict[str, Any]:
        """Everything (except vectors, which are recomputable) as a JSON-serializable dict."""
        out: dict[str, Any] = {
            "noesis_export": SCHEMA_VERSION,
            "meta": {r[0]: self.get_meta(r[0]) for r in self.conn.execute("SELECT key FROM meta")},
        }
        for cls in _EXPORT_ORDER:
            table, _ = _TABLES[cls]
            out[table] = [json.loads(r[0]) for r in self.conn.execute(f"SELECT data FROM {table} ORDER BY seq")]
        return out

    def export_to(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(json.dumps(self.export_json(), indent=1, default=str), encoding="utf-8")
        return p

    def import_json(self, data: dict[str, Any] | str | Path, *, overwrite: bool = True) -> dict[str, int]:
        """Load an export produced by :meth:`export_json` (dict, JSON string, or file path).

        Returns counts per table. Existing ids are replaced when ``overwrite`` else skipped.
        """
        if isinstance(data, Path) or (isinstance(data, str) and not data.lstrip().startswith("{")):
            data = json.loads(Path(data).read_text(encoding="utf-8"))
        elif isinstance(data, str):
            data = json.loads(data)
        assert isinstance(data, dict)
        counts: dict[str, int] = {}
        for key, value in (data.get("meta") or {}).items():
            if key not in ("schema_version", "created_at") or self.get_meta(key) is None:
                self.set_meta(key, value)
        for cls in _EXPORT_ORDER:
            table, _ = _TABLES[cls]
            n = 0
            for raw in data.get(table, []):
                obj = cls.model_validate(raw)
                if not overwrite and self.get(cls, obj.id) is not None:
                    continue
                self.put(obj, commit=False)
                n += 1
            counts[table] = n
        self.conn.commit()
        return counts

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict[str, int]:
        return {table: self.count(cls) for cls, (table, _) in _TABLES.items()}
