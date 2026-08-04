#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "mcp[cli]>=1.0,<2",
#   "pyyaml>=6.0",
#   "click>=8.1",
#   "watchdog>=4.0",
#   "sqlite-vec>=0.1.6",
#   "fastembed>=0.4",
# ]
# ///
"""vault-search: SQLite FTS5 index over ~/vault. CLI + MCP server in one file."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import click
import sqlite_vec
import yaml
from mcp.server.fastmcp import FastMCP

VAULT = Path(os.environ.get("VAULT_PATH", Path.home() / "vault")).resolve()
DB = Path(os.environ.get("VSEARCH_DB", VAULT / ".vault-search.db"))
SKIP_DIRS = {".git", ".obsidian", ".trash", "node_modules", ".beads"}
BOILERPLATE_SECTIONS = {"see also", "references", "links", "related", "backlinks"}

# Embeddings run in-process via fastembed (ONNX, CPU) — no Ollama, no network.
EMBED_MODEL = os.environ.get("VSEARCH_EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")
EMBED_DIM = 768
RRF_K = 60
MAX_CHUNK_CHARS = 8000  # matches embed input cap (see embed_missing) — keeps each chunk vector-representable


SCHEMA_VERSION = 4


def get_db() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(f"""
    CREATE TABLE IF NOT EXISTS _meta(k TEXT PRIMARY KEY, v TEXT);
    CREATE TABLE IF NOT EXISTS atoms(
      path TEXT PRIMARY KEY,
      mtime REAL NOT NULL,
      frontmatter TEXT,
      title TEXT,
      body TEXT
    );
    CREATE TABLE IF NOT EXISTS atom_tags(
      path TEXT NOT NULL,
      tag TEXT NOT NULL,
      PRIMARY KEY (path, tag)
    );
    CREATE INDEX IF NOT EXISTS idx_atom_tags_tag ON atom_tags(tag);
    CREATE VIRTUAL TABLE IF NOT EXISTS sections_fts USING fts5(
      path UNINDEXED,
      section_title,
      tags,
      body,
      tokenize='porter unicode61'
    );
    CREATE VIRTUAL TABLE IF NOT EXISTS section_vecs USING vec0(
      rowid INTEGER PRIMARY KEY,
      embedding FLOAT[{EMBED_DIM}]
    );
    """)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    row = conn.execute("SELECT v FROM _meta WHERE k='schema_version'").fetchone()
    current = int(row[0]) if row else 0
    if current >= SCHEMA_VERSION:
        return
    if current < 2:
        has_old = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='atoms_fts'"
        ).fetchone()
        if has_old:
            print("migrating v1 -> v2: per-H2 section chunking", file=sys.stderr)
            n = 0
            for path, fm_json, title, body in conn.execute(
                "SELECT path, frontmatter, title, body FROM atoms"
            ).fetchall():
                try:
                    fm = json.loads(fm_json or "{}")
                except json.JSONDecodeError:
                    fm = {}
                tags = _flatten_tags(fm.get("tags", []))
                for sect_title, sect_body in chunk_sections(title or "", body or ""):
                    conn.execute(
                        "INSERT INTO sections_fts(path, section_title, tags, body) VALUES(?,?,?,?)",
                        (path, sect_title, tags, sect_body),
                    )
                    n += 1
            conn.execute("DROP TABLE atoms_fts")
            print(f"migrated: {n} sections from {len(list(conn.execute('SELECT 1 FROM atoms')))} atoms", file=sys.stderr)
    if current < 3:
        print("migrating to v3: atom_tags table", file=sys.stderr)
        n = 0
        for path, fm_json in conn.execute("SELECT path, frontmatter FROM atoms").fetchall():
            try:
                fm = json.loads(fm_json or "{}")
            except json.JSONDecodeError:
                fm = {}
            for tag in _tag_list(fm.get("tags", [])):
                conn.execute(
                    "INSERT OR IGNORE INTO atom_tags(path, tag) VALUES(?, ?)",
                    (path, tag),
                )
                n += 1
        print(f"migrated: {n} tag rows", file=sys.stderr)
    if current < 4:
        print("migrating to v4: section_vecs added (run 'vsearch embed' to populate)", file=sys.stderr)
    conn.execute(
        "INSERT INTO _meta(k,v) VALUES('schema_version',?) "
        "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()


def _flatten_tags(tags_field) -> str:
    return " ".join(_tag_list(tags_field))


def _tag_list(tags_field) -> list[str]:
    """Normalise a frontmatter tags field to a list[str]. Handles str | list | None."""
    if tags_field is None:
        return []
    if isinstance(tags_field, str):
        return [t.strip() for t in tags_field.split() if t.strip()]
    if isinstance(tags_field, list):
        return [str(t).strip() for t in tags_field if str(t).strip()]
    return []


def _sub_split(text: str, max_chars: int) -> list[str]:
    """Split a too-large chunk at paragraph (\\n\\n) boundaries, falling back to line
    boundaries, then hard mid-line cuts. Each output piece <= max_chars.
    """
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    cur = ""
    for para in text.split("\n\n"):
        candidate = f"{cur}\n\n{para}" if cur else para
        if len(candidate) <= max_chars:
            cur = candidate
            continue
        if cur:
            chunks.append(cur)
            cur = ""
        if len(para) <= max_chars:
            cur = para
            continue
        # paragraph alone too big — fall to lines
        buf = ""
        for ln in para.splitlines(keepends=True):
            if len(buf) + len(ln) <= max_chars:
                buf += ln
                continue
            if buf:
                chunks.append(buf)
                buf = ""
            if len(ln) <= max_chars:
                buf = ln
            else:
                # hard split mid-line
                for i in range(0, len(ln), max_chars):
                    chunks.append(ln[i:i + max_chars])
        if buf:
            cur = buf
    if cur:
        chunks.append(cur)
    return chunks


def chunk_sections(title: str, body: str) -> list[tuple[str, str]]:
    """Split body on '## ' boundaries. First chunk inherits H1 title. Sections
    longer than MAX_CHUNK_CHARS are sub-split at paragraph/line boundaries; the
    sub-pieces share the parent section title with a '(part i/n)' suffix.

    Returns [(section_title, section_body), ...]. Skips empty sections.
    """
    lines = body.splitlines()
    sections: list[tuple[str, list[str]]] = []
    cur_title = title
    cur_lines: list[str] = []
    for line in lines:
        if line.startswith("## "):
            if cur_lines:
                sections.append((cur_title, cur_lines))
            cur_title = line[3:].strip()
            cur_lines = [line]
        else:
            cur_lines.append(line)
    if cur_lines:
        sections.append((cur_title, cur_lines))
    out: list[tuple[str, str]] = []
    for st, ls in sections:
        if st.strip().lower() in BOILERPLATE_SECTIONS:
            continue
        text = "\n".join(ls).strip()
        if not text:
            continue
        parts = _sub_split(text, MAX_CHUNK_CHARS)
        if len(parts) == 1:
            out.append((st, parts[0]))
        else:
            for i, part in enumerate(parts, start=1):
                out.append((f"{st} (part {i}/{len(parts)})", part))
    return out


FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


def parse_atom(path: Path) -> tuple[dict, str, str, str]:
    text = path.read_text(errors="replace")
    m = FRONTMATTER_RE.match(text)
    if m:
        try:
            fm = yaml.safe_load(m.group(1)) or {}
        except yaml.YAMLError:
            fm = {}
        body = m.group(2)
    else:
        fm, body = {}, text
    if not isinstance(fm, dict):
        fm = {}
    title = ""
    for line in body.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break
    tags = _flatten_tags(fm.get("tags", []))
    return fm, title, tags, body


def should_skip(rel: Path) -> bool:
    return any(part.startswith(".") or part in SKIP_DIRS for part in rel.parts[:-1])


def index_one(conn: sqlite3.Connection, abs_path: Path, force: bool = False) -> str:
    """Upsert one file. Returns 'new' | 'updated' | 'skipped' | 'invalid'.
    force=True bypasses the mtime-based skip and reprocesses the file.
    """
    if abs_path.suffix != ".md" or not abs_path.is_file():
        return "invalid"
    try:
        rel_path = abs_path.relative_to(VAULT)
    except ValueError:
        return "invalid"
    if should_skip(rel_path):
        return "invalid"
    rel = str(rel_path)
    st = abs_path.stat()
    row = conn.execute("SELECT mtime FROM atoms WHERE path=?", (rel,)).fetchone()
    if row and not force and row[0] >= st.st_mtime:
        return "skipped"
    fm, title, tags, body = parse_atom(abs_path)
    fm_json = json.dumps(fm, default=str, ensure_ascii=False)
    if row:
        conn.execute(
            "UPDATE atoms SET mtime=?, frontmatter=?, title=?, body=? WHERE path=?",
            (st.st_mtime, fm_json, title, body, rel),
        )
        verb = "updated"
    else:
        conn.execute(
            "INSERT INTO atoms VALUES(?,?,?,?,?)",
            (rel, st.st_mtime, fm_json, title, body),
        )
        verb = "new"
    old_rowids = [
        r[0] for r in conn.execute(
            "SELECT rowid FROM sections_fts WHERE path=?", (rel,)
        ).fetchall()
    ]
    for old_rowid in old_rowids:
        conn.execute("DELETE FROM section_vecs WHERE rowid=?", (old_rowid,))
    conn.execute("DELETE FROM sections_fts WHERE path=?", (rel,))
    for sect_title, sect_body in chunk_sections(title, body):
        conn.execute(
            "INSERT INTO sections_fts(path, section_title, tags, body) VALUES(?,?,?,?)",
            (rel, sect_title, tags, sect_body),
        )
    conn.execute("DELETE FROM atom_tags WHERE path=?", (rel,))
    for tag in _tag_list(fm.get("tags", [])):
        conn.execute(
            "INSERT OR IGNORE INTO atom_tags(path, tag) VALUES(?, ?)",
            (rel, tag),
        )
    return verb


def delete_one(conn: sqlite3.Connection, rel: str) -> None:
    old_rowids = [
        r[0] for r in conn.execute(
            "SELECT rowid FROM sections_fts WHERE path=?", (rel,)
        ).fetchall()
    ]
    for old_rowid in old_rowids:
        conn.execute("DELETE FROM section_vecs WHERE rowid=?", (old_rowid,))
    conn.execute("DELETE FROM atoms WHERE path=?", (rel,))
    conn.execute("DELETE FROM sections_fts WHERE path=?", (rel,))
    conn.execute("DELETE FROM atom_tags WHERE path=?", (rel,))


def index_vault(verbose: bool = False, force: bool = False, embed: bool = True) -> dict:
    conn = get_db()
    seen: set[str] = set()
    n_new = n_upd = 0
    for f in VAULT.rglob("*.md"):
        rel_path = f.relative_to(VAULT)
        if should_skip(rel_path):
            continue
        seen.add(str(rel_path))
        verb = index_one(conn, f, force=force)
        if verb == "new":
            n_new += 1
        elif verb == "updated":
            n_upd += 1
        if verbose and verb in ("new", "updated"):
            print(f"  + {rel_path}", file=sys.stderr)
    deleted = [
        r[0]
        for r in conn.execute("SELECT path FROM atoms").fetchall()
        if r[0] not in seen
    ]
    for p in deleted:
        delete_one(conn, p)
    conn.commit()
    conn.close()
    stats = {"new": n_new, "updated": n_upd, "removed": len(deleted), "total": len(seen)}
    print(
        f"indexed: {n_new} new, {n_upd} updated, {len(deleted)} removed; total {len(seen)}",
        file=sys.stderr,
    )
    # Re-embed changed sections. index_one drops the vectors of updated sections
    # (new rowids), so without this every reindex silently leaves them vec-invisible
    # until the next manual `embed`, and semantic recall rots as the vault is edited.
    if embed and (n_new + n_upd) > 0:
        emb = embed_missing(verbose=verbose)
        stats["embedded"] = emb.get("embedded", 0)
        stats["embed_failed"] = emb.get("failed", 0)
    return stats


def _pack_embedding(emb: list[float]) -> bytes:
    return struct.pack(f"{len(emb)}f", *emb)


_EMBEDDER = None


def _get_embedder():
    """Lazy-load the fastembed model. Downloads ONNX weights to cache on first call."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from fastembed import TextEmbedding

        _EMBEDDER = TextEmbedding(model_name=EMBED_MODEL)
    return _EMBEDDER


# Remote embed backend: offload to ollama on the LAN box (.161 has GPU/headroom;
# the Mac is 16GB and SIGKILLs local fastembed under load). Unset to fall back to
# local fastembed. nomic via ollama does NOT auto-add the task prefix, so we add it
# manually — the same model family as fastembed nomic-embed-text-v1.5, 768-dim.
OLLAMA_URL = os.environ.get("VSEARCH_OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.environ.get("VSEARCH_OLLAMA_EMBED_MODEL", "nomic-embed-text")


def _embed_ollama(text: str, is_query: bool) -> list[float] | None:
    import urllib.request
    prefix = "search_query: " if is_query else "search_document: "
    # nomic-embed-text on ollama has a 2048-token context and returns HTTP 500
    # (not a silent empty) when the input exceeds it. Dense content (tables, path
    # lists) packs ~3 chars/token, so 6000 chars can exceed 2048 tokens while
    # prose of the same length fits. Retry at shorter caps so dense sections
    # still embed (at reduced coverage) instead of being dropped from vec search.
    for cap in (6000, 4000, 2500):
        payload = json.dumps({"model": OLLAMA_EMBED_MODEL,
                              "prompt": prefix + text[:cap]}).encode()
        req = urllib.request.Request(
            OLLAMA_URL.rstrip("/") + "/api/embeddings",
            data=payload, headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                emb = json.load(r).get("embedding")
            if emb and len(emb) == EMBED_DIM:
                return [float(x) for x in emb]
        except Exception:
            continue
    return None


def _embed(text: str, is_query: bool = False) -> list[float] | None:
    """Embed text via ollama (VSEARCH_OLLAMA_URL) or, if unset, local fastembed.

    nomic models distinguish query from passage: is_query=True applies the
    search-query prefix, False the search-document prefix. The two must match
    the convention used when section_vecs was populated. Returns the vector,
    or None on any failure.
    """
    if OLLAMA_URL:
        return _embed_ollama(text, is_query)
    try:
        model = _get_embedder()
        gen = model.query_embed(text) if is_query else model.embed([text])
        vec = next(iter(gen))
        emb = [float(x) for x in vec]
        if len(emb) == EMBED_DIM:
            return emb
    except Exception:
        return None
    return None


EMBED_BATCH = int(os.environ.get("VSEARCH_EMBED_BATCH", "256"))


def embed_missing(verbose: bool = False, batch_status_every: int = 25) -> dict:
    """Backfill embeddings for sections that don't yet have one. Idempotent.

    Embeds in batches: fastembed's model.embed(list) amortises per-call overhead,
    which is ~100x faster than one text at a time on CPU."""
    conn = get_db()
    missing = conn.execute(
        """
        SELECT s.rowid, s.section_title, s.body FROM sections_fts s
        WHERE s.rowid NOT IN (SELECT rowid FROM section_vecs)
          AND length(trim(s.body)) > 0
        """
    ).fetchall()
    if not missing:
        print("no sections need embedding", file=sys.stderr)
        conn.close()
        return {"embedded": 0, "failed": 0, "total_missing": 0}
    # Route through _embed so docs land in the SAME space as queries (ollama if
    # configured, else fastembed). Using _get_embedder()/model.embed() directly
    # would write fastembed-space vectors that mismatch ollama query vectors.
    import concurrent.futures as _cf
    embedded = failed = 0
    total = len(missing)

    def _do(row):
        rid, title, body = row
        return rid, _embed(f"{title}\n\n{body}".strip()[:8000], is_query=False)

    workers = 10 if OLLAMA_URL else 1  # ollama is remote+concurrent; fastembed is local single
    with _cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (rowid, emb) in enumerate(ex.map(_do, missing), start=1):
            if emb and len(emb) == EMBED_DIM:
                conn.execute(
                    "INSERT OR REPLACE INTO section_vecs(rowid, embedding) VALUES(?, ?)",
                    (rowid, _pack_embedding(emb)),
                )
                embedded += 1
            else:
                failed += 1
            if i % 200 == 0:
                conn.commit()
                if verbose:
                    print(f"  embedded {embedded}/{total}", file=sys.stderr)
    conn.commit()
    conn.close()
    print(f"embeddings: {embedded} new, {failed} failed, {total} total missing", file=sys.stderr)
    return {"embedded": embedded, "failed": failed, "total_missing": total}


def fts_query(q: str) -> str:
    """Injection-safe FTS5 query builder. Word tokens with prefix-match, OR-joined.

    OR (not implicit AND) so a verbose transcript that contains *some* of the query
    words still matches; BM25 ranking floats the rows matching more/rarer terms to
    the top. Strict AND was missing fuzzy 'find me that video where someone said X'
    recall, because no single section held every query word."""
    words = re.findall(r"\w+", q.lower())
    return " OR ".join(f'"{w}"*' for w in words)


def _fts_rows(conn, query: str, tags: list[str] | None, limit: int) -> list[tuple]:
    fts_q = fts_query(query)
    if not fts_q:
        return []
    where = ["sections_fts MATCH ?"]
    params: list = [fts_q]
    for t in tags or []:
        where.append(
            "path IN (SELECT path FROM atom_tags WHERE tag = ? OR tag LIKE ?)"
        )
        params.extend([t, t + "/%"])
    params.append(limit)
    sql = f"""
        SELECT rowid, path, section_title,
               snippet(sections_fts, 3, '<<', '>>', '...', 16) AS body_snip,
               bm25(sections_fts) AS score
        FROM sections_fts
        WHERE {" AND ".join(where)}
        ORDER BY bm25(sections_fts)
        LIMIT ?
    """
    return conn.execute(sql, params).fetchall()


def _vec_rows(conn, query: str, tags: list[str] | None, limit: int) -> list[tuple] | None:
    emb = _embed(query, is_query=True)
    if emb is None:
        return None  # embedder unavailable — distinct from a genuine empty result ([])
    if tags:
        allowed = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT path FROM atom_tags WHERE "
                + " OR ".join(["tag = ? OR tag LIKE ?"] * len(tags)),
                [v for t in tags for v in (t, t + "/%")],
            ).fetchall()
        }
        if not allowed:
            return []
    else:
        allowed = None
    # When tag-scoped, the global nearest neighbours are dominated by out-of-scope
    # atoms, so a small KNN k post-filters to ~nothing. Over-fetch a deep pool (the
    # whole vector set is only a few thousand rows) and filter down to `limit`.
    if allowed is not None:
        # sqlite-vec caps KNN k at 4096; fetch that deep pool, then filter to scope.
        total = conn.execute("SELECT count(*) FROM section_vecs").fetchone()[0]
        kfetch = min(4096, total)
    else:
        kfetch = limit
    rows = conn.execute(
        """
        SELECT v.rowid, s.path, s.section_title, s.body, v.distance
        FROM section_vecs v
        JOIN sections_fts s ON s.rowid = v.rowid
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance
        """,
        (_pack_embedding(emb), kfetch),
    ).fetchall()
    if allowed is not None:
        rows = [r for r in rows if r[1] in allowed]
    return rows[:limit]


def _snippet_from_body(body: str, query: str, span: int = 160) -> str:
    """Cheap snippet around the first matching word, for vec results."""
    words = re.findall(r"\w+", query.lower())
    lower = body.lower()
    pos = -1
    for w in words:
        i = lower.find(w)
        if i >= 0 and (pos < 0 or i < pos):
            pos = i
    if pos < 0:
        return body[:span].strip() + ("..." if len(body) > span else "")
    start = max(0, pos - span // 2)
    end = min(len(body), start + span)
    out = body[start:end].strip()
    return ("..." if start > 0 else "") + out + ("..." if end < len(body) else "")


def search(
    query: str,
    k: int = 5,
    tags: list[str] | None = None,
    mode: str = "hybrid",
) -> list[dict]:
    """Modes:
      - 'fts': BM25 over sections_fts only.
      - 'vec': cosine over section_vecs only (in-process fastembed embeddings).
      - 'hybrid': both, fused via RRF.

    If the embedder is unavailable, hybrid degrades to FTS-only. The degradation
    is NOT silent: degraded results carry via='fts-fallback' (instead of 'fts'),
    and a warning is written to stderr. A genuine empty vec result (embedder fine,
    no semantic matches) is distinct and does not trigger the marker.
    """
    if mode not in ("hybrid", "fts", "vec"):
        raise ValueError(f"unknown mode: {mode}")
    requested_mode = mode
    conn = get_db()
    try:
        fetch = max(k * 3, 20)
        fts = _fts_rows(conn, query, tags, fetch) if mode in ("hybrid", "fts") else []
        vec = _vec_rows(conn, query, tags, fetch) if mode in ("hybrid", "vec") else []
        vec_degraded = vec is None  # _vec_rows returns None only when the embedder is unavailable
        if vec is None:
            vec = []
        if vec_degraded and requested_mode in ("hybrid", "vec"):
            print(
                "vault-search: embedder unavailable — semantic search degraded; "
                "results are FTS-only (via='fts-fallback')",
                file=sys.stderr,
            )
        fts_via = "fts-fallback" if (requested_mode == "hybrid" and vec_degraded) else "fts"
        if mode == "hybrid" and not vec and fts:
            mode = "fts"
        if mode == "fts":
            return [
                {
                    "path": r[1],
                    "section": r[2],
                    "snippet": r[3],
                    "score": round(r[4], 3),
                    "via": fts_via,
                }
                for r in fts[:k]
            ]
        if mode == "vec":
            return [
                {
                    "path": r[1],
                    "section": r[2],
                    "snippet": _snippet_from_body(r[3], query),
                    "distance": round(r[4], 4),
                    "via": "vec",
                }
                for r in vec[:k]
            ]
        scores: dict[int, float] = {}
        details: dict[int, dict] = {}
        for rank, r in enumerate(fts, start=1):
            rowid = r[0]
            scores[rowid] = scores.get(rowid, 0.0) + 1.0 / (RRF_K + rank)
            details[rowid] = {
                "path": r[1],
                "section": r[2],
                "snippet": r[3],
                "via": "fts",
            }
        for rank, r in enumerate(vec, start=1):
            rowid = r[0]
            scores[rowid] = scores.get(rowid, 0.0) + 1.0 / (RRF_K + rank)
            if rowid not in details:
                details[rowid] = {
                    "path": r[1],
                    "section": r[2],
                    "snippet": _snippet_from_body(r[3], query),
                    "via": "vec",
                }
            else:
                details[rowid]["via"] = "both"
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return [
            {**details[rowid], "rrf": round(score, 4)}
            for rowid, score in ranked
        ]
    finally:
        conn.close()


def list_tags(prefix: str | None = None, limit: int = 50) -> list[dict]:
    conn = get_db()
    if prefix:
        rows = conn.execute(
            "SELECT tag, COUNT(*) c FROM atom_tags WHERE tag = ? OR tag LIKE ? "
            "GROUP BY tag ORDER BY c DESC, tag ASC LIMIT ?",
            (prefix, prefix + "/%", limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT tag, COUNT(*) c FROM atom_tags GROUP BY tag "
            "ORDER BY c DESC, tag ASC LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()
    return [{"tag": r[0], "count": r[1]} for r in rows]


def safe_resolve(path: str) -> Path:
    full = (VAULT / path).resolve()
    if not str(full).startswith(str(VAULT) + os.sep) and full != VAULT:
        raise ValueError(f"path escapes vault: {path}")
    return full


def read_atom(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
    full = safe_resolve(path)
    if not full.is_file():
        raise FileNotFoundError(path)
    text = full.read_text(errors="replace")
    if start_line is None and end_line is None:
        return text
    lines = text.splitlines()
    s = max((start_line or 1) - 1, 0)
    e = end_line if end_line is not None else len(lines)
    return "\n".join(lines[s:e])


# ---------------- LLM (vault_ask) ----------------


def _ollama_chat(url: str, model: str, prompt: str, timeout: int = 60) -> str | None:
    """Call Ollama /api/chat. Returns text or None on failure / empty content."""
    try:
        req = urllib.request.Request(
            f"{url.rstrip('/')}/api/chat",
            data=json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
        text = ((payload.get("message") or {}).get("content") or "").strip()
        return text or None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def _anthropic_chat(api_key: str, model: str, prompt: str, timeout: int = 60) -> str | None:
    """Call Anthropic /v1/messages. Returns text or None on failure / empty content."""
    try:
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps({
                "model": model,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            }).encode(),
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
        parts = payload.get("content") or []
        text = "".join(p.get("text", "") for p in parts if p.get("type") == "text").strip()
        return text or None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def _ask_llm(prompt: str) -> tuple[str | None, str]:
    """Tier cascade. Returns (answer, tier_name). Advances on failure / empty.

    Tier 1 'local'  — Ollama at VSEARCH_LLM_URL (default localhost:11434),
                      VSEARCH_LLM_MODEL (default llama3.1:8b-16k).
    Tier 2 'studio' — Ollama at VSEARCH_STUDIO_URL (unset = skipped),
                      VSEARCH_STUDIO_MODEL (defaults to tier-1 model).
    Tier 3 'api'    — Anthropic /v1/messages (ANTHROPIC_API_KEY unset = skipped),
                      VSEARCH_ANTHROPIC_MODEL (default claude-haiku-4-5).
    Tier 4 'raw'    — no LLM reachable; caller falls back to raw excerpts.
    """
    local_url = os.environ.get("VSEARCH_LLM_URL", "http://localhost:11434")
    local_model = os.environ.get("VSEARCH_LLM_MODEL", "llama3.1:8b-16k")
    ans = _ollama_chat(local_url, local_model, prompt)
    if ans:
        return ans, "local"
    studio_url = os.environ.get("VSEARCH_STUDIO_URL")
    if studio_url:
        studio_model = os.environ.get("VSEARCH_STUDIO_MODEL", local_model)
        ans = _ollama_chat(studio_url, studio_model, prompt)
        if ans:
            return ans, "studio"
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        api_model = os.environ.get("VSEARCH_ANTHROPIC_MODEL", "claude-haiku-4-5")
        ans = _anthropic_chat(api_key, api_model, prompt)
        if ans:
            return ans, "api"
    return None, "raw"


def _ask(question: str, k: int = 6, tags: list[str] | None = None) -> dict:
    """Retrieve top sections, synthesise via tiered LLM, return {answer, tier, sources, excerpts}."""
    hits = search(question, k=k, tags=tags, mode="hybrid")
    if not hits:
        return {"answer": None, "tier": "none", "sources": [], "excerpts": []}
    conn = get_db()
    excerpts: list[dict] = []
    try:
        for h in hits:
            row = conn.execute(
                "SELECT body FROM sections_fts WHERE path=? AND section_title=? LIMIT 1",
                (h["path"], h["section"]),
            ).fetchone()
            body = (row[0] if row else h.get("snippet") or "") or ""
            excerpts.append({
                "path": h["path"],
                "section": h["section"],
                "body": body[:2000],
            })
    finally:
        conn.close()
    context = "\n\n---\n\n".join(
        f"[{i+1}] {e['path']} § {e['section']}\n{e['body']}"
        for i, e in enumerate(excerpts)
    )
    prompt = (
        "Answer the question using ONLY the provided context. "
        "Cite the source paths in brackets like [1], [2]. "
        "If the context does not contain the answer, say so plainly.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    answer, tier = _ask_llm(prompt)
    return {
        "answer": answer,
        "tier": tier,
        "sources": [{"path": e["path"], "section": e["section"]} for e in excerpts],
        "excerpts": excerpts if tier == "raw" else [],
    }


# ---------------- CLI ----------------


@click.group()
def cli() -> None:
    """vault-search CLI."""


@cli.command()
@click.option("-v", "--verbose", is_flag=True)
@click.option("--force", is_flag=True, help="Re-process every file regardless of mtime. Use after a chunker change.")
def index(verbose: bool, force: bool) -> None:
    """Walk ~/vault and (re)build the FTS5 index."""
    index_vault(verbose=verbose, force=force)


@cli.command(name="search")
@click.argument("query")
@click.option("-k", default=5, show_default=True)
@click.option("--tag", "-t", "tag_filters", multiple=True, help="Filter by tag (hierarchical: -t type/moc).")
@click.option("--mode", type=click.Choice(["hybrid", "fts", "vec"]), default="hybrid", show_default=True)
@click.option("--json", "as_json", is_flag=True)
def search_cmd(query: str, k: int, tag_filters: tuple[str, ...], mode: str, as_json: bool) -> None:
    """Run a query against the index."""
    results = search(query, k, tags=list(tag_filters) if tag_filters else None, mode=mode)
    if as_json:
        click.echo(json.dumps(results, indent=2, ensure_ascii=False))
        return
    if not results:
        click.echo("(no hits)", err=True)
        return
    for r in results:
        score_tag = (
            f"rrf {r['rrf']}" if "rrf" in r
            else f"dist {r['distance']}" if "distance" in r
            else f"score {r['score']}"
        )
        click.echo(f"{r['path']}  [{score_tag}, via {r['via']}]")
        if r["section"]:
            click.echo(f"  §  {r['section']}")
        click.echo(f"  {r['snippet']}")
        click.echo()


@cli.command(name="embed")
@click.option("-v", "--verbose", is_flag=True)
def embed_cmd(verbose: bool) -> None:
    """Backfill embeddings for sections missing one. Idempotent. In-process (fastembed)."""
    embed_missing(verbose=verbose)


@cli.command(name="tags")
@click.option("--prefix", "-p", help="Filter to tags under this prefix (hierarchical).")
@click.option("--limit", "-n", default=50, show_default=True)
@click.option("--json", "as_json", is_flag=True)
def tags_cmd(prefix: str | None, limit: int, as_json: bool) -> None:
    """List tags in the vault with usage counts."""
    rows = list_tags(prefix=prefix, limit=limit)
    if as_json:
        click.echo(json.dumps(rows, indent=2, ensure_ascii=False))
        return
    if not rows:
        click.echo("(no tags)", err=True)
        return
    width = max(len(r["tag"]) for r in rows)
    for r in rows:
        click.echo(f"{r['tag'].ljust(width)}  {r['count']}")


@cli.command(name="ask")
@click.argument("question")
@click.option("-k", default=6, show_default=True)
@click.option("--tag", "-t", "tag_filters", multiple=True, help="Filter by tag (hierarchical).")
@click.option("--json", "as_json", is_flag=True)
def ask_cmd(question: str, k: int, tag_filters: tuple[str, ...], as_json: bool) -> None:
    """Ask a question. Retrieves top sections and synthesises via tiered LLM."""
    result = _ask(question, k=k, tags=list(tag_filters) if tag_filters else None)
    if as_json:
        click.echo(json.dumps(result, indent=2, ensure_ascii=False))
        return
    tier = result.get("tier", "?")
    if result.get("answer"):
        click.echo(f"[tier={tier}]")
        click.echo(result["answer"])
        click.echo()
        click.echo("Sources:")
        for i, s in enumerate(result.get("sources", []), start=1):
            click.echo(f"  [{i}] {s['path']} § {s['section']}")
    elif tier == "none":
        click.echo("(no hits)", err=True)
    else:
        click.echo(f"[tier={tier}] LLM unavailable. Excerpts:", err=True)
        for i, e in enumerate(result.get("excerpts", []), start=1):
            click.echo(f"  [{i}] {e['path']} § {e['section']}")
            preview = e["body"][:240].strip().replace("\n", " ")
            click.echo(f"      {preview}...")


@cli.command()
@click.argument("path")
@click.option("--start", type=int)
@click.option("--end", type=int)
def read(path: str, start: int | None, end: int | None) -> None:
    """Read a vault file (optionally a line range)."""
    click.echo(read_atom(path, start, end))


@cli.command()
@click.option("-v", "--verbose", is_flag=True)
@click.option("--initial-index/--no-initial-index", default=True, help="Catch up the index before watching")
def watch(verbose: bool, initial_index: bool) -> None:
    """Watch VAULT_PATH and reindex on change. Runs until SIGINT."""
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    if initial_index:
        index_vault(verbose=verbose)

    class Handler(FileSystemEventHandler):
        def _process(self, src: str, event_type: str) -> None:
            p = Path(src)
            if p.suffix != ".md":
                return
            try:
                rel_path = p.relative_to(VAULT)
            except ValueError:
                return
            if should_skip(rel_path):
                return
            rel = str(rel_path)
            conn = sqlite3.connect(DB)
            try:
                if event_type == "deleted":
                    delete_one(conn, rel)
                    conn.commit()
                    if verbose:
                        print(f"  - {rel}", file=sys.stderr)
                elif p.is_file():
                    verb = index_one(conn, p)
                    conn.commit()
                    if verbose and verb in ("new", "updated"):
                        print(f"  {verb[0]} {rel}", file=sys.stderr)
            finally:
                conn.close()

        def on_created(self, event):
            if not event.is_directory:
                self._process(event.src_path, "created")

        def on_modified(self, event):
            if not event.is_directory:
                self._process(event.src_path, "modified")

        def on_deleted(self, event):
            if not event.is_directory:
                self._process(event.src_path, "deleted")

        def on_moved(self, event):
            if not event.is_directory:
                self._process(event.src_path, "deleted")
                self._process(event.dest_path, "created")

    obs = Observer()
    obs.schedule(Handler(), str(VAULT), recursive=True)
    obs.start()
    print(f"watching {VAULT}", file=sys.stderr)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        obs.stop()
        obs.join()


@cli.command()
def serve() -> None:
    """Run as an MCP server (stdio)."""
    mcp = FastMCP("vault-search")

    @mcp.tool()
    def vault_search(
        query: str,
        k: int = 5,
        tags: list[str] | None = None,
        mode: str = "hybrid",
    ) -> list[dict]:
        """Hybrid (FTS5 + embeddings) search over Stefan's ~/vault, ranked per H2 section. Use BEFORE grep.

        Indexed unit is a section (H2 split), not a whole atom. A long atom can return multiple sections; a hit's `section` field tells you which one matched.

        Modes:
          - 'hybrid' (default): runs both keyword (BM25 via FTS5) and semantic (cosine via in-process fastembed / nomic-embed-text), merged with Reciprocal Rank Fusion. Best for natural-language questions. If the embedder is unavailable it degrades to FTS-only — NOT silently: degraded results carry `via='fts-fallback'` and a stderr warning is emitted. Seeing `fts-fallback` means semantic search was down for that query.
          - 'fts': keyword-only. Best for exact-term lookups.
          - 'vec': semantic-only. Best for paraphrased or conceptual queries when you know the terms won't match.

        Optional `tags`: list of hierarchical-tag filters (AND across them). 'type/moc' matches that tag exactly or anything nested under it. Use vault_list_tags to discover available tags.

        Returns: list of dicts with {path, section, snippet, via} plus mode-specific score field (`rrf` for hybrid, `score` for fts, `distance` for vec).
        Cost: ~50-400 tokens per call vs 1000+ for raw grep.
        """
        return search(query, k, tags=tags, mode=mode)

    @mcp.tool()
    def vault_list_tags(prefix: str | None = None, limit: int = 50) -> list[dict]:
        """List hierarchical tags used across the vault with usage counts.

        Use when: you want to discover what tags exist before filtering vault_search,
        or you need to see what's tagged under a prefix ('type/', 'status/').
        Returns: list of {tag, count} sorted by count desc, then alphabetically.
        """
        return list_tags(prefix=prefix, limit=limit)

    @mcp.tool()
    def vault_read(
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        """Read a vault atom by path (relative to ~/vault). Optional line range.

        Use when: vault_search returned a hit and you need full content.
        path: e.g. 'projects/my-project/architecture.md'.
        Cheaper than Read tool for known paths because no client-side overhead.
        """
        return read_atom(path, start_line, end_line)

    @mcp.tool()
    def vault_reindex(verbose: bool = False, force: bool = False) -> dict:
        """Rebuild the FTS5 index from ~/vault. Returns counts.

        Use when: you know files have changed and the index might be stale.
        Cheap (~1s for 1000 atoms). Incremental: only re-parses changed files by mtime.
        force=True bypasses the mtime check and re-chunks every file — use after a chunker change.
        """
        return index_vault(verbose=verbose, force=force)

    @mcp.tool()
    def vault_ask(
        question: str,
        k: int = 6,
        tags: list[str] | None = None,
    ) -> dict:
        """Answer a natural-language question from the vault. Runs hybrid search, retrieves the top k sections, synthesises an answer via a tiered LLM cascade. Use when you want a synthesised answer with citations rather than a raw hit list.

        Tiers, tried in order, advance on failure (unreachable / error / empty output):
          1. 'local'  — Ollama on the LAN box (VSEARCH_LLM_URL, VSEARCH_LLM_MODEL — default localhost:11434, llama3.1:8b-16k).
          2. 'studio' — Ollama on a second host over tailscale (VSEARCH_STUDIO_URL — unset = skipped).
          3. 'api'    — Anthropic /v1/messages (ANTHROPIC_API_KEY — unset = skipped). VSEARCH_ANTHROPIC_MODEL (default claude-haiku-4-5).
          4. 'raw'    — all LLM tiers unreachable; returns ranked excerpts only.

        Returns: {answer, tier, sources, excerpts}.
          - tier in {'local','studio','api','raw','none'}
          - 'raw' = LLM tiers failed; excerpts populated for the caller to read
          - 'none' = search returned no hits
        The 'tier' field is the non-silent degradation marker, same pattern as vault_search's 'fts-fallback'.
        """
        return _ask(question, k=k, tags=tags)

    mcp.run()


if __name__ == "__main__":
    cli()
