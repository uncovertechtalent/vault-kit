# vault-kit

A fit-it-yourself second brain for working with Claude Code: a plain-markdown vault your agent can read, write, and search, with hybrid full-text + semantic search exposed as MCP tools, and a seeding skill that builds the structure around YOUR life instead of copying mine.

Sibling to [vestige-kit](https://github.com/uncovertechtalent/vestige-kit): same doctrine, different layer. That one fits the model's output register to you; this one gives the model a durable memory that is yours.

## The idea in five layers

An agent's memory is a stack. Context window at the bottom, ephemeral. Session summaries above it, lossy. This kit is the layer that doesn't rot: plain markdown files, one idea per file, linked, indexed, searchable. The agent reads atoms instead of re-deriving your life every session, and writes new atoms instead of forgetting what it learned. Everything above (multi-agent fleets, durable task graphs) can come later; nothing below survives without this layer.

Plain files beat databases here for one reason: you can read them, grep them, edit them, and walk away with them. No lock-in, no schema, no export problem. Obsidian renders them nicely; nothing requires it.

## What's in the box

| Piece | What it does |
|---|---|
| `vault-starter/` | The scaffold: folder conventions, a CLAUDE.md template that teaches your agent the vault's rules, an example atom and MOC. |
| `search/vsearch.py` | Single-file hybrid search: SQLite FTS5 keyword + in-process semantic embeddings (fastembed, CPU, no network), merged with reciprocal rank fusion. Runs as a CLI or as an MCP server with five tools. Degrades to keyword-only if the embedder is unavailable, and says so rather than failing silently. |
| `skills/seed/` | Interactive skill: interviews you about your domains, then generates your vault structure and CLAUDE.md fitted to your actual life. |
| `skills/atomize/` | Turns any document, chat log, or braindump into linked atoms plus an index note. |
| `SEEDING.md` | The conventions and the reasoning behind them, for running the process by hand. |

## Quickstart

```bash
git clone https://github.com/uncovertechtalent/vault-kit
cp -r vault-kit/vault-starter ~/vault
cp -r vault-kit/skills/* ~/.claude/skills/
```

Index and search (requires [uv](https://docs.astral.sh/uv/); dependencies resolve on first run):

```bash
uv run vault-kit/search/vsearch.py index
uv run vault-kit/search/vsearch.py search "your query"
```

Wire the MCP server into Claude Code (`.mcp.json` or `claude mcp add`):

```json
{
  "mcpServers": {
    "vault-search": {
      "command": "uv",
      "args": ["run", "/path/to/vault-kit/search/vsearch.py", "serve"]
    }
  }
}
```

Then in Claude Code:

```
/seed
```

and answer the questions. The skill replaces the starter's generic folders with a structure derived from your domains, and writes a CLAUDE.md the agent will actually follow.

Environment knobs (all optional): `VAULT_PATH` (default `~/vault`), `VSEARCH_OLLAMA_URL` (route embeddings to an Ollama box instead of in-process CPU), `VSEARCH_LLM_URL`/`VSEARCH_LLM_MODEL` (local model for the ask command).

## Why seed instead of copy

A vault's structure encodes what its owner thinks about. Mine has clusters for things you don't care about, and yours will need clusters I can't predict. Copying a finished vault structure gives you empty folders shaped like someone else's head: the folders feel organized and stay empty, because nothing in your day produces content shaped like them. The seed skill works from your actual inputs (what you work on, what you keep re-explaining to the model, what you look up twice a week) and builds only the structure those inputs justify. Structure that arrives before content is decoration; the kit's rule is content first, folders when the content demands them.

## The honest limits

- Search quality tracks atomization quality. A vault of 5,000-word documents searches like a filing cabinet; a vault of one-idea atoms searches like a memory. The atomize skill is not optional glue, it is half the product.
- The embedder downloads ~130MB of ONNX weights on first use and runs on CPU. Fine to a few thousand atoms; beyond that, point `VSEARCH_OLLAMA_URL` at any Ollama box.
- This is the memory layer only. It does not schedule anything, run agents, or sync anywhere. Files plus search, deliberately.

## Requirements

Python 3.12+ via uv (deps inline in the script: mcp, fastembed). Node not required. MIT.
