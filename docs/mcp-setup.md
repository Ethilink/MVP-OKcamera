# MCP Setup — `company-brain`

This repo ships a project-scoped MCP config in [`.mcp.json`](../.mcp.json) so
everyone on the team gets the same tools when they open the project in Claude
Code (or any MCP-aware client).

The `company-brain` server exposes a local **Obsidian vault** as a read/write
filesystem the agent can browse — notes, docs, and any other markdown that lives
in your "company brain".

## ⚠️ One thing you MUST change

The path in `.mcp.json` is a **placeholder**. Point it at *your own* vault:

```json
{
  "mcpServers": {
    "company-brain": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/you/Documents/Obsidian/company-brain/root"   // 👈 change this
      ]
    }
  }
}
```

Replace `/Users/you/Documents/Obsidian/company-brain/root` with the absolute
path to your vault, e.g. `/Users/bram/Documents/Obsidian/company-brain/root`.

> The path is machine-specific, so **don't commit your personal path**. See
> [Keeping your path private](#keeping-your-path-private) below.

## How to find your path

On macOS, open the folder in Finder, then drag it into a terminal (it pastes the
full path), or:

```bash
cd /path/to/your/vault && pwd
```

## Requirements

- **Node.js** installed (`npx` ships with it). The `-y` flag lets `npx` fetch
  `@modelcontextprotocol/server-filesystem` on first run — no manual install.

## Verify it works

In Claude Code, run:

```
/mcp
```

You should see `company-brain` listed as connected. If it errors, double-check
the path exists and is spelled correctly.

## Keeping your path private

`.mcp.json` is committed and shared. If you'd rather **not** edit the shared file
with your machine-specific path (and risk committing it), register the server at
**local scope** instead — it's stored in `~/.claude.json` under your project
entry, never in the repo:

```bash
claude mcp add company-brain \
  --scope local \
  -- npx -y @modelcontextprotocol/server-filesystem \
     /Users/you/Documents/Obsidian/company-brain/root
```

Claude Code supports three scopes:

| Scope | Stored in | Shared with team? |
|-------|-----------|-------------------|
| `local` (default) | `~/.claude.json` (this project only) | No |
| `project` | `.mcp.json` (repo root) | Yes — via git |
| `user` | `~/.claude.json` (all your projects) | No |

So: use the committed `.mcp.json` for the team default, or `--scope local` to
keep your real path off git. A local-scoped server with the same name takes
precedence over the project one.
