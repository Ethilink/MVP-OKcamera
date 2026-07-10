# Connecting the `company-brain` Obsidian vault

Project context (client brief, PRD, design docs, research) lives in the team's Obsidian
vault, `company-brain`, not in this repo. This doc covers how to make it readable —
by a human or by an agent — via MCP, or directly off disk if MCP isn't available.

Nothing here is repo-shared config: MCP servers are registered in your own local
`~/.claude.json`, never committed.

## Option 1 — Obsidian Local REST API (MCP, recommended)

Requires the Obsidian app open with the vault loaded (the plugin serves the API from
inside the running app).

1. In Obsidian: **Settings → Community plugins → Browse** → search "Local REST API" →
   install and enable it.
2. Open the plugin's settings and copy its API key. Note the HTTP port (default `27123`).
3. Register the MCP server:
   ```bash
   claude mcp add --transport http obsidian http://127.0.0.1:27123/mcp/ \
     --header "Authorization: Bearer <your-api-key>" \
     --scope user
   ```
   Use `--scope user` so it's available in every project on the machine (the vault
   isn't specific to this repo). Use `--scope local` instead to limit it to one project.
4. Verify: `claude mcp list` should show `obsidian: ... - ✔ Connected`, or run `/mcp`
   inside Claude Code.

Caveat: this stops working the moment Obsidian is closed, since the plugin only runs
inside the app.

## Option 2 — Direct path reading (fallback, no app or plugin required)

An Obsidian vault is just a folder of Markdown files on disk. Any agent with normal
filesystem tools (Read/Glob/Grep) can read it directly, with no setup at all, as long
as it knows the vault's absolute path (e.g. `~/Documents/company-brain`). This works
even if Obsidian isn't running and the MCP server above is unreachable.

If you'd rather have it registered as MCP without depending on the Obsidian app being
open, point the generic filesystem MCP server at the vault folder instead:

```bash
claude mcp add company-brain --scope local -- npx -y @modelcontextprotocol/server-filesystem /absolute/path/to/company-brain
```

This exposes the folder over MCP with zero dependency on Obsidian running, but gives
plain file access rather than the richer Obsidian-aware operations (search, tags,
frontmatter) that Option 1 provides.

## What an agent should do if the vault isn't reachable

1. Check whether it's connected (`/mcp`, or attempt a read via the MCP tools).
2. If not, try direct path reading (Option 2) — if the vault's path is already known
   from prior context, just read it; no setup needed.
3. If neither works, don't guess at product context or invent decisions that belong in
   the vault. Offer to set it up per this doc instead — ask the user for their local
   vault path and get explicit permission before running `claude mcp add`, since it
   changes their local Claude Code config.
