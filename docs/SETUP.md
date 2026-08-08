# Setup Guide — Running Archiva on a New Machine

Step-by-step instructions to get Archiva running from a fresh clone, on
Windows, macOS, or Linux. For the architecture overview and API reference,
see [README.md](../README.md) — this file is just the "zero to running" path.

---

## 0. What you need before you start

| Requirement | Version | Check with |
|---|---|---|
| Python | 3.10+ | `python --version` |
| Node.js | 18+ | `node --version` |
| PostgreSQL | 13+ | `psql --version` |
| Git | any | `git --version` |
| A Groq API key | — | https://console.groq.com (free tier is fine) |

Don't have Postgres installed yet?
- **Windows:** https://www.postgresql.org/download/windows/ (the installer
  asks you to set a password for the `postgres` superuser — write it down,
  you'll need it once in step 2)
- **macOS:** `brew install postgresql@16 && brew services start postgresql@16`
- **Linux (Debian/Ubuntu):** `sudo apt install postgresql && sudo systemctl start postgresql`

You do **not** need Docker, and you do **not** need pgvector — Archiva
deliberately stores embeddings as a plain array column and does similarity
search in Python (see `db/schema.sql`'s header comment for why).

---

## 1. Clone the repo

```bash
git clone <your-fork-or-repo-url> archiva
cd archiva
```

---

## 2. Create the Postgres role and database

This is the **one manual step that can't be automated** — everything after
this (the actual tables) is created automatically when the app starts.

Open a terminal with access to `psql` as the Postgres superuser:

**Windows** (from an elevated PowerShell, or the "SQL Shell (psql)" app
installed alongside Postgres):
```powershell
psql -U postgres
```

**macOS / Linux:**
```bash
sudo -u postgres psql
```

You'll be prompted for the superuser password you set during install. Once
inside the `psql` prompt, run:

```sql
CREATE ROLE archiva LOGIN PASSWORD 'choose-a-password-here';
CREATE DATABASE archiva OWNER archiva;
\q
```

Use a real password (not the placeholder above) and keep it — you'll paste
it into `.env` in the next step. Don't reuse the Postgres superuser
account for the app; `archiva` is a dedicated, lower-privilege role scoped
to its own database.

**If Postgres is on a non-default port** (e.g. you have more than one
instance installed), note the port — you'll need it for `DATABASE_URL` too.
Check with:
```sql
SHOW port;
```

---

## 3. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set the two required values:

```dotenv
GROQ_API_KEY=gsk_your_actual_key_from_console.groq.com

DATABASE_URL=postgresql://archiva:choose-a-password-here@localhost:5432/archiva
```

(Swap `5432` for your actual port if it's different.) Everything else in
`.env.example` has a working default — leave it alone unless you know you
want to tune it (see `config.py` for what each setting does).

> **Multiple Groq keys?** Set `GROQ_API_KEYS=key1,key2,key3` instead of
> `GROQ_API_KEY` — the app rotates across them automatically when one hits
> a rate limit. `GROQ_API_KEY` still works fine for a single key.

---

## 4. Install backend dependencies

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

This pulls in `sentence-transformers` (dense embeddings) and its `torch`
dependency — expect a ~1-2 GB download and a couple of minutes on first
install. Everything else is small.

---

## 5. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

---

## 6. Run it

**Windows:**
```bat
.\start.bat
```

**macOS / Linux:**
```bash
bash start.sh
```

This starts the FastAPI backend on `http://localhost:8000` and the Vite
frontend on `http://localhost:3000` (`frontend/vite.config.js` pins the
port there explicitly). Note: `start.sh`'s own printed summary says
`5173` — that's stale, Vite's old default before the port was pinned;
the frontend is actually on `3000` on every OS.

**The database schema is created automatically the first time the backend
starts** — `main.py`'s startup sequence runs `db/schema.sql` (`CREATE TABLE
IF NOT EXISTS ...`, fully idempotent) against whatever `DATABASE_URL`
points to. There is no separate migration step and nothing to run with
`psql -f schema.sql` by hand. If you ever want to confirm it worked:

```sql
psql -U archiva -d archiva -c "\dt"
```
should list `documents`, `chunks`, and `feedback_logs`.

---

## 7. Verify it worked

1. `curl http://localhost:8000/health` → should return `{"status": "ok", ...}`.
2. Open the frontend URL in a browser — you should see the Archiva chat UI.
3. Upload a small `.txt` or `.pdf` file via the Documents panel.
4. Ask a question about it in the chat box.

If step 4 returns a real answer with a source citation, setup is complete.

---

## Troubleshooting

**"password authentication failed for user archiva"**
Double-check the password in `DATABASE_URL` matches what you set in step 2
exactly (no quotes, no trailing spaces).

**Forgot the Postgres superuser password and can't even get into `psql`:**
1. Find `pg_hba.conf` (Windows: `C:\Program Files\PostgreSQL\<version>\data\pg_hba.conf`;
   Linux: `/etc/postgresql/<version>/main/pg_hba.conf`).
2. Change the `METHOD` column for local/`127.0.0.1` connections from
   `scram-sha-256` (or `md5`) to `trust`, temporarily.
3. Restart the Postgres service (Windows: `Restart-Service postgresql-x64-<version>`
   from an elevated PowerShell; Linux: `sudo systemctl restart postgresql`).
4. Connect with no password: `psql -U postgres`, then
   `ALTER ROLE postgres PASSWORD 'new-password';`.
5. **Revert `pg_hba.conf`** back to its original auth method and restart
   the service again — leaving it on `trust` means anyone on the machine
   can connect as any role with no password.

**App starts but `/upload` fails / documents don't persist:**
`DATABASE_URL` is either unset or unreachable — the app falls back to an
empty in-memory store so it can still start, but nothing is saved. Check
the backend terminal output on startup for a Postgres connection error.

**"port already in use" on 8000 or 3000/5173:**
Something else is already running there, or a previous `start.bat`/`start.sh`
run didn't shut down cleanly. `start.bat` kills anything on those ports
automatically; on macOS/Linux, find and stop it manually:
```bash
lsof -i :8000    # or :5173
kill <pid>
```

**`npm install` fails / frontend won't start:**
Confirm Node 18+ with `node --version`. Delete `frontend/node_modules` and
`frontend/package-lock.json` and retry `npm install` if it's a stale
lockfile issue.

**Tests won't run / Postgres-dependent tests are skipped:**
Expected if `DATABASE_URL` isn't set when running `pytest` — those tests
skip cleanly rather than fail. Export the same `DATABASE_URL` from your
`.env` into the shell before running `pytest` to include them:
```bash
# macOS/Linux
export DATABASE_URL=postgresql://archiva:...@localhost:5432/archiva
pytest -v

# Windows (PowerShell)
$env:DATABASE_URL = "postgresql://archiva:...@localhost:5432/archiva"
pytest -v
```

---

## Moving to yet another machine later

Nothing above is machine-specific once it's working — to set up a second
machine (or redeploy), repeat steps 1-6 there. The Postgres role/database
(step 2) is the only step that can't be scripted generically, since it
depends on how Postgres was installed and secured on that machine; every
other step is identical.
