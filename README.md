# Powerleague Stats — Fresh Setup (Supabase-first Import)

This repo bootstraps a **Streamlit + Supabase** app with a **Supabase-native import pipeline** for historical data (no CSV upload in the app).

## 0) Prereqs

- Supabase account
- Python 3.10+
- (Optional) GitHub repo to host this code

## 1) Create a new Supabase project

1. In the Supabase Dashboard, create a project.
2. Open **SQL Editor** → run `schema.sql` (from this repo).  
   - This creates tables, indexes, RLS, and the public-read `avatars` bucket entry.
3. Go to **Storage** → **Create bucket** → name `avatars` → Public ✅.

## 2) Import historical data directly in Supabase

We use **staging tables** and a robust SQL pipeline to normalize data.

1. In **SQL Editor**, run `import.sql` to create the staging tables.
2. In **Table Editor**, open each staging table and **Import data**:
   - `players.csv` → `staging_players`
   - `matches.csv` → `staging_matches`
   - `lineups.csv` → `staging_lineups`
3. Back in **SQL Editor**, run the **NORMALIZE + UPSERT** blocks in `import.sql` **in order**:
   - Players → Matches → Lineups → Awards
4. (Optional) Truncate staging tables at the end (see `import.sql`).

> You can repeat the process as many times as you like; matches upsert by `(season, gw)`, lineups delete-then-insert per `(match, team)`, and awards are deduped.

## 3) Local app setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:

```toml
SUPABASE_URL = "<your-project-url>"
SUPABASE_ANON_KEY = "<anon-key>"
SUPABASE_SERVICE_KEY = "<service-role-key>"  # needed for admin-only writes (avatars/awards)
ADMIN_PASSWORD = "<set-a-strong-password>"
AVATAR_BUCKET = "avatars"
```

Run:

```bash
streamlit run app.py
```

## 4) Deploy

- **Streamlit Community Cloud**: Add the above secrets in the app’s **Settings → Secrets**.
- **Docker/Cloud Run/Render**: Ensure the same env vars are set.

## 5) Admin actions

- Admin login uses the single password from secrets.
- Admin can:
  - Upload/replace player **photos** (HEIC/JPG/PNG → PNG via Pillow HEIF).
  - Add **awards** manually.
  - Export current tables to CSV.
- Historical imports are done **in Supabase**, not via the app.

## 6) Notes

- RLS allows public SELECT on all tables. Writes are only allowed with the **service role** key (held in Streamlit secrets).
- Default formations when blank: `1-2-1` for 5s, `2-1-2-1` for 7s.
- Pitches are mobile-first; chips show ⚽/🅰️, with a MOTM banner per match.
- Reads are cached; use the Admin page "Force refresh caches" after imports.

## 7) Troubleshooting

- If you ever want to **start over**, you can safely wipe data without dropping tables:

```sql
truncate table public.lineups restart identity cascade;
truncate table public.awards restart identity cascade;
truncate table public.matches restart identity cascade;
truncate table public.players restart identity cascade;
delete from storage.objects where bucket_id = 'avatars';
```

- If you imported wrong rows into staging, just:
```sql
truncate table public.staging_players;
truncate table public.staging_matches;
truncate table public.staging_lineups;
```

- Create/rename/merge players using SQL examples from the docs conversation, or ask the app admin to adjust photos/awards.

Enjoy! ⚽
