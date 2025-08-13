-- Powerleague Stats - core schema & policies
create extension if not exists pgcrypto;
create extension if not exists pg_trgm;

-- Core tables
create table if not exists public.players (
  id uuid primary key default gen_random_uuid(),
  name text unique not null,
  photo_url text,
  notes text,
  inserted_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.matches (
  id uuid primary key default gen_random_uuid(),
  season int,
  gw int,
  side_count int,
  team_a text,
  team_b text,
  score_a int,
  score_b int,
  date date,
  motm_name text,
  is_draw boolean default false,
  formation_a text,
  formation_b text,
  notes text,
  inserted_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (season, gw)
);

create table if not exists public.lineups (
  id uuid primary key default gen_random_uuid(),
  season int,
  gw int,
  match_id uuid references public.matches(id) on delete cascade,
  team text check (team in ('Non-bibs','Bibs')),
  player_id uuid references public.players(id) on delete cascade,
  player_name text,
  is_gk boolean default false,
  goals int default 0,
  assists int default 0,
  line int,
  slot int,
  position text,
  inserted_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.awards (
  id uuid primary key default gen_random_uuid(),
  season int,
  month int,
  type text check (type in ('MOTM','POTM')),
  gw int,
  player_id uuid references public.players(id),
  player_name text,
  notes text,
  inserted_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- updated_at triggers
create or replace function public.set_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists trg_players_updated on public.players;
create trigger trg_players_updated before update on public.players
for each row execute function public.set_updated_at();

drop trigger if exists trg_matches_updated on public.matches;
create trigger trg_matches_updated before update on public.matches
for each row execute function public.set_updated_at();

drop trigger if exists trg_lineups_updated on public.lineups;
create trigger trg_lineups_updated before update on public.lineups
for each row execute function public.set_updated_at();

drop trigger if exists trg_awards_updated on public.awards;
create trigger trg_awards_updated before update on public.awards
for each row execute function public.set_updated_at();

-- Helpful indexes
create index if not exists idx_players_name_trgm on public.players using gin (name gin_trgm_ops);
create index if not exists idx_lineups_match_team on public.lineups (match_id, team);
create index if not exists idx_lineups_player on public.lineups (player_id);
create index if not exists idx_awards_season_month on public.awards (season, month);
create index if not exists idx_matches_date on public.matches (date);

-- RLS: public read, writes via service_role only
alter table public.players enable row level security;
alter table public.matches enable row level security;
alter table public.lineups enable row level security;
alter table public.awards enable row level security;

drop policy if exists players_select_all on public.players;
create policy players_select_all on public.players for select using (true);

drop policy if exists matches_select_all on public.matches;
create policy matches_select_all on public.matches for select using (true);

drop policy if exists lineups_select_all on public.lineups;
create policy lineups_select_all on public.lineups for select using (true);

drop policy if exists awards_select_all on public.awards;
create policy awards_select_all on public.awards for select using (true);

-- Storage: ensure public avatars bucket entry; create the bucket via Dashboard (Storage) if missing
insert into storage.buckets (id, name, public)
values ('avatars','avatars', true)
on conflict (id) do nothing;

drop policy if exists "Avatar public read" on storage.objects;
create policy "Avatar public read"
on storage.objects for select
using ( bucket_id = 'avatars' );

-- Awards uniqueness hardening
drop index if exists uq_awards_motm_unique;
create unique index uq_awards_motm_unique
  on public.awards (season, gw)
  where type = 'MOTM';

drop index if exists uq_awards_potm_unique;
create unique index uq_awards_potm_unique
  on public.awards (season, month)
  where type = 'POTM';
