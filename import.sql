-- STAGING TABLES
drop table if exists public.staging_players;
drop table if exists public.staging_matches;
drop table if exists public.staging_lineups;

create table public.staging_players (
  name text, photo_url text, notes text
);

create table public.staging_matches (
  season text, gw text, side_count text,
  team_a text, team_b text, score_a text, score_b text,
  date text, motm_name text, is_draw text,
  formation_a text, formation_b text, notes text
);

create table public.staging_lineups (
  season text, gw text, match_id text,
  team text, player_name text, player_id text,
  is_gk text, goals text, assists text,
  line text, slot text, position text
);

-- HOW TO USE:
-- 1) Upload your CSVs into these staging tables using the Table Editor (Import data).
--    - players.csv -> staging_players
--    - matches.csv -> staging_matches
--    - lineups.csv -> staging_lineups
-- 2) Run the normalization/upsert blocks below IN ORDER.
-- 3) (Optional) Truncate the staging tables after import.

-- NORMALIZE + UPSERT: PLAYERS
insert into public.players (name, photo_url, notes)
select distinct
  nullif(trim(name),'') as name,
  nullif(trim(photo_url),'') as photo_url,
  nullif(trim(notes),'') as notes
from public.staging_players
where coalesce(trim(name),'') <> ''
on conflict (name)
do update set
  photo_url = coalesce(excluded.photo_url, public.players.photo_url),
  notes     = coalesce(excluded.notes, public.players.notes);

-- NORMALIZE + UPSERT: MATCHES
insert into public.matches (
  season, gw, side_count, team_a, team_b,
  score_a, score_b, date, motm_name, is_draw,
  formation_a, formation_b, notes
)
select
  nullif(season,'')::int,
  nullif(gw,'')::int,
  coalesce(nullif(side_count,'')::int, 5),
  coalesce(nullif(trim(team_a),''),'Non-bibs'),
  coalesce(nullif(trim(team_b),''),'Bibs'),
  nullif(score_a,'')::int,
  nullif(score_b,'')::int,
  case when nullif(date,'') is null then null else nullif(date,'')::date end,
  nullif(trim(motm_name),'') as motm_name,
  case
    when nullif(is_draw,'') is null then (nullif(score_a,'')::int = nullif(score_b,'')::int)
    else lower(trim(is_draw)) in ('t','true','1','yes','y')
  end as is_draw,
  case
    when coalesce(trim(formation_a),'')='' then case coalesce(nullif(side_count,'')::int,5) when 7 then '2-1-2-1' else '1-2-1' end
    else formation_a end as formation_a,
  case
    when coalesce(trim(formation_b),'')='' then case coalesce(nullif(side_count,'')::int,5) when 7 then '2-1-2-1' else '1-2-1' end
    else formation_b end as formation_b,
  nullif(notes,'')
from public.staging_matches
where coalesce(trim(season),'') <> '' and coalesce(trim(gw),'') <> ''
on conflict (season, gw) do update
  set score_a = excluded.score_a,
      score_b = excluded.score_b,
      date = excluded.date,
      motm_name = excluded.motm_name,
      is_draw = excluded.is_draw,
      formation_a = excluded.formation_a,
      formation_b = excluded.formation_b,
      notes = excluded.notes;

-- DELETE-THEN-INSERT LINEUPS per (match, team)
delete from public.lineups l
using (
  select distinct m.id as match_id,
         case when lower(trim(s.team))='bibs' then 'Bibs' else 'Non-bibs' end as team
  from public.staging_lineups s
  join public.matches m
    on m.season = nullif(s.season,'')::int
   and m.gw     = nullif(s.gw,'')::int
  where coalesce(trim(s.team),'') <> ''
) d
where l.match_id = d.match_id
  and l.team = d.team;

insert into public.lineups (
  season, gw, match_id, team,
  player_id, player_name, is_gk, goals, assists,
  line, slot, position
)
select
  m.season, m.gw, m.id,
  case when lower(trim(s.team))='bibs' then 'Bibs' else 'Non-bibs' end as team,
  p.id as player_id,
  s.player_name,
  case when lower(coalesce(trim(s.is_gk),'0')) in ('1','true','t','yes','y') then true else false end as is_gk,
  coalesce(nullif(s.goals,'')::int, 0)   as goals,
  coalesce(nullif(s.assists,'')::int, 0) as assists,
  nullif(s.line,'')::int,
  nullif(s.slot,'')::int,
  coalesce(s.position,'')
from public.staging_lineups s
join public.matches m
  on m.season = nullif(s.season,'')::int
 and m.gw     = nullif(s.gw,'')::int
left join public.players p
  on lower(p.name) = lower(trim(s.player_name));

-- AUTO-FILL MOTM AWARDS from matches
insert into public.awards (season, month, type, gw, player_id, player_name, notes)
select
  m.season,
  case when m.date is null then null else extract(month from m.date)::int end as month,
  'MOTM' as type,
  m.gw,
  p.id as player_id,
  m.motm_name as player_name,
  null::text as notes
from public.matches m
left join public.players p on lower(p.name) = lower(trim(m.motm_name))
where coalesce(trim(m.motm_name),'') <> ''
on conflict do nothing;

-- OPTIONAL: clear staging after successful import
-- truncate table public.staging_players;
-- truncate table public.staging_matches;
-- truncate table public.staging_lineups;
