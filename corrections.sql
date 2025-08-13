-- Name corrections & merges after import
begin;

-- Ani -> Anirudh (merge if both exist)
with ids as (
  select
    max(case when lower(name)='ani' then id end) as src_id,
    max(case when lower(name)='anirudh' then id end) as dst_id
  from public.players
)
update public.lineups l set player_id = ids.dst_id
from ids where l.player_id = ids.src_id;

update public.awards a set player_id = ids.dst_id
from ids where a.player_id = ids.src_id;

update public.lineups set player_name = 'Anirudh' where lower(player_name)='ani';
update public.awards  set player_name = 'Anirudh' where lower(player_name)='ani';

delete from public.players p using ids where p.id = ids.src_id;

-- Abdullah Y13 -> Mohammad Abdullah
-- Update players.name if present
update public.players set name='Mohammad Abdullah' where name='Abdullah Y13';

-- Update text copies
update public.lineups set player_name='Mohammad Abdullah' where player_name='Abdullah Y13';
update public.awards  set player_name='Mohammad Abdullah' where player_name='Abdullah Y13';

commit;
