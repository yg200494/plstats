# app.py — Powerleague Stats (FotMob-style pitch + duo/nemesis metrics)

import io
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import pillow_heif
from supabase import create_client, Client


# =========================
# App config & theme
# =========================
st.set_page_config(page_title="Powerleague Stats", page_icon="⚽", layout="wide", initial_sidebar_state="collapsed")

CSS = """
<style>
:root{
  --bg1:#0b1324; --bg2:#0a1a2f;
  --card:rgba(15,23,42,.66); --card-br:rgba(255,255,255,.10);
  --chip-bg:rgba(255,255,255,.10); --chip-br:rgba(255,255,255,.30);
}
html,body,[data-testid="stAppViewContainer"]{
  background:linear-gradient(180deg,var(--bg1) 0%, var(--bg2) 100%);
  color:#e6edf3;
}
footer,#MainMenu{display:none}
.card{background:var(--card);border:1px solid var(--card-br);border-radius:16px;padding:16px}
.banner{
  background:linear-gradient(90deg,#0f172a 0%,#0b1324 100%);
  border:1px solid rgba(255,255,255,.12); border-radius:16px; padding:14px 16px;
  display:flex; align-items:center; justify-content:space-between; gap:12px; color:#fff;
}
.banner .title{font-size:18px;font-weight:700}
.banner .sub{opacity:.85;font-size:13px}
.badge{background:#22c55e;color:#062616;font-weight:700;border-radius:999px;padding:6px 12px;font-size:12px;border:1px solid rgba(0,0,0,.25)}
.hr{height:1px; background:rgba(255,255,255,.10); margin:10px 0 16px}
.chip{
  display:inline-flex; align-items:center; gap:10px;
  background:var(--chip-bg); border:1px solid var(--chip-br);
  padding:9px 12px; border-radius:18px; font-size:15px; line-height:1.1; color:#fff; max-width:100%;
}
.chip img{width:28px;height:28px;border-radius:50%;object-fit:cover;border:1px solid rgba(255,255,255,.35)}
.chip .meta{font-size:12px; opacity:.9}
.kv{display:flex;gap:10px;flex-wrap:wrap}
.kv .k{opacity:.8}
.smallcap{opacity:.8; font-size:13px}
.pitch-wrap{position:relative;width:100%;padding-top:150%;border-radius:20px;overflow:hidden;box-shadow:0 4px 12px rgba(0,0,0,.35);border:1px solid rgba(255,255,255,.18)}
.pitch-bg{position:absolute;inset:0;background-image:url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiPjxyZWN0IHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiIGZpbGw9IiMzZTllNGQiLz48cGF0aCBkPSJNMCwxMGgxMDAwdjEwSDB6IiBmaWxsPSIjNDZhODUxIiBvcGFjaXR5PSIwLjIiLz48cGF0aCBkPSJNMCwzMGgxMDAwdjEwSDB6IiBmaWxsPSIjNDZhODUxIiBvcGFjaXR5PSIwLjIiLz48cGF0aCBkPSJNMCw1MGgxMDAwdjEwSDB6IiBmaWxsPSIjNDZhODUxIiBvcGFjaXR5PSIwLjIiLz48L3N2Zz4=');background-size:cover}
.pitch-layer{position:absolute;inset:0}
.spot{position:absolute;transform:translate(-50%,-50%);display:flex;flex-direction:column;align-items:center}
.avatar{position:relative;width:56px;height:56px;border-radius:50%;overflow:hidden;border:2px solid rgba(255,255,255,.35);box-shadow:0 2px 8px rgba(0,0,0,.35)}
.avatar img{width:100%;height:100%;object-fit:cover}
.avatar .init{width:100%;height:100%;display:flex;align-items:center;justify-content:center;background:#cfd8dc;color:#263238;font-weight:800}
.motm{position:absolute;top:-8px;right:-8px;background:gold;color:#000;border-radius:50%;padding:2px 4px;font-size:12px;border:1px solid rgba(0,0,0,.25)}
.name{margin-top:4px;font-size:12.5px;font-weight:700;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,.6);white-space:nowrap}
.chips{display:flex;gap:4px;margin-top:2px}
.statchip{background:#fff;color:#111;padding:2px 6px;border-radius:10px;font-size:12px}
.slotBtn{width:100%;height:42px;border-radius:10px;border:1px dashed rgba(255,255,255,.25);background:rgba(255,255,255,.04)}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =========================
# Secrets & clients
# =========================
REQ = ["SUPABASE_URL", "SUPABASE_ANON_KEY", "ADMIN_PASSWORD", "AVATAR_BUCKET"]
for k in REQ:
    if k not in st.secrets:
        st.error(f"Missing secret: {k}")
        st.stop()

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
SUPABASE_SERVICE_KEY = st.secrets.get("SUPABASE_SERVICE_KEY")
ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
AVATAR_BUCKET = st.secrets["AVATAR_BUCKET"]

sb_public: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

def service() -> Optional[Client]:
    if st.session_state.get("is_admin") and SUPABASE_SERVICE_KEY:
        return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return None

try:
    pillow_heif.register_heif_opener()
except Exception:
    pass


# =========================
# Cached reads
# =========================
@st.cache_data(ttl=30)
def fetch_players() -> pd.DataFrame:
    return pd.DataFrame(sb_public.table("players").select("*").execute().data or [])

@st.cache_data(ttl=30)
def fetch_matches() -> pd.DataFrame:
    res = (
        sb_public.table("matches")
        .select("*")
        .order("season")
        .order("gw")
        .execute()
        .data
        or []
    )
    return pd.DataFrame(res)

@st.cache_data(ttl=30)
def fetch_lineups() -> pd.DataFrame:
    return pd.DataFrame(sb_public.table("lineups").select("*").execute().data or [])

@st.cache_data(ttl=30)
def fetch_awards() -> pd.DataFrame:
    return pd.DataFrame(sb_public.table("awards").select("*").execute().data or [])

def clear_caches():
    fetch_players.clear(); fetch_matches.clear(); fetch_lineups.clear(); fetch_awards.clear()


# =========================
# Data shaping
# =========================
def formation_to_lines(form: str):
    """Return counts per outfield line, GK handled as line 0 implicitly."""
    if not form: return [1,2,1]  # default to 5s 1-2-1
    parts = [int(x) for x in str(form).split("-") if x.strip().isdigit()]
    return [1] + (parts if parts else [2,1])

def build_fact(players, matches, lineups):
    players = players.copy(); matches = matches.copy(); lineups = lineups.copy()
    for df in (players, matches, lineups):
        if "id" in df.columns: df["id"] = df["id"].astype(str)

    use_cols = ["id","team_a","team_b","score_a","score_b","season","gw","date","is_draw","motm_name","formation_a","formation_b","side_count"]
    mi = matches[[c for c in use_cols if c in matches.columns]].rename(columns={"id":"match_id"})
    l = lineups.merge(mi, on="match_id", how="left")

    # prefer plain season/gw or *_x/*_y
    def pick_col(df, base):
        if base in df.columns: return df[base]
        if f"{base}_x" in df.columns: return df[f"{base}_x"]
        if f"{base}_y" in df.columns: return df[f"{base}_y"]
        return pd.Series([np.nan]*len(df), index=df.index)

    l["season"] = pd.to_numeric(pick_col(l, "season"), errors="coerce").astype("Int64")
    l["gw"]     = pd.to_numeric(pick_col(l, "gw"), errors="coerce").astype("Int64")
    for c in ["season_x","season_y","gw_x","gw_y"]:
        if c in l.columns: del l[c]

    l["goals"]   = pd.to_numeric(l.get("goals"), errors="coerce").fillna(0).astype(int)
    l["assists"] = pd.to_numeric(l.get("assists"), errors="coerce").fillna(0).astype(int)

    l["side"] = np.where(l["team"]==l["team_a"], "A", np.where(l["team"]==l["team_b"], "B", None))
    l["team_goals"] = np.where(l["side"]=="A", l["score_a"], np.where(l["side"]=="B", l["score_b"], np.nan))
    l["opp_goals"]  = np.where(l["side"]=="A", l["score_b"], np.where(l["side"]=="B", l["score_a"], np.nan))
    l["team_goals"] = pd.to_numeric(l["team_goals"], errors="coerce")
    l["opp_goals"]  = pd.to_numeric(l["opp_goals"],  errors="coerce")

    l["result"] = np.where(l["is_draw"]==True, "D", np.where(l["team_goals"]>l["opp_goals"], "W", "L"))
    l["ga"] = (l["goals"] + l["assists"]).astype(int)

    # attach player info
    p = players.rename(columns={"id":"player_id"})
    if not p.empty:
        l = l.merge(p[["player_id","name","photo_url"]], on="player_id", how="left")
    l["name"] = l["player_name"].where(l["player_name"].notna() & (l["player_name"].astype(str).str.strip()!=""), l.get("name"))
    l["name"] = l["name"].fillna("Unknown")
    l["photo"] = l.get("photo_url")

    return l, matches


def player_agg(l, season=None, min_games=0, last_gw=None):
    df = l.copy()
    if season is not None: df = df[df["season"]==season]
    if last_gw:
        max_gw = pd.to_numeric(df["gw"], errors="coerce").max()
        if pd.notna(max_gw): df = df[pd.to_numeric(df["gw"], errors="coerce") >= (int(max_gw)-int(last_gw)+1)]
    if df.empty:
        return pd.DataFrame(columns=["name","gp","w","d","l","win_pct","goals","assists","ga","ga_pg","g_pg","a_pg","team_contrib_pct","photo"])
    df["goals"]   = pd.to_numeric(df["goals"], errors="coerce").fillna(0).astype(int)
    df["assists"] = pd.to_numeric(df["assists"], errors="coerce").fillna(0).astype(int)
    df["team_goals"] = pd.to_numeric(df["team_goals"], errors="coerce")

    gp = df.groupby("name").size().rename("gp")
    w  = df[df["result"]=="W"].groupby("name").size().reindex(gp.index, fill_value=0)
    d  = df[df["result"]=="D"].groupby("name").size().reindex(gp.index, fill_value=0)
    l_ = df[df["result"]=="L"].groupby("name").size().reindex(gp.index, fill_value=0)
    goals = df.groupby("name")["goals"].sum().reindex(gp.index, fill_value=0)
    assists = df.groupby("name")["assists"].sum().reindex(gp.index, fill_value=0)

    ga = (goals + assists).astype(float)
    ga_pg = (ga / gp.replace(0,np.nan)).round(2).fillna(0)
    g_pg = (goals / gp.replace(0,np.nan)).round(2).fillna(0)
    a_pg = (assists / gp.replace(0,np.nan)).round(2).fillna(0)

    team_goals_sum = df.groupby("name")["team_goals"].sum(min_count=1).reindex(gp.index)
    denom = team_goals_sum.replace(0,np.nan)
    team_contrib = ((ga/denom)*100).round(1).fillna(0)
    win_pct = ((w.values/np.maximum(gp.values,1))*100).round(1)
    photo = df.groupby("name")["photo"].last().reindex(gp.index)

    out = pd.DataFrame({
        "name": gp.index, "gp": gp.values, "w": w.values, "d": d.values, "l": l_.values,
        "win_pct": win_pct,
        "goals": goals.values.astype(int), "assists": assists.values.astype(int),
        "ga": (goals.values+assists.values).astype(int),
        "ga_pg": ga_pg.values, "g_pg": g_pg.values, "a_pg": a_pg.values,
        "team_contrib_pct": team_contrib.values, "photo": photo.values
    }).sort_values(["ga","goals","assists"], ascending=False).reset_index(drop=True)
    if min_games>0: out = out[out["gp"]>=min_games]
    return out


# ---------- Duos / Nemesis helpers (player & global) ----------
def duo_table(lfact: pd.DataFrame, player: str, season: Optional[int]=None):
    df = lfact.copy()
    if season is not None: df = df[df["season"]==season]
    my = df[df["name"]==player]
    if my.empty: return pd.DataFrame(columns=["teammate","gp","w","d","l","win_pct","ga"])
    t = my.merge(df, on=["match_id","team"], suffixes=("_me","_tm"))
    t = t[t["name_tm"]!=player]
    if t.empty: return pd.DataFrame(columns=["teammate","gp","w","d","l","win_pct","ga"])
    gp = t.groupby("name_tm").size(); w=t[t["result_me"]=="W"].groupby("name_tm").size().reindex(gp.index, fill_value=0)
    d = t[t["result_me"]=="D"].groupby("name_tm").size().reindex(gp.index, fill_value=0)
    l = t[t["result_me"]=="L"].groupby("name_tm").size().reindex(gp.index, fill_value=0)
    ga = (t.groupby("name_tm")["goals_me"].sum() + t.groupby("name_tm")["assists_me"].sum()).reindex(gp.index, fill_value=0)
    out = pd.DataFrame({"teammate":gp.index,"gp":gp.values,"w":w.values,"d":d.values,"l":l.values,
                        "win_pct": ((w.values/np.maximum(gp.values,1))*100).round(1),
                        "ga": ga.values.astype(int)})
    return out.sort_values(["win_pct","gp","ga"], ascending=False)

def nemesis_table(lfact: pd.DataFrame, player: str, season: Optional[int]=None):
    df = lfact.copy()
    if season is not None: df = df[df["season"]==season]
    my = df[df["name"]==player]
    if my.empty: return pd.DataFrame(columns=["opponent","gp","w","d","l","win_pct"])
    opp = my.merge(df, on=["match_id"], suffixes=("_me","_op"))
    opp = opp[opp["team_me"]!=opp["team_op"]]
    opp = opp[opp["name_op"]!=player]
    if opp.empty: return pd.DataFrame(columns=["opponent","gp","w","d","l","win_pct"])
    gp = opp.groupby("name_op").size()
    w = opp[opp["result_me"]=="W"].groupby("name_op").size().reindex(gp.index, fill_value=0)
    d = opp[opp["result_me"]=="D"].groupby("name_op").size().reindex(gp.index, fill_value=0)
    l = opp[opp["result_me"]=="L"].groupby("name_op").size().reindex(gp.index, fill_value=0)
    out = pd.DataFrame({"opponent":gp.index,"gp":gp.values,"w":w.values,"d":d.values,"l":l.values,
                        "win_pct": ((w.values/np.maximum(gp.values,1))*100).round(1)})
    return out.sort_values(["win_pct","gp"], ascending=[True,False])

def duo_global(lfact: pd.DataFrame, min_gp: int = 3):
    df = lfact.copy()
    a = df.merge(df, on=["match_id","team"], suffixes=("_a","_b"))
    a = a[a["name_a"] < a["name_b"]]
    if a.empty: return pd.DataFrame(columns=["pair","gp","win_pct"])
    gp = a.groupby(["name_a","name_b"]).size()
    w = a[a["result_a"]=="W"].groupby(["name_a","name_b"]).size().reindex(gp.index, fill_value=0)
    out = pd.DataFrame({"gp":gp, "w":w}).reset_index()
    out["win_pct"] = ((out["w"]/out["gp"])*100).round(1)
    out["pair"] = out["name_a"]+" + "+out["name_b"]
    out = out[out["gp"]>=min_gp].sort_values(["win_pct","gp"], ascending=[False,False])[["pair","gp","win_pct"]]
    return out

def nemesis_global(lfact: pd.DataFrame, min_gp: int = 3):
    df = lfact.copy()
    a = df.merge(df, on=["match_id"], suffixes=("_a","_b"))
    a = a[a["team_a"]!=a["team_b"]]
    a = a[a["name_a"] < a["name_b"]]
    if a.empty: return pd.DataFrame(columns=["pair","gp","win_pct_vs"])
    gp = a.groupby(["name_a","name_b"]).size()
    w_a = a[a["result_a"]=="W"].groupby(["name_a","name_b"]).size().reindex(gp.index, fill_value=0)
    win_pct_a = ((w_a/gp)*100).round(1)
    out = pd.DataFrame({"gp":gp, "win_pct_vs":win_pct_a}).reset_index()
    out["pair"] = out["name_a"]+" vs "+out["name_b"]
    out = out[out["gp"]>=min_gp].sort_values(["win_pct_vs","gp"], ascending=[True,False])[["pair","gp","win_pct_vs"]]
    return out


# =========================
# FotMob-style pitch (read)
# =========================
def _ensure_positions(team_df: pd.DataFrame, formation: str) -> pd.DataFrame:
    """Ensure line/slot present: if missing, distribute by formation order."""
    rows = team_df.copy()
    lines = formation_to_lines(formation)
    max_slots = max(lines)

    has_positions = rows["line"].notna().sum()>0 and rows["slot"].notna().sum()>0
    if has_positions:
        # Assume slot is absolute in 0..max_slots-1; keep as-is
        return rows

    # Auto layout: GK center back, then distribute remaining
    rows["line"] = np.nan; rows["slot"] = np.nan
    gk = rows[rows.get("is_gk", False)==True]
    if not gk.empty:
        center = max_slots//2
        rows.loc[gk.index[0], ["line","slot"]] = [0, center]
    others = rows.drop(index=(gk.index[0] if not gk.empty else []), errors="ignore")

    cur_line = 1
    filled = [0]*len(lines)
    for idx,_ in others.iterrows():
        if cur_line >= len(lines): cur_line = 1
        slots_here = lines[cur_line]; offset = (max_slots - slots_here)//2
        pos_in_line = filled[cur_line] % slots_here
        rows.loc[idx,"line"] = cur_line
        rows.loc[idx,"slot"] = offset + pos_in_line
        filled[cur_line]+=1; cur_line+=1
    return rows

def _avatar_html(name: str, photo_url: Optional[str]) -> str:
    if photo_url and str(photo_url).strip():
        return f"<div class='avatar'><img src='{photo_url}'/></div>"
    init = "".join([p[0] for p in str(name).split() if p])[:2].upper() or "?"
    return f"<div class='avatar'><div class='init'>{init}</div></div>"

def render_pitch(team_df: pd.DataFrame, formation: str, motm_name: Optional[str], show_stats=True, show_photos=True):
    rows = _ensure_positions(team_df, formation)
    lines = formation_to_lines(formation)
    max_slots = max(lines)
    L = len(lines)  # includes GK line as 0
    # y position: 12% ... 88% across lines
    def y_for(line_i): 
        if L==1: return 50
        return 12 + (76 * (line_i/(L-1)))

    # Build HTML
    html = ["<div class='pitch-wrap'><div class='pitch-bg'></div><div class='pitch-layer'>"]
    for _, r in rows.iterrows():
        line_i = int(r.get("line") if pd.notna(r.get("line")) else 0)
        slot = int(r.get("slot") if pd.notna(r.get("slot")) else max_slots//2)
        # Convert absolute slot to percentage across max_slots grid
        x = (100 * (slot+1)/(max_slots+1))
        y = y_for(line_i)
        name = r.get("name","")
        g = int(r.get("goals") or 0); a = int(r.get("assists") or 0)
        is_m = (motm_name and str(name).strip()==str(motm_name).strip())
        avatar = _avatar_html(name, r.get("photo") if show_photos else None)
        chips = []
        if show_stats and g>0: chips.append(f"<div class='statchip'>⚽ {g}</div>")
        if show_stats and a>0: chips.append(f"<div class='statchip'>🅰️ {a}</div>")
        chips_html = f"<div class='chips'>{''.join(chips)}</div>" if chips else ""
        star = "<div class='motm'>★</div>" if is_m else ""
        html.append(
            f"<div class='spot' style='left:{x}%;top:{y}%;'>"
            f"<div class='avatar'>{avatar[12:-6]}{star}</div>"
            f"<div class='name'>{name}</div>"
            f"{chips_html}"
            f"</div>"
        )
    html.append("</div></div>")
    st.markdown("".join(html), unsafe_allow_html=True)


# =========================
# Admin — tap-to-place editor
# =========================
def _formation_slots(formation: str):
    lines = formation_to_lines(formation)
    max_slots = max(lines)
    offsets = [(max_slots - s)//2 for s in lines]
    return lines, max_slots, offsets

def place_player(team_rows: pd.DataFrame, formation: str, player_id: str, line_i: int, grid_slot: int):
    s = service()
    if not s: st.error("Admin required."); return
    # Clear occupant of that cell
    occ = team_rows[(team_rows["line"]==line_i) & (team_rows["slot"]==grid_slot)]
    payload=[]
    for _,r in occ.iterrows():
        payload.append({"id": str(r["id"]), "line": None, "slot": None})
    # Place selected
    payload.append({"id": str(player_id), "line": int(line_i), "slot": int(grid_slot)})
    s.table("lineups").upsert(payload, on_conflict="id").execute()
    clear_caches()
    st.success("Position saved.")

def set_gk(team_rows: pd.DataFrame, formation: str, player_id: str):
    s = service()
    if not s: st.error("Admin required."); return
    lines, max_slots, _ = _formation_slots(formation)
    center = max_slots//2
    payload=[]
    for _,r in team_rows.iterrows():
        payload.append({"id": str(r["id"]), "is_gk": bool(r["id"]==player_id)})
    payload.append({"id": str(player_id), "line": 0, "slot": center})
    s.table("lineups").upsert(payload, on_conflict="id").execute()
    clear_caches()
    st.success("GK updated.")

def pitch_editor(team_name: str, formation: str, team_rows: pd.DataFrame, key_prefix: str):
    st.caption(f"{team_name} — tap a slot to place the selected player")
    rows = _ensure_positions(team_rows, formation)
    # picker
    rows["label"] = rows.apply(lambda r: f"{r['name']} ({'GK' if r.get('is_gk') else 'Outfield'})", axis=1)
    opts = rows["label"].tolist()
    sel_map = {opts[i]: str(rows.iloc[i]["id"]) for i in range(len(rows))}
    pick = st.selectbox("Player", opts, key=f"{key_prefix}_pick")
    pid = sel_map.get(pick)

    c1,c2 = st.columns(2)
    with c1:
        if st.button("Set as GK", key=f"{key_prefix}_gk"): set_gk(rows, formation, pid); st.rerun()
    with c2:
        if st.button("Clear position", key=f"{key_prefix}_clear"):
            s = service()
            if not s: st.error("Admin required.")
            else:
                s.table("lineups").upsert([{"id": pid, "line": None, "slot": None}], on_conflict="id").execute()
                clear_caches(); st.success("Cleared."); st.rerun()

    # visual grid of slot buttons
    lines, max_slots, offsets = _formation_slots(formation)
    for line_i, slots in enumerate(lines):
        cols = st.columns(max_slots)
        for grid_slot in range(max_slots):
            start = offsets[line_i]; end = start + slots
            if grid_slot < start or grid_slot >= end:
                with cols[grid_slot]: st.write("")
                continue
            with cols[grid_slot]:
                occ = rows[(rows["line"]==line_i) & (rows["slot"]==grid_slot)]
                label = "Empty" if occ.empty else occ.iloc[0]["name"]
                if st.button(label, key=f"{key_prefix}_{line_i}_{grid_slot}", use_container_width=True):
                    place_player(rows, formation, pid, line_i, grid_slot); st.rerun()


# =========================
# UI helpers
# =========================
def chip(name, goals, assists, photo):
    img = f'<img src="{photo}" alt=""/>' if (photo and str(photo).strip()) else ""
    g = int(goals or 0); a = int(assists or 0)
    stats = " · ".join(s for s in [f"⚽ {g}" if g else "", f"🅰️ {a}" if a else ""] if s)
    if stats:
        return f'<span class="chip">{img}<span>{name}</span><span class="meta">{stats}</span></span>'
    return f'<span class="chip">{img}<span>{name}</span></span>'

def header():
    left, right = st.columns([1,1])
    with left: st.title("⚽ Powerleague Stats")
    with right:
        if "is_admin" not in st.session_state: st.session_state["is_admin"]=False
        if st.session_state["is_admin"]:
            st.success("Admin mode", icon="🔐")
            if st.button("Logout"): st.session_state["is_admin"]=False; st.rerun()
        else:
            with st.popover("Admin login"):
                pw = st.text_input("Password", type="password")
                if st.button("Login"):
                    if pw==ADMIN_PASSWORD: st.session_state["is_admin"]=True; st.rerun()
                    else: st.error("Invalid password")


# =========================
# Pages
# =========================
def page_matches():
    players = fetch_players(); matches = fetch_matches(); lineups = fetch_lineups()
    lfact, _ = build_fact(players, matches, lineups)

    st.subheader("Matches")
    if matches.empty:
        st.info("No matches yet."); return

    seasons = sorted(matches["season"].dropna().unique().tolist())
    sel_season = st.selectbox("Season", seasons, index=len(seasons)-1 if seasons else 0)
    opts = matches[matches["season"]==sel_season].sort_values("gw")
    labels = opts.apply(lambda r: f"GW {int(r['gw'])} — {r['team_a']} {r['score_a']}–{r['score_b']} {r['team_b']}", axis=1).tolist()
    id_map = {labels[i]: str(opts.iloc[i]["id"]) for i in range(len(opts))}
    sel_label = st.selectbox("Match", labels)
    mid = id_map[sel_label]

    m = matches[matches["id"].astype(str)==mid].iloc[0]
    show_photos = st.toggle("Show photos", True, key=f"sp_{mid}")

    # Summary banner
    st.markdown(
        f'<div class="banner">'
        f'<div><div class="title">Season {m["season"]} · GW {m["gw"]}</div>'
        f'<div class="sub">{m.get("date") or ""}</div></div>'
        f'<div class="title">{m["team_a"]} {m["score_a"]} – {m["score_b"]} {m["team_b"]}</div>'
        f'</div>', unsafe_allow_html=True
    )
    if m.get("motm_name"):
        st.markdown(f'<div class="banner"><span>Man of the Match</span><span class="badge">🏅 {m["motm_name"]}</span></div>', unsafe_allow_html=True)

    g = lfact[lfact["match_id"]==mid]
    a_rows = g[g["team"]==m["team_a"]].copy()
    b_rows = g[g["team"]==m["team_b"]].copy()

    c1, c2 = st.columns(2)
    with c1:
        st.caption(m["team_a"])
        render_pitch(a_rows, m.get("formation_a") or ("2-1-2-1" if int(m.get("side_count") or 5)==7 else "1-2-1"), m.get("motm_name"), show_stats=True, show_photos=show_photos)
    with c2:
        st.caption(m["team_b"])
        render_pitch(b_rows, m.get("formation_b") or ("2-1-2-1" if int(m.get("side_count") or 5)==7 else "1-2-1"), m.get("motm_name"), show_stats=True, show_photos=show_photos)

    if st.session_state.get("is_admin"):
        with st.expander("✏️ Edit formations & positions"):
            sc = int(m.get("side_count") or 5)
            presets5 = ["1-2-1","1-3","2-2","3-1"]; presets7=["2-1-2-1","3-2-1","2-3-1"]
            presets = presets7 if sc==7 else presets5
            colf1, colf2, colf3 = st.columns([2,2,1])
            fa = colf1.selectbox("Formation (Non-bibs)", presets, index=presets.index(m.get("formation_a") or presets[0]) if (m.get("formation_a") in presets) else 0)
            fb = colf2.selectbox("Formation (Bibs)", presets, index=presets.index(m.get("formation_b") or presets[0]) if (m.get("formation_b") in presets) else 0)
            s = service()
            if colf3.button("Save formations") and s:
                s.table("matches").update({"formation_a": fa, "formation_b": fb}).eq("id", mid).execute()
                clear_caches(); st.success("Formations updated."); st.rerun()

            st.markdown("###### Tap-to-place")
            ea, eb = st.columns(2)
            with ea:
                pitch_editor(m["team_a"], fa or m.get("formation_a") or "1-2-1", a_rows, key_prefix=f"A_{mid}")
            with eb:
                pitch_editor(m["team_b"], fb or m.get("formation_b") or "1-2-1", b_rows, key_prefix=f"B_{mid}")


def page_players():
    players = fetch_players(); matches = fetch_matches(); lineups = fetch_lineups()
    lfact, _ = build_fact(players, matches, lineups)

    st.subheader("Players")
    names = sorted(players["name"].dropna().astype(str).unique().tolist())
    sel = st.selectbox("Select player", [None]+names, index=0)
    if not sel: st.info("Choose a player"); return

    prow = players[players["name"]==sel]
    p = prow.iloc[0].to_dict() if not prow.empty else {"id":None,"name":sel,"photo_url":None,"notes":None}
    mine = lfact[lfact["name"]==sel].copy()
    agg_all = player_agg(lfact); me = agg_all[agg_all["name"]==sel]
    me_row = me.iloc[0] if not me.empty else None

    col1, col2 = st.columns([1,2])
    with col1:
        st.image(p.get("photo_url") or "https://placehold.co/240x240?text=No+Photo", width=180)
        st.markdown(f"### {sel}")
        if p.get("notes"): st.caption(p["notes"])
        if st.session_state.get("is_admin") and p.get("id"):
            up = st.file_uploader("Update photo (HEIC/JPG/PNG)", type=["heic","HEIC","jpg","jpeg","png"])
            if up and st.button("Upload"):
                url = upload_avatar(up, sel)
                if url:
                    s = service(); s.table("players").update({"photo_url":url}).eq("id", p["id"]).execute()
                    clear_caches(); st.success("Photo updated."); st.rerun()
    with col2:
        if me_row is not None:
            st.markdown(
                f'<div class="card"><div class="kv">'
                f'<div><span class="k">GP</span> <b>{int(me_row["gp"])}</b></div>'
                f'<div><span class="k">W-D-L</span> <b>{int(me_row["w"])}-{int(me_row["d"])}-{int(me_row["l"])}</b></div>'
                f'<div><span class="k">Win%</span> <b>{me_row["win_pct"]}%</b></div>'
                f'<div><span class="k">Goals</span> <b>{int(me_row["goals"])}</b></div>'
                f'<div><span class="k">Assists</span> <b>{int(me_row["assists"])}</b></div>'
                f'<div><span class="k">G+A</span> <b>{int(me_row["ga"])}</b></div>'
                f'<div><span class="k">G/PG</span> <b>{me_row["g_pg"]}</b></div>'
                f'<div><span class="k">A/PG</span> <b>{me_row["a_pg"]}</b></div>'
                f'<div><span class="k">Team Contrib%</span> <b>{me_row["team_contrib_pct"]}%</b></div>'
                f'</div></div>', unsafe_allow_html=True
            )

        st.markdown("#### Duos & Nemesis")
        colduo, colnem = st.columns(2)
        with colduo:
            st.caption("Best Teammates (by Win% ≥ 3 GP)")
            duo = duo_table(lfact, sel)
            if duo.empty: st.caption("—")
            else:
                d2 = duo[duo["gp"]>=3].head(5)
                for _,r in d2.iterrows():
                    st.write(f"👥 {r['teammate']} — {r['win_pct']}% · GP {int(r['gp'])}")
        with colnem:
            st.caption("Nemesis (lowest Win% vs, ≥ 3 GP)")
            nem = nemesis_table(lfact, sel)
            if nem.empty: st.caption("—")
            else:
                n2 = nem[nem["gp"]>=3].head(5)
                for _,r in n2.iterrows():
                    st.write(f"⚔️ {r['opponent']} — {r['win_pct']}% · GP {int(r['gp'])}")

        st.markdown("#### Recent Games")
        if mine.empty: st.caption("No games.")
        else:
            mine["season"] = pd.to_numeric(mine.get("season"), errors="coerce")
            mine["gw"] = pd.to_numeric(mine.get("gw"), errors="coerce")
            for _, r in mine.sort_values(["season","gw"], ascending=[False,False]).head(10).iterrows():
                st.write(f"S{int(r['season'])} GW{int(r['gw'])} · {r['team_a']} {r['score_a']}–{r['score_b']} {r['team_b']} — {r['team']}")
                st.caption(f"⚽ {int(r.get('goals') or 0)} · 🅰️ {int(r.get('assists') or 0)} · Result: {r['result']}")


def page_stats():
    players = fetch_players(); matches = fetch_matches(); lineups = fetch_lineups()
    lfact, _ = build_fact(players, matches, lineups)
    st.subheader("Stats")

    c1,c2,c3,c4 = st.columns(4)
    seasons = sorted(matches["season"].dropna().unique().tolist()) if not matches.empty else []
    season = c1.selectbox("Season", [None]+seasons, index=0)
    min_games = int(c2.number_input("Min games", min_value=0, value=0, step=1))
    last_x = c3.selectbox("Last X GWs", [None,3,5,10], index=0)
    top_n = int(c4.number_input("Rows (Top N)", min_value=5, value=10, step=1))

    metric = st.selectbox(
        "Metric",
        ["G+A","Goals","Assists","Goals per Game","Assists per Game","G+A per Game","Win %","Team Contribution %","MOTM","Best Duos","Nemesis Pairs"],
        index=0
    )
    show_photos = st.toggle("Show photos", True)

    agg = player_agg(lfact, season=season, min_games=min_games, last_gw=last_x)

    def render_row(name, value, photo):
        cA,cB = st.columns([6,1])
        with cA: 
            st.markdown(chip(name, 0, 0, photo), unsafe_allow_html=True)
        with cB:
            if isinstance(value, float): value = round(value, 2)
            st.metric("Value", value)

    if metric == "MOTM":
        aw = fetch_awards(); motm = aw[aw["type"]=="MOTM"].copy()
        if season is not None: motm = motm[motm["season"]==season]
        df = motm.groupby("player_name").size().reset_index(name="MOTM").sort_values("MOTM", ascending=False)
        df = df.merge(players[["name","photo_url"]], left_on="player_name", right_on="name", how="left")
        for _, r in df.head(top_n).iterrows():
            render_row(r["player_name"], r["MOTM"], r["photo_url"] if show_photos else None)

    elif metric == "Best Duos":
        dg = duo_global(lfact, min_gp=3)
        if season is not None:
            # filter to season by rebuilding on season slice
            dg = duo_global(lfact[lfact["season"]==season], min_gp=3)
        if dg.empty: st.caption("No duo data.")
        else:
            for _, r in dg.head(top_n).iterrows():
                st.write(f"👥 {r['pair']} — {r['win_pct']}% · GP {int(r['gp'])}")

    elif metric == "Nemesis Pairs":
        ng = nemesis_global(lfact, min_gp=3)
        if season is not None:
            ng = nemesis_global(lfact[lfact["season"]==season], min_gp=3)
        if ng.empty: st.caption("No nemesis data.")
        else:
            for _, r in ng.head(top_n).iterrows():
                st.write(f"⚔️ {r['pair']} — {r['win_pct_vs']}% · GP {int(r['gp'])}")

    else:
        # Player metrics
        df = agg.copy()
        if metric == "Goals":
            df = df.sort_values(["goals","ga","assists"], ascending=False)
            for _, r in df.head(top_n).iterrows(): render_row(r["name"], int(r["goals"]), r["photo"] if show_photos else None)
        elif metric == "Assists":
            df = df.sort_values(["assists","ga","goals"], ascending=False)
            for _, r in df.head(top_n).iterrows(): render_row(r["name"], int(r["assists"]), r["photo"] if show_photos else None)
        elif metric == "G+A":
            df = df.sort_values(["ga","goals","assists"], ascending=False)
            for _, r in df.head(top_n).iterrows(): render_row(r["name"], int(r["ga"]), r["photo"] if show_photos else None)
        elif metric == "G+A per Game":
            df = df.sort_values(["ga_pg","ga","goals"], ascending=False)
            for _, r in df.head(top_n).iterrows(): render_row(r["name"], r["ga_pg"], r["photo"] if show_photos else None)
        elif metric == "Goals per Game":
            df = df.sort_values(["g_pg","goals"], ascending=False)
            for _, r in df.head(top_n).iterrows(): render_row(r["name"], r["g_pg"], r["photo"] if show_photos else None)
        elif metric == "Assists per Game":
            df = df.sort_values(["a_pg","assists"], ascending=False)
            for _, r in df.head(top_n).iterrows(): render_row(r["name"], r["a_pg"], r["photo"] if show_photos else None)
        elif metric == "Win %":
            df = df.sort_values(["win_pct","ga","goals"], ascending=False)
            for _, r in df.head(top_n).iterrows(): render_row(r["name"], r["win_pct"], r["photo"] if show_photos else None)
        elif metric == "Team Contribution %":
            df = df.sort_values(["team_contrib_pct","ga","goals"], ascending=False)
            for _, r in df.head(top_n).iterrows(): render_row(r["name"], r["team_contrib_pct"], r["photo"] if show_photos else None)


def page_awards():
    aw = fetch_awards()
    st.subheader("Awards")
    if st.session_state.get("is_admin"):
        with st.form("add_award"):
            season = st.number_input("Season", min_value=1, value=1, step=1)
            month = st.number_input("Month (1-12, POTM)", min_value=1, max_value=12, value=1, step=1)
            atype = st.selectbox("Type", ["MOTM","POTM"])
            gw = st.number_input("GW (for MOTM)", min_value=1, value=1, step=1)
            player_name = st.text_input("Player name")
            notes = st.text_input("Notes")
            if st.form_submit_button("Add"):
                s = service()
                if not s: st.error("Admin required.")
                else:
                    s.table("awards").insert({
                        "season": int(season),
                        "month": int(month) if atype=="POTM" else None,
                        "type": atype,
                        "gw": int(gw) if atype=="MOTM" else None,
                        "player_name": player_name,
                        "notes": notes
                    }).execute()
                    clear_caches(); st.success("Saved."); st.rerun()


# =========================
# Avatar upload (players)
# =========================
def upload_avatar(file, name):
    s = service()
    if not s: st.error("Admin required."); return None
    img = Image.open(file).convert("RGBA")
    buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
    key = f"{name.lower().replace(' ','_')}.png"
    try:
        s.storage.from_(AVATAR_BUCKET).upload(key, buf.getvalue(), {"content-type":"image/png","upsert":"true"})
    except Exception:
        try: s.storage.from_(AVATAR_BUCKET).remove([key])
        except Exception: pass
        s.storage.from_(AVATAR_BUCKET).upload(key, buf.getvalue(), {"content-type":"image/png"})
    # public URL
    return f"{SUPABASE_URL}/storage/v1/object/public/{AVATAR_BUCKET}/{key}"


# =========================
# Router
# =========================
def run_app():
    header()
    Page = getattr(st, "Page", None); nav = getattr(st, "navigation", None)
    pages = {
        "Matches": page_matches,
        "Players": page_players,
        "Stats": page_stats,
        "Awards": page_awards,
    }
    if Page and nav:
        sections = {"Main":[Page(page_matches, title="Matches", icon="📋"),
                            Page(page_players, title="Players", icon="👤"),
                            Page(page_stats, title="Stats", icon="📊"),
                            Page(page_awards, title="Awards", icon="🏆")]}
        n = nav(sections); n.run()
    else:
        sel = st.sidebar.radio("Go to", list(pages.keys()), index=0)
        pages[sel]()

if __name__ == "__main__":
    run_app()
