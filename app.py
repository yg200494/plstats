# app.py — Powerleague Stats (v3)
# - FotMob-style pitch (centered singles, GK bottom center, gold MOTM star)
# - Formation change + Auto-arrange per team
# - Player profile redesign with streaks, tabs, last N games default=5
# - Stats page: single dropdown, global filters (season, min games, last N GWs, top N) apply to ALL metrics,
#   including Duos & Nemesis, shown as sortable tables (no photos on Stats page)

import io
from typing import Optional, List, Dict, Tuple

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
  --pitch:#114f39; --line:#a9e0bd; --text:#e6edf3; --accent:#22c55e;
}
html,body,[data-testid="stAppViewContainer"]{
  background:linear-gradient(180deg,var(--bg1) 0%, var(--bg2) 100%);
  color:var(--text);
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
.badge{background:var(--accent);color:#062616;font-weight:700;border-radius:999px;padding:6px 12px;font-size:12px;border:1px solid rgba(0,0,0,.25)}
.hr{height:1px; background:rgba(255,255,255,.10); margin:10px 0 16px}

/* Pitch */
.pitch{position:relative;width:100%;padding-top:150%;border-radius:20px;overflow:hidden;
       background:var(--pitch);box-shadow:0 6px 18px rgba(0,0,0,.35);border:1px solid rgba(255,255,255,.18)}
.pitch::before{
  content:"";position:absolute;inset:0;
  background:repeating-linear-gradient(180deg, rgba(255,255,255,.03) 0 6%, rgba(0,0,0,0) 6% 12%);
}
.pitch svg{position:absolute;inset:0}

/* Players */
.spot{position:absolute;transform:translate(-50%,-50%);display:flex;flex-direction:column;align-items:center}
.avatar{position:relative;width:76px;height:76px;border-radius:50%;overflow:hidden;border:2px solid rgba(255,255,255,.55);box-shadow:0 2px 10px rgba(0,0,0,.45);background:#fff}
.avatar img{width:100%;height:100%;object-fit:cover}
.init{width:100%;height:100%;display:flex;align-items:center;justify-content:center;background:#fff;color:#1f2937;font-weight:800;font-size:26px;letter-spacing:.5px}
.motm{position:absolute;top:-8px;right:-8px;background:gold;color:#000;border-radius:50%;padding:2px 4px;font-size:12px;border:1px solid rgba(0,0,0,.25)}
.name{margin-top:4px;font-size:13px;font-weight:700;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,.6);white-space:nowrap}
.chips{display:flex;gap:4px;margin-top:2px}
.statchip{background:#fff;color:#111;padding:2px 6px;border-radius:10px;font-size:12px;border:1px solid rgba(0,0,0,.08)}
/* Editor */
.slotBtn{width:100%;height:44px;border-radius:10px;border:1px dashed rgba(255,255,255,.25);background:rgba(255,255,255,.04)}
.kv{display:flex;gap:10px;flex-wrap:wrap}
.kv .k{opacity:.8}

/* Tabs tweak */
[data-baseweb="tab-list"]{gap:4px}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =========================
# Secrets & clients
# =========================
REQ = ["SUPABASE_URL","SUPABASE_ANON_KEY","ADMIN_PASSWORD","AVATAR_BUCKET"]
for k in REQ:
    if k not in st.secrets:
        st.error(f"Missing secret: {k}"); st.stop()

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
@st.cache_data(ttl=20)
def fetch_players() -> pd.DataFrame:
    return pd.DataFrame(sb_public.table("players").select("*").execute().data or [])

@st.cache_data(ttl=20)
def fetch_matches() -> pd.DataFrame:
    res = sb_public.table("matches").select("*").order("season").order("gw").execute().data or []
    return pd.DataFrame(res)

@st.cache_data(ttl=20)
def fetch_lineups() -> pd.DataFrame:
    return pd.DataFrame(sb_public.table("lineups").select("*").execute().data or [])

@st.cache_data(ttl=20)
def fetch_awards() -> pd.DataFrame:
    return pd.DataFrame(sb_public.table("awards").select("*").execute().data or [])

def clear_caches():
    fetch_players.clear(); fetch_matches.clear(); fetch_lineups.clear(); fetch_awards.clear()

# =========================
# Helpers: formation, facts, streaks
# =========================
def formation_to_lines(form: str) -> List[int]:
    """Outfield rows (top->bottom). GK is drawn as final row."""
    if not form: return [1,2,1]
    parts = [int(x) for x in str(form).split("-") if x.strip().isdigit()]
    return parts if parts else [1,2,1]

def build_fact(players, matches, lineups):
    players = players.copy(); matches = matches.copy(); lineups = lineups.copy()
    for df in (players, matches, lineups):
        if "id" in df.columns: df["id"] = df["id"].astype(str)

    cols = ["id","team_a","team_b","score_a","score_b","season","gw","date","is_draw","motm_name","formation_a","formation_b","side_count","notes"]
    mi = matches[[c for c in cols if c in matches.columns]].rename(columns={"id":"match_id"})
    l = lineups.merge(mi, on="match_id", how="left")

    def pick(df, base):
        if base in df.columns: return df[base]
        if f"{base}_x" in df.columns: return df[f"{base}_x"]
        if f"{base}_y" in df.columns: return df[f"{base}_y"]
        return pd.Series([np.nan]*len(df), index=df.index)

    l["season"] = pd.to_numeric(pick(l,"season"), errors="coerce").astype("Int64")
    l["gw"] = pd.to_numeric(pick(l,"gw"), errors="coerce").astype("Int64")
    for c in ["season_x","season_y","gw_x","gw_y"]:
        if c in l.columns: del l[c]

    l["goals"] = pd.to_numeric(l.get("goals"), errors="coerce").fillna(0).astype(int)
    l["assists"] = pd.to_numeric(l.get("assists"), errors="coerce").fillna(0).astype(int)
    l["line"] = pd.to_numeric(l.get("line"), errors="coerce")
    l["slot"] = pd.to_numeric(l.get("slot"), errors="coerce")

    l["side"] = np.where(l["team"]==l["team_a"], "A", np.where(l["team"]==l["team_b"], "B", None))
    l["team_goals"] = np.where(l["side"]=="A", l["score_a"], np.where(l["side"]=="B", l["score_b"], np.nan))
    l["opp_goals"]  = np.where(l["side"]=="A", l["score_b"], np.where(l["side"]=="B", l["score_a"], np.nan))
    l["team_goals"] = pd.to_numeric(l["team_goals"], errors="coerce")
    l["opp_goals"]  = pd.to_numeric(l["opp_goals"],  errors="coerce")

    l["result"] = np.where(l["is_draw"]==True, "D", np.where(l["team_goals"]>l["opp_goals"], "W", "L"))
    l["ga"] = (l["goals"] + l["assists"]).astype(int)

    p = players.rename(columns={"id":"player_id"})
    if not p.empty:
        l = l.merge(p[["player_id","name","photo_url"]], on="player_id", how="left")
    l["name"] = l["player_name"].where(l["player_name"].notna() & (l["player_name"].astype(str).str.strip()!=""), l.get("name"))
    l["name"] = l["name"].fillna("Unknown")
    l["photo"] = l.get("photo_url")

    return l, matches

def compute_streak(series_vals, predicate):
    """Length of current consecutive True from the end."""
    count=0
    for v in list(series_vals)[::-1]:
        if predicate(v): count+=1
        else: break
    return count

# ---------- Aggregations ----------
def player_agg(l, season=None, min_games=0, last_gw=None):
    df = l.copy()
    if season is not None: df = df[df["season"]==season]
    if last_gw and last_gw>0:
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

def df_filter_by(df: pd.DataFrame, season=None, last_gw=None) -> pd.DataFrame:
    d = df.copy()
    if season is not None: d = d[d["season"]==season]
    if last_gw and last_gw>0:
        max_gw = pd.to_numeric(d["gw"], errors="coerce").max()
        if pd.notna(max_gw): d = d[pd.to_numeric(d["gw"], errors="coerce") >= (int(max_gw)-int(last_gw)+1)]
    return d

def duo_global(lfact: pd.DataFrame, min_gp: int = 3, season=None, last_gw=None):
    df = df_filter_by(lfact, season, last_gw)
    a = df.merge(df, on=["match_id","team"], suffixes=("_a","_b"))
    a = a[a["name_a"] < a["name_b"]]
    if a.empty: return pd.DataFrame(columns=["pair","gp","w","win_pct"])
    gp = a.groupby(["name_a","name_b"]).size()
    w = a[a["result_a"]=="W"].groupby(["name_a","name_b"]).size().reindex(gp.index, fill_value=0)
    out = pd.DataFrame({"gp":gp, "w":w}).reset_index()
    out["win_pct"] = ((out["w"]/out["gp"])*100).round(1)
    out["pair"] = out["name_a"]+" + "+out["name_b"]
    out = out[out["gp"]>=min_gp].sort_values(["win_pct","gp"], ascending=[False,False])[["pair","gp","w","win_pct"]]
    return out

def nemesis_global(lfact: pd.DataFrame, min_gp: int = 3, season=None, last_gw=None):
    df = df_filter_by(lfact, season, last_gw)
    a = df.merge(df, on=["match_id"], suffixes=("_a","_b"))
    a = a[a["team_a"]!=a["team_b"]]
    a = a[a["name_a"] < a["name_b"]]
    if a.empty: return pd.DataFrame(columns=["pair","gp","win_pct_vs"])
    gp = a.groupby(["name_a","name_b"]).size()
    w_a = a[a["result_a"]=="W"].groupby(["name_a","name_b"]).size().reindex(gp.index, fill_value=0)
    out = pd.DataFrame({"gp":gp, "win_a":w_a}).reset_index()
    out["win_pct_vs"] = ((out["win_a"]/out["gp"])*100).round(1)
    out["pair"] = out["name_a"]+" vs "+out["name_b"]
    out = out[out["gp"]>=min_gp].sort_values(["win_pct_vs","gp"], ascending=[True,False])[["pair","gp","win_pct_vs"]]
    return out

# =========================
# Pitch rendering & editor
# =========================
def _avatar_html(name: str, photo_url: Optional[str]) -> str:
    if photo_url and str(photo_url).strip():
        return f"<div class='avatar'><img src='{photo_url}'/></div>"
    init = "".join([p[0] for p in str(name).split() if p])[:2].upper() or "?"
    return f"<div class='avatar'><div class='init'>{init}</div></div>"

def _pitch_svg() -> str:
    return f"""
    <svg viewBox="0 0 100 150" preserveAspectRatio="none">
      <rect x="2" y="3" width="96" height="144" fill="none" stroke="{ '#a9e0bd' }" stroke-width="1.2" rx="2"/>
      <rect x="12" y="3" width="76" height="24" fill="none" stroke="{ '#a9e0bd' }" stroke-width="1.2"/>
      <rect x="30" y="3" width="40" height="12" fill="none" stroke="{ '#a9e0bd' }" stroke-width="1.2"/>
      <rect x="12" y="123" width="76" height="24" fill="none" stroke="{ '#a9e0bd' }" stroke-width="1.2"/>
      <rect x="30" y="135" width="40" height="12" fill="none" stroke="{ '#a9e0bd' }" stroke-width="1.2"/>
      <line x1="2" y1="75" x2="98" y2="75" stroke="{ '#a9e0bd' }" stroke-width="1.2"/>
      <circle cx="50" cy="75" r="17" fill="none" stroke="{ '#a9e0bd' }" stroke-width="1.2"/>
    </svg>
    """

def _ensure_positions(team_df: pd.DataFrame, formation: str) -> pd.DataFrame:
    """Guarantee line/slot (GK = last row center). If already set, keep."""
    rows = team_df.copy()
    parts = formation_to_lines(formation)
    max_slots = max(parts+[1])

    have = rows["line"].notna().sum()>0 and rows["slot"].notna().sum()>0
    if have:  # keep existing, but snap singles visually via offsets during Auto-arrange
        return rows

    rows["line"] = np.nan; rows["slot"] = np.nan
    gk = rows[rows.get("is_gk", False)==True]
    others = rows.drop(index=(gk.index[0] if not gk.empty else []), errors="ignore")

    # Distribute outfield across lines; center singles via offset
    cur=0; filled=[0]*len(parts)
    for idx,_ in others.iterrows():
        if cur>=len(parts): cur=0
        slots=parts[cur]; offset=(max_slots-slots)//2  # centers 1/2/3 across grid
        pos=filled[cur] % slots
        rows.loc[idx,["line","slot"]] = [cur, offset+pos]
        filled[cur]+=1; cur+=1

    # GK bottom center
    if not gk.empty:
        center = max_slots//2
        rows.loc[gk.index[0],["line","slot"]] = [len(parts), center]
    return rows

def render_pitch(team_df: pd.DataFrame, formation: str, motm_name: Optional[str], show_stats=True, show_photos=True):
    rows = _ensure_positions(team_df, formation)
    parts = formation_to_lines(formation)
    max_slots = max(parts+[1])
    total_rows = len(parts)+1  # GK

    def y_for(i):
        return 6 + (88 * (i/(total_rows-1))) if total_rows>1 else 50

    html = [f"<div class='pitch'>{_pitch_svg()}"]

    for _, r in rows.iterrows():
        row_idx = int(r.get("line") if pd.notna(r.get("line")) else len(parts))
        if bool(r.get("is_gk")): row_idx = len(parts)
        slot = int(r.get("slot") if pd.notna(r.get("slot")) else max_slots//2)
        x = (100 * (slot+1)/(max_slots+1)); y = y_for(row_idx)
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
            f"{avatar.replace('</div>', f'{star}</div>', 1)}"
            f"<div class='name'>{name}</div>"
            f"{chips_html}"
            f"</div>"
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)

# ---------- Editor ops ----------
def update_lineup_row(row_id: str, fields: Dict):
    s = service()
    if not s: st.error("Admin required."); return
    s.table("lineups").update(fields).eq("id", str(row_id)).execute()

def add_players_to_team(match_id: str, team_label: str, player_ids: List[str], players_df: pd.DataFrame):
    s = service()
    if not s: st.error("Admin required."); return
    payload=[]
    for pid in player_ids:
        prow = players_df[players_df["id"].astype(str)==str(pid)]
        if prow.empty: continue
        payload.append({
            "match_id": match_id, "team": team_label,
            "player_id": str(pid), "player_name": prow.iloc[0]["name"],
            "is_gk": False, "goals": 0, "assists": 0,
            "line": None, "slot": None, "position": None
        })
    if payload:
        s.table("lineups").insert(payload).execute()
        clear_caches()

def remove_lineup_row(row_id: str):
    s = service()
    if not s: st.error("Admin required."); return
    s.table("lineups").delete().eq("id", str(row_id)).execute()
    clear_caches()

def auto_arrange_team(team_rows: pd.DataFrame, formation: str):
    """Snap current team to formation grid: GK bottom center, centered lines."""
    s = service()
    if not s: st.error("Admin required."); return
    rows = _ensure_positions(team_rows, formation)  # gives ideal centered grid
    parts = formation_to_lines(formation)
    max_slots = max(parts+[1])

    # GK
    rid_gk = rows.loc[rows["is_gk"]==True, "id"]
    if not rid_gk.empty:
        s.table("lineups").update({"line": len(parts), "slot": max_slots//2}).eq("id", str(rid_gk.iloc[0])).execute()

    # Others
    others = rows[rows["is_gk"]!=True].copy()
    cur=0; filled=[0]*len(parts)
    for _, r in others.iterrows():
        if cur>=len(parts): cur=0
        slots=parts[cur]; offset=(max_slots-slots)//2
        pos=filled[cur] % slots
        s.table("lineups").update({"line": cur, "slot": offset+pos}).eq("id", str(r["id"])).execute()
        filled[cur]+=1; cur+=1
    clear_caches()

def pitch_editor(team_name: str, formation: str, match_id: str, team_rows: pd.DataFrame, key_prefix: str):
    st.caption(f"{team_name} — tap a slot to place the selected player (GK row included)")
    rows = _ensure_positions(team_rows, formation)

    rows["label"] = rows.apply(lambda r: f"{r['name']} ({'GK' if r.get('is_gk') else 'Outfield'})", axis=1)
    opts = rows["label"].tolist()
    id_by_label = {opts[i]: str(rows.iloc[i]["id"]) for i in range(len(rows))}
    pick = st.selectbox("Player", opts, key=f"{key_prefix}_pick")
    rid = id_by_label.get(pick)

    c1,c2,c3,c4 = st.columns(4)
    with c1:
        if st.button("Set as GK", key=f"{key_prefix}_gk"):
            parts = formation_to_lines(formation); center = (max(parts+[1])//2)
            s = service()
            if not s: st.error("Admin required.")
            else:
                s.table("lineups").update({"is_gk": False}).eq("match_id", match_id).eq("team", team_name).execute()
                s.table("lineups").update({"is_gk": True, "line": len(parts), "slot": center}).eq("id", rid).execute()
                clear_caches(); st.success("GK updated."); st.rerun()
    with c2:
        if st.button("Clear position", key=f"{key_prefix}_clear"):
            update_lineup_row(rid, {"line": None, "slot": None}); clear_caches(); st.rerun()
    with c3:
        if st.button("Remove from lineup", key=f"{key_prefix}_rm"):
            remove_lineup_row(rid); st.rerun()
    with c4:
        if st.button("Auto-arrange to formation", key=f"{key_prefix}_auto"):
            auto_arrange_team(team_rows, formation); st.rerun()

    # grid buttons
    parts = formation_to_lines(formation)
    max_slots = max(parts+[1]); total_rows = len(parts)+1
    offsets = [(max_slots - s_)//2 for s_ in parts] + [0]  # GK center
    for line_idx in range(total_rows):
        slots = (parts[line_idx] if line_idx < len(parts) else 1)
        cols = st.columns(max_slots)
        start = offsets[line_idx]; end = start + (slots if line_idx < len(parts) else 1)
        for grid_slot in range(max_slots):
            with cols[grid_slot]:
                if grid_slot < start or grid_slot >= end:
                    st.write(""); continue
                occ = rows[(rows["line"]==line_idx) & (rows["slot"]==grid_slot)]
                label = "Empty" if occ.empty else occ.iloc[0]["name"]
                if st.button(label, key=f"{key_prefix}_{line_idx}_{grid_slot}", use_container_width=True):
                    for _, r in occ.iterrows():
                        update_lineup_row(r["id"], {"line": None, "slot": None})
                    update_lineup_row(rid, {"line": int(line_idx), "slot": int(grid_slot)})
                    clear_caches(); st.success("Position saved."); st.rerun()

# =========================
# UI helpers
# =========================
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
# Add/Edit Matches & Fixtures
# =========================
def add_match_wizard():
    st.markdown("### ➕ Add / Update Match")
    with st.form("add_match"):
        c1,c2,c3 = st.columns(3)
        season = int(c1.number_input("Season", min_value=1, value=1))
        gw = int(c2.number_input("Gameweek", min_value=1, value=1))
        date = c3.date_input("Date")
        c4,c5 = st.columns(2)
        side_count = c4.selectbox("Side count", [5,7], index=0)
        team_a, team_b = "Non-bibs", "Bibs"
        sc1, sc2 = st.columns(2)
        score_a = int(sc1.number_input(f"{team_a} goals", min_value=0, value=0))
        score_b = int(sc2.number_input(f"{team_b} goals", min_value=0, value=0))
        motm_name = st.text_input("Man of the Match (optional)")
        fa, fb = st.columns(2)
        presets5 = ["1-2-1","1-3","2-2","3-1"]
        presets7 = ["2-1-2-1","3-2-1","2-3-1"]
        form_a = fa.selectbox("Formation (Non-bibs)", presets7 if side_count==7 else presets5, index=0)
        form_b = fb.selectbox("Formation (Bibs)", presets7 if side_count==7 else presets5, index=0)
        notes = st.text_area("Notes", "")
        submit = st.form_submit_button("Save match")
        if submit:
            s = service()
            if not s: st.error("Admin required.")
            else:
                payload = {
                    "season": season, "gw": gw, "side_count": side_count,
                    "team_a": team_a, "team_b": team_b,
                    "score_a": score_a, "score_b": score_b, "is_draw": (score_a==score_b),
                    "date": str(date), "motm_name": motm_name or None,
                    "formation_a": form_a, "formation_b": form_b, "notes": notes or None
                }
                s.table("matches").upsert(payload, on_conflict="season,gw").execute()
                clear_caches(); st.success("Match saved."); st.rerun()

def fixtures_admin_table():
    st.markdown("### 📅 All Fixtures (Admin inline edit)")
    matches = fetch_matches()
    if matches.empty:
        st.info("No matches yet."); return
    s = service()
    if not s:
        st.info("Login as admin to edit fixtures.")
        st.dataframe(matches.sort_values(["season","gw"])[["season","gw","date","team_a","score_a","score_b","team_b","motm_name"]], use_container_width=True, hide_index=True)
        return
    for _, m in matches.sort_values(["season","gw"]).iterrows():
        with st.expander(f"S{int(m['season'])} GW{int(m['gw'])} — {m['team_a']} {m['score_a']}–{m['score_b']} {m['team_b']}"):
            d1,d2,d3,d4 = st.columns([2,2,1,1])
            date = d1.date_input("Date", value=pd.to_datetime(m["date"]).date() if pd.notna(m["date"]) else pd.Timestamp.today().date(), key=f"dt_{m['id']}")
            sa = int(d2.number_input("Score A", min_value=0, value=int(m.get("score_a") or 0), key=f"sa_{m['id']}"))
            sb = int(d3.number_input("Score B", min_value=0, value=int(m.get("score_b") or 0), key=f"sb_{m['id']}"))
            motm = d4.text_input("MOTM", value=m.get("motm_name") or "", key=f"mm_{m['id']}")
            fa, fb = st.columns(2)
            presets = ["1-2-1","1-3","2-2","3-1","2-1-2-1","3-2-1","2-3-1"]
            idxa = presets.index(m.get("formation_a")) if m.get("formation_a") in presets else 0
            idxb = presets.index(m.get("formation_b")) if m.get("formation_b") in presets else 0
            forma = fa.selectbox("Formation A", presets, index=idxa, key=f"fa_{m['id']}")
            formb = fb.selectbox("Formation B", presets, index=idxb, key=f"fb_{m['id']}")
            if st.button("Save", key=f"fixsave_{m['id']}"):
                s.table("matches").update({
                    "date": str(date), "score_a": sa, "score_b": sb, "is_draw": (sa==sb),
                    "motm_name": motm or None, "formation_a": forma, "formation_b": formb
                }).eq("id", m["id"]).execute()
                clear_caches(); st.success("Updated."); st.rerun()

# =========================
# Pages
# =========================
def page_matches():
    players = fetch_players(); matches = fetch_matches(); lineups = fetch_lineups()
    lfact, _ = build_fact(players, matches, lineups)

    st.subheader("Matches")
    if st.session_state.get("is_admin"):
        with st.expander("Admin: Add / Update Match", expanded=False):
            add_match_wizard()
        with st.expander("Admin: All Fixtures (edit dates/scores/formations/MOTM)", expanded=False):
            fixtures_admin_table()

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

    # live formation pickers + save + auto-arrange
    if st.session_state.get("is_admin"):
        presets5 = ["1-2-1","1-3","2-2","3-1"]
        presets7 = ["2-1-2-1","3-2-1","2-3-1"]
        preset_list = presets7 if int(m.get("side_count") or 5)==7 else presets5
        colf1, colf2, colf3, colf4 = st.columns([2,2,1,1])
        fa = colf1.selectbox("Formation (Non-bibs)", preset_list, index=(preset_list.index(m.get("formation_a")) if m.get("formation_a") in preset_list else 0))
        fb = colf2.selectbox("Formation (Bibs)", preset_list, index=(preset_list.index(m.get("formation_b")) if m.get("formation_b") in preset_list else 0))
        if colf3.button("Save"):
            s = service()
            if not s: st.error("Admin required.")
            else:
                s.table("matches").update({"formation_a": fa, "formation_b": fb}).eq("id", mid).execute()
                clear_caches(); st.success("Formations updated."); st.rerun()
        if colf4.button("Auto-arrange both"):
            auto_arrange_team(a_rows, fa); auto_arrange_team(b_rows, fb); st.rerun()
    else:
        fa = m.get("formation_a") or ("2-1-2-1" if int(m.get("side_count") or 5)==7 else "1-2-1")
        fb = m.get("formation_b") or ("2-1-2-1" if int(m.get("side_count") or 5)==7 else "1-2-1")

    c1, c2 = st.columns(2)
    with c1:
        st.caption(m["team_a"])
        render_pitch(a_rows, fa, m.get("motm_name"), show_stats=True, show_photos=show_photos)
    with c2:
        st.caption(m["team_b"])
        render_pitch(b_rows, fb, m.get("motm_name"), show_stats=True, show_photos=show_photos)

    if st.session_state.get("is_admin"):
        with st.expander("✏️ Edit lineups & positions", expanded=False):
            pa, pb = st.columns(2)
            players = fetch_players()
            with pa:
                st.markdown("**Non-bibs: Add players**")
                current_ids = set(a_rows["player_id"].astype(str).tolist())
                options = players[~players["id"].astype(str).isin(current_ids)]
                sel = st.multiselect("Choose players", options["name"].tolist(), key=f"addA_{mid}")
                name_to_id = dict(zip(options["name"], options["id"].astype(str)))
                if st.button("Add to Non-bibs", key=f"pushA_{mid}"):
                    add_players_to_team(mid, m["team_a"], [name_to_id[n] for n in sel], players); st.rerun()
            with pb:
                st.markdown("**Bibs: Add players**")
                current_ids = set(b_rows["player_id"].astype(str).tolist())
                options = players[~players["id"].astype(str).isin(current_ids)]
                sel = st.multiselect("Choose players", options["name"].tolist(), key=f"addB_{mid}")
                name_to_id = dict(zip(options["name"], options["id"].astype(str)))
                if st.button("Add to Bibs", key=f"pushB_{mid}"):
                    add_players_to_team(mid, m["team_b"], [name_to_id[n] for n in sel], players); st.rerun()

            st.markdown("---")
            ea, eb = st.columns(2)
            with ea:
                pitch_editor(m["team_a"], fa, mid, a_rows, key_prefix=f"A_{mid}")
            with eb:
                pitch_editor(m["team_b"], fb, mid, b_rows, key_prefix=f"B_{mid}")

def page_players():
    players = fetch_players(); matches = fetch_matches(); lineups = fetch_lineups(); awards = fetch_awards()
    lfact, _ = build_fact(players, matches, lineups)

    st.subheader("Players")
    names = sorted(players["name"].dropna().astype(str).unique().tolist())
    sel = st.selectbox("Select player", [None]+names, index=0)
    if not sel: st.info("Choose a player"); return

    prow = players[players["name"]==sel]
    p = prow.iloc[0].to_dict() if not prow.empty else {"id":None,"name":sel,"photo_url":None,"notes":None}
    mine = lfact[lfact["name"]==sel].copy().sort_values(["season","gw"])
    agg_all = player_agg(lfact); me = agg_all[agg_all["name"]==sel]
    me_row = me.iloc[0] if not me.empty else None

    tab1, tab2, tab3 = st.tabs(["Overview", "Match Log", "Duos & Nemesis"])

    with tab1:
        # Hero
        c1, c2 = st.columns([1,2])
        with c1:
            st.image(p.get("photo_url") or f"https://placehold.co/240x240?text={''.join([w[0] for w in sel.split()[:2]]).upper()}", width=200)
        with c2:
            st.markdown(f"## {sel}")
            if p.get("notes"): st.caption(p["notes"])

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
                    f'<div><span class="k">G+A/PG</span> <b>{me_row["ga_pg"]}</b></div>'
                    f'<div><span class="k">Team Contrib%</span> <b>{me_row["team_contrib_pct"]}%</b></div>'
                    f'</div></div>', unsafe_allow_html=True
                )

            # Streaks
            if not mine.empty:
                mine_ord = mine.sort_values(["season","gw"])
                res_streak = compute_streak(mine_ord["result"].tolist(), lambda r: r=="W")
                ga_streak  = compute_streak(mine_ord["ga"].tolist(), lambda x: int(x)>0)
                g_streak   = compute_streak(mine_ord["goals"].tolist(), lambda x: int(x)>0)
                a_streak   = compute_streak(mine_ord["assists"].tolist(), lambda x: int(x)>0)
                st.markdown(
                    f'<div class="card"><div class="kv">'
                    f'<div><span class="k">Win streak</span> <b>{res_streak}</b></div>'
                    f'<div><span class="k">G+A streak</span> <b>{ga_streak}</b></div>'
                    f'<div><span class="k">Goals streak</span> <b>{g_streak}</b></div>'
                    f'<div><span class="k">Assists streak</span> <b>{a_streak}</b></div>'
                    f'</div></div>', unsafe_allow_html=True
                )

    with tab2:
        st.markdown("#### Match Log")
        last_n = int(st.number_input("Show last N games (0 = all)", min_value=0, value=5, step=1, key="pl_lastN"))
        mine2 = mine.sort_values(["season","gw"], ascending=[False,False])
        if last_n>0: mine2 = mine2.head(last_n)
        cols = ["season","gw","team_a","score_a","score_b","team_b","team","goals","assists","ga","result"]
        mine2 = mine2[cols].rename(columns={
            "season":"Season","gw":"GW","team_a":"Team A","score_a":"A",
            "score_b":"B","team_b":"Team B","team":"Side","goals":"Goals","assists":"Assists","ga":"G+A","result":"Result"
        })
        st.dataframe(mine2, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("#### Duos & Nemesis (for player)")
        mg = int(st.number_input("Min games together/against", min_value=1, value=3, step=1, key="pl_min_gp"))
        last_x = int(st.number_input("Last N GWs (0 = all)", min_value=0, value=0, step=1, key="pl_lastX"))
        filt = df_filter_by(lfact[lfact["name"]==sel], None, last_x)
        # Duos: teammates same match+team
        a = filt.merge(lfact, on=["match_id","team"], suffixes=("_me","_tm"))
        a = a[a["name_tm"]!=sel]
        if not a.empty:
            duo_gp = a.groupby("name_tm").size().rename("GP")
            duo_w  = a[a["result_me"]=="W"].groupby("name_tm").size().reindex(duo_gp.index, fill_value=0)
            duo_df = pd.DataFrame({"GP":duo_gp,"W":duo_w}).reset_index().rename(columns={"name_tm":"Teammate"})
            duo_df["Win %"] = ((duo_df["W"]/duo_df["GP"])*100).round(1)
            duo_df = duo_df[duo_df["GP"]>=mg].sort_values(["Win %","GP"], ascending=[False,False])
            st.markdown("**Best Duos**")
            st.dataframe(duo_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No duo data yet.")
        # Nemesis: opponents same match, different teams
        b = filt.merge(lfact, on=["match_id"], suffixes=("_me","_op"))
        b = b[b["team_me"]!=b["team_op"]]
        b = b[b["name_op"]!=sel]
        if not b.empty:
            nem_gp = b.groupby("name_op").size().rename("GP")
            nem_w  = b[b["result_me"]=="W"].groupby("name_op").size().reindex(nem_gp.index, fill_value=0)
            nem_df = pd.DataFrame({"GP":nem_gp,"W":nem_w}).reset_index().rename(columns={"name_op":"Opponent"})
            nem_df["Win % vs"] = ((nem_df["W"]/nem_df["GP"])*100).round(1)
            nem_df = nem_df[nem_df["GP"]>=mg].sort_values(["Win % vs","GP"], ascending=[True,False])
            st.markdown("**Nemesis**")
            st.dataframe(nem_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No nemesis data yet.")

def page_stats():
    players = fetch_players(); matches = fetch_matches(); lineups = fetch_lineups(); awards = fetch_awards()
    lfact, _ = build_fact(players, matches, lineups)
    st.subheader("Stats")

    c1,c2,c3,c4 = st.columns(4)
    seasons = sorted(matches["season"].dropna().unique().tolist()) if not matches.empty else []
    season = c1.selectbox("Season", [None]+seasons, index=0)
    min_games = int(c2.number_input("Min games", min_value=0, value=0, step=1))
    last_x = int(c3.number_input("Last N GWs (0 = all)", min_value=0, value=0, step=1))
    top_n = int(c4.number_input("Rows (Top N, 0=all)", min_value=0, value=10, step=1))

    metric = st.selectbox(
        "Metric",
        ["G+A","Goals","Assists","Goals per Game","Assists per Game","G+A per Game","Win %","Team Contribution %","MOTM","Best Duos","Nemesis Pairs"],
        index=0
    )

    # Apply global filters to ALL metrics
    agg = player_agg(lfact, season=season, min_games=min_games, last_gw=last_x)

    # MOTM counts (honours)
    motm = awards[awards["type"]=="MOTM"].copy()
    if season is not None: motm = motm[motm["season"]==season]
    motm_cnt = motm.groupby("player_name").size().rename("MOTM")
    agg = agg.merge(motm_cnt, left_on="name", right_index=True, how="left").fillna({"MOTM":0})
    agg["MOTM"] = agg["MOTM"].astype(int)

    def df_players_table(cols_order: List[Tuple[str,str]], sort_cols: List[str], asc: List[bool]):
        df = agg.copy()
        # sorting by chosen metric(s)
        df = df.sort_values(sort_cols, ascending=asc)
        if top_n>0: df = df.head(top_n)
        # rename columns for display & select
        rename = {
            "name":"Player","gp":"GP","w":"W","d":"D","l":"L","win_pct":"Win %","goals":"Goals",
            "assists":"Assists","ga":"G+A","g_pg":"G/PG","a_pg":"A/PG","ga_pg":"G+A/PG",
            "team_contrib_pct":"Team Contrib %","MOTM":"MOTM"
        }
        disp_cols = [rename.get(c,c) for c,_ in cols_order]
        base = df[[c for c,_ in cols_order]].rename(columns=rename)
        st.dataframe(base[disp_cols], use_container_width=True, hide_index=True)

    if metric == "Goals":
        cols = [("Player",""),("GP",""),("Goals",""),("G/PG",""),("Assists",""),("G+A",""),("Win %",""),("Team Contrib %",""),("MOTM","")]
        df_players_table(cols, ["goals","ga","assists"], [False,False,False])

    elif metric == "Assists":
        cols = [("Player",""),("GP",""),("Assists",""),("A/PG",""),("Goals",""),("G+A",""),("Win %",""),("Team Contrib %",""),("MOTM","")]
        df_players_table(cols, ["assists","ga","goals"], [False,False,False])

    elif metric == "G+A":
        cols = [("Player",""),("GP",""),("G+A",""),("G+A/PG",""),("Goals",""),("Assists",""),("Win %",""),("Team Contrib %",""),("MOTM","")]
        df_players_table(cols, ["ga","goals","assists"], [False,False,False])

    elif metric == "Goals per Game":
        cols = [("Player",""),("GP",""),("G/PG",""),("Goals",""),("Assists",""),("G+A",""),("Win %",""),("Team Contrib %",""),("MOTM","")]
        df_players_table(cols, ["g_pg","goals"], [False,False])

    elif metric == "Assists per Game":
        cols = [("Player",""),("GP",""),("A/PG",""),("Assists",""),("Goals",""),("G+A",""),("Win %",""),("Team Contrib %",""),("MOTM","")]
        df_players_table(cols, ["a_pg","assists"], [False,False])

    elif metric == "G+A per Game":
        cols = [("Player",""),("GP",""),("G+A/PG",""),("G+A",""),("Goals",""),("Assists",""),("Win %",""),("Team Contrib %",""),("MOTM","")]
        df_players_table(cols, ["ga_pg","ga","goals"], [False,False,False])

    elif metric == "Win %":
        cols = [("Player",""),("GP",""),("Win %",""),("W",""),("D",""),("L",""),("Goals",""),("Assists",""),("G+A",""),("Team Contrib %",""),("MOTM","")]
        df_players_table(cols, ["win_pct","ga","goals"], [False,False,False])

    elif metric == "Team Contribution %":
        cols = [("Player",""),("GP",""),("Team Contrib %",""),("G+A",""),("Goals",""),("Assists",""),("Win %",""),("MOTM","")]
        df_players_table(cols, ["team_contrib_pct","ga","goals"], [False,False,False])

    elif metric == "MOTM":
        cols = [("Player",""),("MOTM",""),("GP",""),("G+A",""),("Goals",""),("Assists",""),("Win %","")]
        # sort by MOTM then G+A
        df = agg.sort_values(["MOTM","ga","goals"], ascending=[False,False,False])
        if top_n>0: df = df.head(top_n)
        df = df.rename(columns={"name":"Player","gp":"GP","win_pct":"Win %","goals":"Goals","assists":"Assists","ga":"G+A"})
        st.dataframe(df[[c for c,_ in cols]], use_container_width=True, hide_index=True)

    elif metric == "Best Duos":
        mg = max(1, min_games)
        df = duo_global(lfact, min_gp=mg, season=season, last_gw=last_x)
        if top_n>0: df = df.head(top_n)
        st.dataframe(df.rename(columns={"pair":"Duo","gp":"GP","w":"W","win_pct":"Win %"}), use_container_width=True, hide_index=True)

    elif metric == "Nemesis Pairs":
        mg = max(1, min_games)
        df = nemesis_global(lfact, min_gp=mg, season=season, last_gw=last_x)
        if top_n>0: df = df.head(top_n)
        st.dataframe(df.rename(columns={"pair":"Pair","gp":"GP","win_pct_vs":"Win % vs"}), use_container_width=True, hide_index=True)

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
                        "notes": notes or None
                    }).execute()
                    clear_caches(); st.success("Saved."); st.rerun()

    st.markdown("#### Player of the Month")
    potm = aw[aw["type"]=="POTM"]
    if potm.empty: st.caption("No POTM yet.")
    else:
        for _, r in potm.sort_values(["season","month"]).iterrows():
            st.write(f"🏆 S{r['season']} · M{int(r['month'])}: {r['player_name']}")

    st.markdown("#### Man of the Match (History)")
    motm = aw[aw["type"]=="MOTM"]
    if motm.empty: st.caption("No MOTM yet.")
    else:
        for _, r in motm.sort_values(["season","gw"]).iterrows():
            st.write(f"🎖️ S{r['season']} GW{r['gw']}: {r['player_name']}")

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
