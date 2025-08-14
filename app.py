# app.py — Powerleague Stats (FotMob pitch, robust lineup updates, add/edit matches & dates)

import io
from typing import Optional, List, Dict

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
  --pitch:#134d33; --line:#9bd5ad; --text:#e6edf3;
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
.badge{background:#22c55e;color:#062616;font-weight:700;border-radius:999px;padding:6px 12px;font-size:12px;border:1px solid rgba(0,0,0,.25)}
.hr{height:1px; background:rgba(255,255,255,.10); margin:10px 0 16px}

/* Pitch */
.pitch{position:relative;width:100%;padding-top:150%;border-radius:20px;overflow:hidden;
       background:var(--pitch);box-shadow:0 6px 18px rgba(0,0,0,.35);border:1px solid rgba(255,255,255,.18)}
.pitch::before{
  content:"";position:absolute;inset:0;
  background:repeating-linear-gradient(180deg, rgba(255,255,255,.04) 0 6%, rgba(0,0,0,0) 6% 12%);
}
/* Lines */
.lines{position:absolute;inset:4% 4%;border:2px solid var(--line);border-radius:10px;opacity:.8}
.penalty{position:absolute;left:6%;right:6%;top:0;height:22%;border:2px solid var(--line);border-top:0;border-radius:0 0 10px 10px}
.six{position:absolute;left:25%;right:25%;top:0;height:10%;border:2px solid var(--line);border-top:0;border-radius:0 0 8px 8px}
.penalty.bot{top:auto;bottom:0;border-top:2px solid var(--line);border-bottom:0;border-radius:10px 10px 0 0}
.six.bot{top:auto;bottom:0;border-top:2px solid var(--line);border-bottom:0;border-radius:8px 8px 0 0}
.half{position:absolute;left:0;right:0;top:50%;border-top:2px solid var(--line)}
.center{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);width:34%;height:34%;border:2px solid var(--line);border-radius:50%}

/* Players */
.spot{position:absolute;transform:translate(-50%,-50%);display:flex;flex-direction:column;align-items:center}
.avatar{position:relative;width:72px;height:72px;border-radius:50%;overflow:hidden;border:2px solid rgba(255,255,255,.55);box-shadow:0 2px 10px rgba(0,0,0,.45);background:#fff}
.avatar img{width:100%;height:100%;object-fit:cover}
.init{width:100%;height:100%;display:flex;align-items:center;justify-content:center;background:#fff;color:#1f2937;font-weight:800;font-size:24px}
.motm{position:absolute;top:-8px;right:-8px;background:gold;color:#000;border-radius:50%;padding:2px 4px;font-size:12px;border:1px solid rgba(0,0,0,.25)}
.name{margin-top:4px;font-size:13px;font-weight:700;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,.6);white-space:nowrap}
.chips{display:flex;gap:4px;margin-top:2px}
.statchip{background:#fff;color:#111;padding:2px 6px;border-radius:10px;font-size:12px}

/* Editor buttons */
.slotBtn{width:100%;height:44px;border-radius:10px;border:1px dashed rgba(255,255,255,.25);background:rgba(255,255,255,.04)}
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
@st.cache_data(ttl=30)
def fetch_players() -> pd.DataFrame:
    return pd.DataFrame(sb_public.table("players").select("*").execute().data or [])

@st.cache_data(ttl=30)
def fetch_matches() -> pd.DataFrame:
    res = sb_public.table("matches").select("*").order("season").order("gw").execute().data or []
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
def formation_to_lines(form: str) -> List[int]:
    """Outfield rows (top->bottom). GK is drawn as last row."""
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

# ---------- Duos / Nemesis ----------
def duo_global(lfact: pd.DataFrame, min_gp: int = 3, df_filter=None):
    df = lfact.copy()
    if df_filter is not None: df = df_filter(df)
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

def nemesis_global(lfact: pd.DataFrame, min_gp: int = 3, df_filter=None):
    df = lfact.copy()
    if df_filter is not None: df = df_filter(df)
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
# Pitch rendering
# =========================
def _ensure_positions(team_df: pd.DataFrame, formation: str) -> pd.DataFrame:
    """Return df with guaranteed line/slot; GK renders on last row."""
    rows = team_df.copy()
    parts = formation_to_lines(formation)
    max_slots = max(parts+[1])

    have = rows["line"].notna().sum()>0 and rows["slot"].notna().sum()>0
    if have: return rows

    rows["line"] = np.nan; rows["slot"] = np.nan
    gk = rows[rows.get("is_gk", False)==True]
    others = rows.drop(index=(gk.index[0] if not gk.empty else []), errors="ignore")

    # distribute outfield across parts (top->bottom)
    cur=0; filled=[0]*len(parts)
    for idx,_ in others.iterrows():
        if cur>=len(parts): cur=0
        slots=parts[cur]; offset=(max_slots-slots)//2
        pos=filled[cur] % slots
        rows.loc[idx,["line","slot"]] = [cur, offset+pos]
        filled[cur]+=1; cur+=1

    # GK bottom center
    if not gk.empty:
        center = max_slots//2
        rows.loc[gk.index[0],["line","slot"]] = [len(parts), center]
    return rows

def _avatar_html(name: str, photo_url: Optional[str]) -> str:
    if photo_url and str(photo_url).strip():
        return f"<div class='avatar'><img src='{photo_url}'/></div>"
    # initials only (clean)
    init = "".join([p[0] for p in str(name).split() if p])[:2].upper() or "?"
    return f"<div class='avatar'><div class='init'>{init}</div></div>"

def render_pitch(team_df: pd.DataFrame, formation: str, motm_name: Optional[str], show_stats=True, show_photos=True):
    rows = _ensure_positions(team_df, formation)
    parts = formation_to_lines(formation)
    max_slots = max(parts+[1])
    total_rows = len(parts)+1  # GK

    def y_for(i):
        # From 8% (top) to 92% (bottom)
        return 8 + (84 * (i/(total_rows-1))) if total_rows>1 else 50

    html = ["<div class='pitch'>",
            "<div class='lines'></div>",
            "<div class='penalty'></div><div class='six'></div>",
            "<div class='penalty bot'></div><div class='six bot'></div>",
            "<div class='half'></div><div class='center'></div>"]

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

# =========================
# Lineup editor (robust updates)
# =========================
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

def pitch_editor(team_name: str, formation: str, team_rows: pd.DataFrame, key_prefix: str):
    st.caption(f"{team_name} — tap a slot to place the selected player (GK row included)")
    rows = _ensure_positions(team_rows, formation)

    rows["label"] = rows.apply(lambda r: f"{r['name']} ({'GK' if r.get('is_gk') else 'Outfield'})", axis=1)
    opts = rows["label"].tolist()
    id_by_label = {opts[i]: str(rows.iloc[i]["id"]) for i in range(len(rows))}
    pick = st.selectbox("Player", opts, key=f"{key_prefix}_pick")
    rid = id_by_label.get(pick)

    c1,c2,c3 = st.columns(3)
    with c1:
        if st.button("Set as GK", key=f"{key_prefix}_gk"):
            parts = formation_to_lines(formation); center = (max(parts+[1])//2)
            # toggle flags and put chosen on GK row center
            for _, r in rows.iterrows():
                update_lineup_row(r["id"], {"is_gk": bool(str(r["id"])==rid)})
            update_lineup_row(rid, {"line": len(parts), "slot": center})
            clear_caches(); st.success("GK updated."); st.rerun()
    with c2:
        if st.button("Clear position", key=f"{key_prefix}_clear"):
            update_lineup_row(rid, {"line": None, "slot": None}); clear_caches(); st.rerun()
    with c3:
        if st.button("Remove from lineup", key=f"{key_prefix}_rm"):
            remove_lineup_row(rid); st.rerun()

    # grid buttons
    parts = formation_to_lines(formation)
    max_slots = max(parts+[1]); total_rows = len(parts)+1
    offsets = [(max_slots - s)//2 for s in parts] + [0]  # GK center slot visible
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
                    # free any occupant in that cell
                    for _, r in occ.iterrows():
                        update_lineup_row(r["id"], {"line": None, "slot": None})
                    update_lineup_row(rid, {"line": int(line_idx), "slot": int(grid_slot)})
                    clear_caches(); st.success("Position saved."); st.rerun()

# =========================
# UI helpers
# =========================
def chip(name, goals, assists, photo):
    img = f'<img src="{photo}" alt=""/>' if (photo and str(photo).strip()) else ""
    g = int(goals or 0); a = int(assists or 0)
    stats = " · ".join(s for s in [f"⚽ {g}" if g else "", f"🅰️ {a}" if a else ""] if s)
    if stats:
        return f'<span style="display:inline-flex;align-items:center;gap:10px;background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.30);padding:9px 12px;border-radius:18px;font-size:15px;line-height:1.1;color:#fff;max-width:100%;">{img}<span>{name}</span><span style="font-size:12px;opacity:.9">{stats}</span></span>'
    return f'<span style="display:inline-flex;align-items:center;gap:10px;background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.30);padding:9px 12px;border-radius:18px;font-size:15px;line-height:1.1;color:#fff;max-width:100%;">{img}<span>{name}</span></span>'

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
# Add / Edit Matches
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
        is_draw = False  # computed
        team_a, team_b = "Non-bibs", "Bibs"
        sc1, sc2 = st.columns(2)
        score_a = int(sc1.number_input(f"{team_a} goals", min_value=0, value=0))
        score_b = int(sc2.number_input(f"{team_b} goals", min_value=0, value=0))
        is_draw = (score_a==score_b)
        motm_name = st.text_input("Man of the Match (optional)")
        fa, fb = st.columns(2)
        form_a = fa.selectbox("Formation (Non-bibs)", ["1-2-1","1-3","2-2","3-1","2-1-2-1","3-2-1","2-3-1"], index=0)
        form_b = fb.selectbox("Formation (Bibs)", ["1-2-1","1-3","2-2","3-1","2-1-2-1","3-2-1","2-3-1"], index=0)
        notes = st.text_area("Notes", "")
        submit = st.form_submit_button("Save match")
        if submit:
            s = service()
            if not s: st.error("Admin required.")
            else:
                payload = {
                    "season": season, "gw": gw, "side_count": side_count,
                    "team_a": team_a, "team_b": team_b,
                    "score_a": score_a, "score_b": score_b, "is_draw": is_draw,
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
        st.dataframe(matches.sort_values(["season","gw"])[["season","gw","date","team_a","score_a","score_b","team_b","motm_name"]], use_container_width=True)
        return
    for _, m in matches.sort_values(["season","gw"]).iterrows():
        with st.expander(f"S{int(m['season'])} GW{int(m['gw'])} — {m['team_a']} {m['score_a']}–{m['score_b']} {m['team_b']}"):
            d1,d2,d3,d4 = st.columns([2,2,1,1])
            date = d1.date_input("Date", value=pd.to_datetime(m["date"]).date() if pd.notna(m["date"]) else pd.Timestamp.today().date(), key=f"dt_{m['id']}")
            sa = int(d2.number_input("Score A", min_value=0, value=int(m.get("score_a") or 0), key=f"sa_{m['id']}"))
            sb = int(d3.number_input("Score B", min_value=0, value=int(m.get("score_b") or 0), key=f"sb_{m['id']}"))
            motm = d4.text_input("MOTM", value=m.get("motm_name") or "", key=f"mm_{m['id']}")
            fa, fb = st.columns(2)
            forma = fa.selectbox("Formation A", ["1-2-1","1-3","2-2","3-1","2-1-2-1","3-2-1","2-3-1"], index=0 if not m.get("formation_a") else 0, key=f"fa_{m['id']}")
            formb = fb.selectbox("Formation B", ["1-2-1","1-3","2-2","3-1","2-1-2-1","3-2-1","2-3-1"], index=0 if not m.get("formation_b") else 0, key=f"fb_{m['id']}")
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

    # Summary
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
        with st.expander("✏️ Edit lineups & positions", expanded=False):
            pa, pb = st.columns(2)
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
                pitch_editor(m["team_a"], m.get("formation_a") or "1-2-1", a_rows, key_prefix=f"A_{mid}")
            with eb:
                pitch_editor(m["team_b"], m.get("formation_b") or "1-2-1", b_rows, key_prefix=f"B_{mid}")

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
        st.image(p.get("photo_url") or f"https://placehold.co/240x240?text={''.join([w[0] for w in sel.split()[:2]]).upper()}", width=180)
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
                f'<div class="card"><div style="display:flex;gap:12px;flex-wrap:wrap">'
                f'<div>GP<br><b>{int(me_row["gp"])}</b></div>'
                f'<div>W-D-L<br><b>{int(me_row["w"])}-{int(me_row["d"])}-{int(me_row["l"])}</b></div>'
                f'<div>Win%<br><b>{me_row["win_pct"]}%</b></div>'
                f'<div>Goals<br><b>{int(me_row["goals"])}</b></div>'
                f'<div>Assists<br><b>{int(me_row["assists"])}</b></div>'
                f'<div>G+A<br><b>{int(me_row["ga"])}</b></div>'
                f'<div>G/PG<br><b>{me_row["g_pg"]}</b></div>'
                f'<div>A/PG<br><b>{me_row["a_pg"]}</b></div>'
                f'<div>Team Contrib%<br><b>{me_row["team_contrib_pct"]}%</b></div>'
                f'</div></div>', unsafe_allow_html=True
            )

        st.markdown("#### Recent Games")
        last_n = int(st.number_input("Show last N games (0 = all)", min_value=0, value=10, step=1))
        mine2 = mine.sort_values(["season","gw"], ascending=[False,False])
        if last_n>0: mine2 = mine2.head(last_n)
        if mine2.empty: st.caption("No games.")
        else:
            for _, r in mine2.iterrows():
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
    last_x = int(c3.number_input("Last N GWs (0 = all)", min_value=0, value=0, step=1))
    top_n = int(c4.number_input("Rows (Top N)", min_value=5, value=10, step=1))

    metric = st.selectbox(
        "Metric",
        ["G+A","Goals","Assists","Goals per Game","Assists per Game","G+A per Game","Win %","Team Contribution %","MOTM","Best Duos","Nemesis Pairs"],
        index=0
    )
    show_photos = st.toggle("Show photos", True)

    # A helper filter for duos/nemesis based on season & last_x
    def df_filter(df):
        d = df.copy()
        if season is not None: d = d[d["season"]==season]
        if last_x and last_x>0:
            max_gw = pd.to_numeric(d["gw"], errors="coerce").max()
            if pd.notna(max_gw): d = d[pd.to_numeric(d["gw"], errors="coerce") >= (int(max_gw)-int(last_x)+1)]
        return d

    agg = player_agg(lfact, season=season, min_games=min_games, last_gw=last_x)

    def render_row(name, value, photo):
        cA,cB = st.columns([6,1])
        with cA:
            st.markdown(chip(name, 0, 0, photo if show_photos else None), unsafe_allow_html=True)
        with cB:
            if isinstance(value, float): value = round(value, 2)
            st.metric("Value", value)

    if metric == "MOTM":
        aw = fetch_awards(); motm = aw[aw["type"]=="MOTM"].copy()
        if season is not None: motm = motm[motm["season"]==season]
        df = motm.groupby("player_name").size().reset_index(name="MOTM").sort_values("MOTM", ascending=False)
        df = df.merge(players[["name","photo_url"]], left_on="player_name", right_on="name", how="left")
        for _, r in df.head(top_n).iterrows():
            render_row(r["player_name"], r["MOTM"], r["photo_url"])

    elif metric == "Best Duos":
        mg = max(3, min_games)
        df = duo_global(lfact, min_gp=mg, df_filter=df_filter)
        if df.empty: st.caption("No duo data.")
        else:
            for _, r in df.head(top_n).iterrows():
                st.write(f"👥 {r['pair']} — {r['win_pct']}% · GP {int(r['gp'])}")

    elif metric == "Nemesis Pairs":
        mg = max(3, min_games)
        df = nemesis_global(lfact, min_gp=mg, df_filter=df_filter)
        if df.empty: st.caption("No nemesis data.")
        else:
            for _, r in df.head(top_n).iterrows():
                st.write(f"⚔️ {r['pair']} — {r['win_pct_vs']}% · GP {int(r['gp'])}")

    else:
        df = agg.copy()
        if metric == "Goals":
            df = df.sort_values(["goals","ga","assists"], ascending=False)
            for _, r in df.head(top_n).iterrows(): render_row(r["name"], int(r["goals"]), r["photo"])
        elif metric == "Assists":
            df = df.sort_values(["assists","ga","goals"], ascending=False)
            for _, r in df.head(top_n).iterrows(): render_row(r["name"], int(r["assists"]), r["photo"])
        elif metric == "G+A":
            df = df.sort_values(["ga","goals","assists"], ascending=False)
            for _, r in df.head(top_n).iterrows(): render_row(r["name"], int(r["ga"]), r["photo"])
        elif metric == "G+A per Game":
            df = df.sort_values(["ga_pg","ga","goals"], ascending=False)
            for _, r in df.head(top_n).iterrows(): render_row(r["name"], r["ga_pg"], r["photo"])
        elif metric == "Goals per Game":
            df = df.sort_values(["g_pg","goals"], ascending=False)
            for _, r in df.head(top_n).iterrows(): render_row(r["name"], r["g_pg"], r["photo"])
        elif metric == "Assists per Game":
            df = df.sort_values(["a_pg","assists"], ascending=False)
            for _, r in df.head(top_n).iterrows(): render_row(r["name"], r["a_pg"], r["photo"])
        elif metric == "Win %":
            df = df.sort_values(["win_pct","ga","goals"], ascending=False)
            for _, r in df.head(top_n).iterrows(): render_row(r["name"], r["win_pct"], r["photo"])
        elif metric == "Team Contribution %":
            df = df.sort_values(["team_contrib_pct","ga","goals"], ascending=False)
            for _, r in df.head(top_n).iterrows(): render_row(r["name"], r["team_contrib_pct"], r["photo"])

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
# Avatar upload
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
