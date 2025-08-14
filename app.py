# app.py — Powerleague Stats (Streamlit + Supabase)
# Mobile-first, combined pitch, 5s/7s guard, admin tools, stats, players, awards, import/export.

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime
from typing import Optional, List, Tuple, Dict
from supabase import create_client
import uuid
import io
import base64

# Optional HEIC -> PNG
try:
    import pillow_heif
    HEIF_OK = True
except Exception:
    HEIF_OK = False

from PIL import Image

# -----------------------------------
# Streamlit config
# -----------------------------------
st.set_page_config(page_title="Powerleague Stats", layout="wide", initial_sidebar_state="collapsed")

# -----------------------------------
# Secrets / Supabase
# -----------------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
SUPABASE_SERVICE_KEY = st.secrets.get("SUPABASE_SERVICE_KEY", SUPABASE_ANON_KEY)
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "")
AVATAR_BUCKET = st.secrets.get("AVATAR_BUCKET", "avatars")

sb = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
sb_service = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# -----------------------------------
# Small global styles (black & gold vibe)
# -----------------------------------
GLOBAL_CSS = """
<style>
:root{--gold:#D4AF37;--ink:#0b0f14;--panel:#0f141a;--text:#e9eef3}
html,body,.stApp{background:#0b0f14;}
.block-container{padding-top:1rem!important;}
h1,h2,h3,h4,h5{color:var(--text)}
.small{opacity:.8;font-size:.9rem}
.badge{display:inline-flex;align-items:center;gap:.4rem;padding:.25rem .55rem;border:1px solid rgba(255,255,255,.18);
 border-radius:999px;background:rgba(255,255,255,.06)}
.badge .dot{width:10px;height:10px;border-radius:50%}
.form-row{display:flex;gap:.75rem;align-items:end;flex-wrap:wrap}
.form-row>div{flex:1 1 180px}
hr{border-color:rgba(255,255,255,.12)}
/* Table polishing */
thead tr th{background:rgba(255,255,255,.06)!important}
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# -----------------------------------
# Cache utilities
# -----------------------------------
def clear_caches():
    st.cache_data.clear()

@st.cache_data(ttl=90)
def fetch_players() -> pd.DataFrame:
    try:
        data = sb.table("players").select("*").order("name").execute().data
        return pd.DataFrame(data) if data else pd.DataFrame(columns=["id","name","photo_url","notes"])
    except Exception:
        return pd.DataFrame(columns=["id","name","photo_url","notes"])

@st.cache_data(ttl=60)
def fetch_matches() -> pd.DataFrame:
    try:
        data = sb.table("matches").select("*").order("season").order("gw").execute().data
        return pd.DataFrame(data) if data else pd.DataFrame(columns=[
            "id","season","gw","side_count","team_a","team_b","score_a","score_b","date",
            "motm_name","is_draw","formation_a","formation_b","notes"
        ])
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_lineups() -> pd.DataFrame:
    try:
        data = sb.table("lineups").select("*").execute().data
        return pd.DataFrame(data) if data else pd.DataFrame(columns=[
            "id","season","gw","match_id","team","player_id","player_name","name","is_gk","goals","assists","line","slot","position"
        ])
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=120)
def fetch_awards() -> pd.DataFrame:
    try:
        data = sb.table("awards").select("*").execute().data
        return pd.DataFrame(data) if data else pd.DataFrame(columns=[
            "id","season","month","type","gw","player_id","player_name","notes"
        ])
    except Exception:
        return pd.DataFrame()

def service():
    return sb_service if st.session_state.get("is_admin") else None

# -----------------------------------
# Helpers
# -----------------------------------
def initials(name: str) -> str:
    parts = [p for p in (name or "").split() if p]
    return "".join([p[0] for p in parts[:2]]).upper() or "?"

def formation_to_lines(formation: Optional[str]) -> List[int]:
    try:
        return [int(x) for x in str(formation or "").strip().split("-") if str(x).strip().isdigit()]
    except Exception:
        return []

def validate_formation(formation: Optional[str], side_count: int) -> str:
    """
    5s => 4 outfielders (e.g., 1-2-1, 2-2, 3-1)
    7s => 6 outfielders (e.g., 2-1-2-1, 3-2-1, 2-3-1)
    Falls back to defaults if invalid.
    """
    try:
        parts = [int(x) for x in str(formation or "").split("-") if str(x).strip().isdigit()]
    except Exception:
        parts = []
    outfield_needed = 4 if int(side_count or 5) == 5 else 6
    if sum(parts) != outfield_needed or not parts or any(p <= 0 for p in parts):
        return "1-2-1" if outfield_needed == 4 else "2-1-2-1"
    return "-".join(str(p) for p in parts)

def _ensure_positions(df: pd.DataFrame, formation: str) -> pd.DataFrame:
    """
    Guarantee sane values for is_gk/line/slot for a given formation.
    Assign centered defaults if missing; clamp to formation bounds.
    """
    rows = df.copy()
    parts = formation_to_lines(formation) or [1,2,1]
    n_lines = max(1, len(parts))
    max_slots = max(parts + [1])

    for c in ["is_gk","line","slot","goals","assists","name","player_name"]:
        if c not in rows.columns:
            rows[c] = None

    rows["is_gk"] = rows["is_gk"].fillna(False).astype(bool)
    rows["goals"] = pd.to_numeric(rows["goals"], errors="coerce").fillna(0).astype(int)
    rows["assists"] = pd.to_numeric(rows["assists"], errors="coerce").fillna(0).astype(int)

    if "name" not in rows.columns: rows["name"] = rows["player_name"]
    rows["name"] = rows["name"].fillna(rows["player_name"]).fillna("").astype(str)

    rows["line"] = pd.to_numeric(rows["line"], errors="coerce").astype("Int64")
    rows["slot"] = pd.to_numeric(rows["slot"], errors="coerce").astype("Int64")

    center_line = (n_lines - 1) // 2
    for i in rows.index:
        if bool(rows.at[i,"is_gk"]):
            rows.at[i,"line"] = pd.NA; rows.at[i,"slot"] = pd.NA
            continue
        ln = rows.at[i,"line"]
        ln = center_line if pd.isna(ln) else int(ln)
        ln = max(0, min(ln, n_lines-1))
        rows.at[i,"line"] = ln
        slots = int(parts[ln])
        offset = (max_slots - slots)//2
        sl = rows.at[i,"slot"]
        sl = (offset + (slots-1)//2) if pd.isna(sl) else int(sl)
        sl = max(offset, min(sl, offset + slots - 1))
        rows.at[i,"slot"] = sl

    return rows

def normalize_lineup_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "name" not in out.columns:
        out["name"] = out["player_name"] if "player_name" in out.columns else ""
    else:
        if "player_name" in out.columns:
            out["name"] = out["name"].fillna(out["player_name"])
        out["name"] = out["name"].fillna("")
    out["name"] = out["name"].astype(str)
    return out

# -----------------------------------
# Pitch UI (combined, compact, premium)
# -----------------------------------
def stat_pill(goals: int, assists: int) -> str:
    parts = []
    if (goals or 0) > 0: parts.append(f"<span class='tag tag-g'>G</span><b>{int(goals)}</b>")
    if (assists or 0) > 0: parts.append(f"<span class='tag tag-a'>A</span><b>{int(assists)}</b>")
    if not parts: return ""
    return f"<span class='pill'>{'&nbsp;&nbsp;&nbsp;'.join(parts)}</span>"

def slot_html(x_pct: float, y_pct: float, name: str, *, motm: bool=False, pill: str="", is_gk: bool=False) -> str:
    cls = "bubble"; 
    if motm: cls += " motm"
    if is_gk: cls += " gk"
    init = initials(name)
    gk_chip = "<span class='chip-gk'>GK</span>" if is_gk else ""
    return (
        f"<div class='slot' style='left:{x_pct}%;top:{y_pct}%;'>"
        f"  <div class='{cls}'><span class='init'>{init}</span>{gk_chip}</div>"
        f"  <div class='name'>{name}</div>"
        f"  {pill}"
        f"</div>"
    )

def render_match_pitch_combined(a_rows: pd.DataFrame, b_rows: pd.DataFrame,
                                formation_a: str, formation_b: str,
                                motm_name: Optional[str],
                                team_a: str, team_b: str,
                                show_stats: bool=True):
    """
    One compact premium pitch with both teams.
    SIDE-ORIENTED: lines run goal→center along X per half (mirrored).
    Bigger bubbles + tighter margins = less empty space, no overlaps on iPhone.
    """
    css = """
    <style>
    .pitchX{position:relative;width:100%;padding-top:58%;
      border-radius:18px;overflow:hidden;
      background:
        repeating-linear-gradient(0deg, rgba(255,255,255,.05) 0 11px, rgba(255,255,255,0) 11px 24px),
        radial-gradient(1000px 600px at 50% -20%, #2f7a43 0%, #2a6f3c 45%, #235f34 100%);
      border:1px solid rgba(255,255,255,.16)}
    .inner{position:absolute;inset:8px;border-radius:14px}
    .lines{position:absolute;left:3.5%;top:5%;right:3.5%;bottom:5%}
    .outline{position:absolute;inset:0;border:2px solid #ffffff}
    .halfway-v{position:absolute;left:50%;top:0;bottom:0;border-left:2px solid #ffffff}
    .center{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);width:13%;height:13%;
      border:2px solid #ffffff;border-radius:999px}
    .center-dot{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
      width:6px;height:6px;background:#ffffff;border-radius:999px}
    .box-left{position:absolute;left:0;top:20%;bottom:20%;width:16.5%;border:2px solid #ffffff;border-left:none}
    .six-left{position:absolute;left:0;top:39%;bottom:39%;width:7.6%;border:2px solid #ffffff;border-left:none}
    .pen-dot-left{position:absolute;left:11%;top:50%;transform:translate(-50%,-50%);width:6px;height:6px;background:#ffffff;border-radius:999px}
    .goal-left{position:absolute;left:-1.0%;top:47%;bottom:47%;width:1%;border:2px solid #ffffff;border-right:none}
    .box-right{position:absolute;right:0;top:20%;bottom:20%;width:16.5%;border:2px solid #ffffff;border-right:none}
    .six-right{position:absolute;right:0;top:39%;bottom:39%;width:7.6%;border:2px solid #ffffff;border-right:none}
    .pen-dot-right{position:absolute;right:11%;top:50%;transform:translate(50%,-50%);width:6px;height:6px;background:#ffffff;border-radius:999px}
    .goal-right{position:absolute;right:-1.0%;top:47%;bottom:47%;width:1%;border:2px solid #ffffff;border-left:none}

    .slot{position:absolute;transform:translate(-50%,-50%);display:flex;flex-direction:column;align-items:center;gap:.32rem;text-align:center}
    .bubble{
      width:clamp(60px, 7.4vw, 80px); height:clamp(60px, 7.4vw, 80px);
      border-radius:999px; display:flex; align-items:center; justify-content:center; position:relative;
      background:linear-gradient(180deg,#0e1620,#0b131b); border:2px solid #2f4860; box-shadow:0 5px 14px rgba(0,0,0,.34)
    }
    .bubble.motm{ border-color:#D4AF37; box-shadow:0 0 0 2px rgba(212,175,55,.22), 0 10px 22px rgba(212,175,55,.16) }
    .bubble.gk{ background:linear-gradient(180deg,#0c1e2b,#0a1924); border-color:#4db6ff }
    .chip-gk{
      position:absolute; right:-6px; top:-6px; padding:.14rem .34rem; font-size:.66rem; font-weight:900;
      border-radius:8px; background:rgba(77,182,255,.18); color:#bfe6ff; border:1px solid rgba(77,182,255,.45)
    }
    .init{font-weight:900;letter-spacing:.3px;color:#e8f4ff;font-size:clamp(1.0rem,1.15vw,1.12rem)}
    .name{font-size:clamp(.9rem,1.05vw,1.02rem); font-weight:800; color:#F1F6FA; text-shadow:0 1px 0 rgba(0,0,0,.45); max-width:140px}

    .pill{display:inline-flex;align-items:center;gap:.6rem; padding:.26rem .62rem; border-radius:999px;
      background:rgba(0,0,0,.25); border:1px solid rgba(255,255,255,.18); font-size:clamp(.9rem,1.0vw,1.0rem)}
    .tag{
      display:inline-flex;align-items:center;justify-content:center;
      width:20px;height:20px;border-radius:5px;font-size:.74rem;font-weight:900;border:1px solid rgba(255,255,255,.3)
    }
    .tag-g{ color:#D4AF37; background:rgba(212,175,55,.15); border-color:rgba(212,175,55,.55) }
    .tag-a{ color:#86c7ff; background:rgba(134,199,255,.15); border-color:rgba(134,199,255,.55) }

    @media (max-width: 420px){
      .pitchX{ padding-top:60%; }
      .inner{ inset:6px; }
      .lines{ left:3%; right:3%; top:5.5%; bottom:5.5%; }
    }
    </style>
    """

    def lerp(a: float, b: float, t: float) -> float: return a + (b - a) * t

    a_rows = _ensure_positions(a_rows, formation_a)
    b_rows = _ensure_positions(b_rows, formation_b)
    parts_a = formation_to_lines(formation_a) or [1,2,1]
    parts_b = formation_to_lines(formation_b) or [1,2,1]

    y_top_margin, y_bot_margin = 6, 6
    inner_h = 100 - y_top_margin - y_bot_margin

    left_min, left_max  = 5, 48
    right_min, right_max = 52, 95

    def _place_side(rows: pd.DataFrame, parts: List[int], *, left_half: bool):
        out = []
        n_lines = max(1, len(parts))

        gk = rows[rows.get("is_gk") == True]
        if not gk.empty:
            r = gk.iloc[0]; nm = str(r.get("name") or r.get("player_name") or "")
            x = lerp(left_min, left_max, -0.03) if left_half else lerp(right_max, right_min, -0.03)
            y = 50
            pill = stat_pill(int(r.get("goals") or 0), int(r.get("assists") or 0)) if show_stats else ""
            out.append(slot_html(x, y, nm, motm=(motm_name==nm), pill=pill, is_gk=True))

        for line_idx in range(n_lines):
            t = (line_idx + 1) / (n_lines + 1)
            x = lerp(left_min, left_max, t) if left_half else lerp(right_max, right_min, t)
            line_df = rows[(rows.get("is_gk") != True) & (rows.get("line") == line_idx)].copy()
            if line_df.empty: continue
            line_df["slot"] = pd.to_numeric(line_df["slot"], errors="coerce")
            line_df = line_df.sort_values("slot", na_position="last").reset_index(drop=True)
            count = len(line_df)
            for j in range(count):
                rr = line_df.iloc[j]; nm = str(rr.get("name") or rr.get("player_name") or "")
                y_t = (j + 1) / (count + 1); y = y_top_margin + y_t * inner_h
                pill = stat_pill(int(rr.get("goals") or 0), int(rr.get("assists") or 0)) if show_stats else ""
                out.append(slot_html(x, y, nm, motm=(motm_name==nm), pill=pill, is_gk=False))
        return out

    html = [css, "<div class='pitchX'><div class='inner'>",
            "<div class='lines'>"
            "<div class='outline'></div>"
            "<div class='halfway-v'></div>"
            "<div class='center'></div>"
            "<div class='center-dot'></div>"
            "<div class='box-left'></div><div class='six-left'></div><div class='pen-dot-left'></div><div class='goal-left'></div>"
            "<div class='box-right'></div><div class='six-right'></div><div class='pen-dot-right'></div><div class='goal-right'></div>"
            "</div>"]
    html += _place_side(a_rows, parts_a, left_half=True)
    html += _place_side(b_rows, parts_b, left_half=False)
    html.append("</div></div>")
    st.markdown("".join(html), unsafe_allow_html=True)

# -----------------------------------
# Fact table builder for stats
# -----------------------------------
@st.cache_data(ttl=90)
def build_fact(players: pd.DataFrame, matches: pd.DataFrame, lineups: pd.DataFrame) -> pd.DataFrame:
    if lineups.empty or matches.empty: 
        return pd.DataFrame(columns=[
            "match_id","season","gw","date","team","name","is_gk","goals","assists",
            "for","against","result","contrib"
        ])
    l = lineups.copy()
    l["name"] = l["name"].fillna(l.get("player_name")).fillna("").astype(str)
    m = matches.set_index("id")
    # join match meta
    l["season"] = l["match_id"].map(m["season"])
    l["gw"] = l["match_id"].map(m["gw"])
    l["date"] = l["match_id"].map(m["date"])
    l["score_a"] = l["match_id"].map(m["score_a"])
    l["score_b"] = l["match_id"].map(m["score_b"])
    l["team_a"] = l["match_id"].map(m["team_a"])
    l["team_b"] = l["match_id"].map(m["team_b"])

    # compute for/against per row
    def fa(row):
        if row["team"] == "Non-bibs":
            return int(row.get("score_a") or 0), int(row.get("score_b") or 0)
        return int(row.get("score_b") or 0), int(row.get("score_a") or 0)
    fa_cols = l.apply(lambda r: pd.Series(fa(r), index=["for","against"]), axis=1)
    l[["for","against"]] = fa_cols
    l["result"] = np.where(l["for"] > l["against"], "W", np.where(l["for"] == l["against"], "D", "L"))
    l["goals"] = pd.to_numeric(l["goals"], errors="coerce").fillna(0).astype(int)
    l["assists"] = pd.to_numeric(l["assists"], errors="coerce").fillna(0).astype(int)

    # team goals in each match for contribution calc
    team_goals = (l.groupby(["match_id","team"])["goals"].sum().rename("team_goals")).reset_index()
    l = l.merge(team_goals, on=["match_id","team"], how="left")
    l["contrib"] = ((l["goals"] + l["assists"]) / l["team_goals"].replace(0, np.nan) * 100).round(1)
    l["contrib"] = l["contrib"].fillna(0)

    return l[["match_id","season","gw","date","team","name","is_gk","goals","assists","for","against","result","contrib"]]

# -----------------------------------
# Aggregations
# -----------------------------------
def filter_fact(lfact: pd.DataFrame, season: Optional[int], last_gw: Optional[int]) -> pd.DataFrame:
    df = lfact.copy()
    if season and season != -1:
        df = df[df["season"] == int(season)]
    if last_gw and int(last_gw) > 0:
        max_gw = df["gw"].max()
        df = df[df["gw"] > max_gw - int(last_gw)]
    return df

def player_agg(lfact: pd.DataFrame, season: Optional[int], min_games: int, last_gw: Optional[int]) -> pd.DataFrame:
    df = filter_fact(lfact, season, last_gw)
    if df.empty: return pd.DataFrame(columns=["name"])
    gp = df.groupby("name")["match_id"].nunique().rename("GP")
    w = df[df["result"]=="W"].groupby("name")["match_id"].nunique().rename("W")
    d = df[df["result"]=="D"].groupby("name")["match_id"].nunique().rename("D")
    l = df[df["result"]=="L"].groupby("name")["match_id"].nunique().rename("L")
    g = df.groupby("name")["goals"].sum().rename("Goals")
    a = df.groupby("name")["assists"].sum().rename("Assists")
    ga = (g + a).rename("G+A")
    winp = ((w.fillna(0) / gp) * 100).round(1).rename("Win%")
    contrib = df.groupby("name")["contrib"].mean().round(1).rename("Team Contrib%")
    out = pd.concat([gp,w,d,l,g,a,ga,winp,contrib], axis=1).fillna(0)
    out = out[out["GP"] >= int(min_games)]
    out = out.sort_values(["G+A","Goals","Assists","Win%"], ascending=[False,False,False,False])
    out.index.name = "name"; return out.reset_index()

def duos_table(lfact: pd.DataFrame, season: Optional[int], min_games_together: int, last_gw: Optional[int]) -> pd.DataFrame:
    df = filter_fact(lfact, season, last_gw)
    if df.empty: return pd.DataFrame(columns=["A","B","GP","W","D","L","Win%","G+A"])
    same = df.merge(df, on=["match_id","team"])
    same = same[same["name_x"] < same["name_y"]]  # unique pairs
    grp = same.groupby(["name_x","name_y"])
    gp = grp["match_id"].nunique().rename("GP")
    w = same[same["result_x"]=="W"].groupby(["name_x","name_y"])["match_id"].nunique().rename("W")
    d = same[same["result_x"]=="D"].groupby(["name_x","name_y"])["match_id"].nunique().rename("D")
    l = same[same["result_x"]=="L"].groupby(["name_x","name_y"])["match_id"].nunique().rename("L")
    ga = (grp["goals_x"].sum() + grp["assists_x"].sum() + grp["goals_y"].sum() + grp["assists_y"].sum()).rename("G+A")
    out = pd.concat([gp,w,d,l,ga], axis=1).fillna(0)
    out["Win%"] = ((out["W"]/out["GP"])*100).round(1)
    out = out[out["GP"] >= int(min_games_together)]
    out = out.sort_values(["Win%","G+A","GP"], ascending=[False,False,False]).reset_index()
    out = out.rename(columns={"name_x":"A","name_y":"B"})
    return out

def nemesis_table(lfact: pd.DataFrame, season: Optional[int], min_meetings: int, last_gw: Optional[int]) -> pd.DataFrame:
    df = filter_fact(lfact, season, last_gw)
    if df.empty: return pd.DataFrame(columns=["Player","Nemesis","GP","W","D","L","Win%"])
    opp = df.merge(df, on="match_id")
    opp = opp[opp["team_x"] != opp["team_y"]]
    grp = opp.groupby(["name_x","name_y"])
    gp = grp["match_id"].nunique().rename("GP")
    w = opp[(opp["result_x"]=="W")].groupby(["name_x","name_y"])["match_id"].nunique().rename("W")
    d = opp[(opp["result_x"]=="D")].groupby(["name_x","name_y"])["match_id"].nunique().rename("D")
    l = opp[(opp["result_x"]=="L")].groupby(["name_x","name_y"])["match_id"].nunique().rename("L")
    out = pd.concat([gp,w,d,l], axis=1).fillna(0)
    out["Win%"] = ((out["W"]/out["GP"])*100).round(1)
    out = out[out["GP"] >= int(min_meetings)]
    out = out.sort_values(["Win%","GP"], ascending=[True,False]).reset_index().rename(columns={"name_x":"Player","name_y":"Nemesis"})
    return out

def form_string(results: List[str], n: int = 5) -> str:
    r = results[-n:][::-1]  # last n, most recent first
    out = []
    for x in r:
        if x == "W": out.append("🟩")
        elif x == "D": out.append("🟨")
        else: out.append("🟥")
    return "".join(out) if out else "—"

# -----------------------------------
# Storage: upload avatar
# -----------------------------------
def upload_avatar(file) -> Optional[str]:
    """Accept HEIC/JPG/PNG; convert HEIC to PNG; upload to public avatars bucket; return public URL."""
    if file is None:
        return None
    suffix = file.name.split(".")[-1].lower()
    img: Image.Image
    try:
        if suffix in ["heic","heif"] and HEIF_OK:
            heif_img = pillow_heif.read_heif(file.read())
            img = Image.frombytes(heif_img.mode, heif_img.size, heif_img.data, "raw")
        else:
            img = Image.open(file).convert("RGB")
        img = img.resize((512,512))
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        buf.seek(0)
        key = f"{uuid.uuid4().hex}.png"
        sb_service.storage.from_(AVATAR_BUCKET).upload(file=buf, path=key, file_options={"content-type":"image/png","upsert":"true"})
        pub = sb.storage.from_(AVATAR_BUCKET).get_public_url(key)
        return pub
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return None

# -----------------------------------
# Header / auth
# -----------------------------------
def header():
    left, mid, right = st.columns([3,2,3])
    with left:
        st.markdown("## Powerleague Stats")
    with mid:
        if st.button("Clear cache", use_container_width=True, key="clear_cache"):
            clear_caches(); st.success("Cache cleared."); st.rerun()
    with right:
        if not st.session_state.get("is_admin"):
            with st.expander("Admin login", expanded=False):
                pwd = st.text_input("Password", type="password", key="admin_pwd")
                if st.button("Login", use_container_width=True, key="btn_login"):
                    if ADMIN_PASSWORD and pwd == ADMIN_PASSWORD:
                        st.session_state["is_admin"] = True; st.success("Admin enabled"); st.rerun()
                    else:
                        st.error("Wrong password")
        else:
            st.markdown("**Admin:** ✅")
            if st.button("Log out", use_container_width=True, key="btn_logout"):
                st.session_state["is_admin"] = False; st.rerun()

# -----------------------------------
# Matches page (view + admin controls)
# -----------------------------------
def page_matches():
    matches = fetch_matches()
    lineups = fetch_lineups()
    players = fetch_players()

    if matches.empty:
        st.info("No matches yet. Use 'Add Match' to create one.")
        return

    # Select Season & GW
    seasons = sorted(matches["season"].dropna().astype(int).unique().tolist())
    colA, colB = st.columns(2)
    sel_season = colA.selectbox("Season", seasons, index=len(seasons)-1, key="pm_season")

    msub = matches[matches["season"] == sel_season].copy().sort_values("gw")
    labels = msub.apply(lambda r: f"GW {int(r['gw'])} — {r['team_a']} {int(r.get('score_a') or 0)}–{int(r.get('score_b') or 0)} {r['team_b']}", axis=1)
    id_map = {labels.iloc[i]: msub.iloc[i]["id"] for i in range(len(msub))}
    pick = colB.selectbox("Match", list(id_map.keys()), index=len(id_map)-1, key="pm_pick")
    mid = id_map[pick]
    m = msub[msub["id"] == mid].iloc[0]

    # Lineups filtered
    a_rows = lineups[(lineups["match_id"] == mid) & (lineups["team"] == "Non-bibs")].copy()
    b_rows = lineups[(lineups["match_id"] == mid) & (lineups["team"] == "Bibs")].copy()
    both = normalize_lineup_names(pd.concat([a_rows, b_rows], ignore_index=True))
    if both.empty:
        st.info("No lineup data yet for this match. Add players in the editor below.")

    # Banner
    lcol, ccol, rcol = st.columns([3, 2, 3])
    with lcol:  st.markdown(f"### **{m['team_a']}**")
    with ccol:
        st.markdown(f"### **{int(m.get('score_a') or 0)} – {int(m.get('score_b') or 0)}**")
        motm = str(m.get("motm_name") or "")
        if motm: st.caption(f"⭐ MOTM: **{motm}**")
    with rcol:  st.markdown(f"### **{m['team_b']}**")

    # Admin edit: quick match info
    if st.session_state.get("is_admin"):
        with st.expander("Edit match info (admin)", expanded=False):
            c1,c2,c3,c4,c5 = st.columns([1,1,1,1,1])
            sc_a = c1.number_input("Score (Non-bibs)", 0, 999, int(m.get("score_a") or 0), key=f"sc_a_{mid}")
            sc_b = c2.number_input("Score (Bibs)", 0, 999, int(m.get("score_b") or 0), key=f"sc_b_{mid}")
            motm_in = c3.text_input("MOTM name", value=str(m.get("motm_name") or ""), key=f"motm_{mid}")
            dt_val = pd.to_datetime(m.get("date") or date.today()).date()
            d = c4.date_input("Date", value=dt_val, key=f"dt_{mid}")
            side_count = int(m.get("side_count") or 5)
            side_new = c5.selectbox("Side count", [5,7], index=(0 if side_count==5 else 1), key=f"side_{mid}")
            if st.button("Save match info", key=f"save_m_{mid}"):
                s = service()
                if not s: st.error("Admin required.")
                else:
                    s.table("matches").update({
                        "score_a": int(sc_a),
                        "score_b": int(sc_b),
                        "motm_name": motm_in,
                        "date": str(d),
                        "side_count": int(side_new)
                    }).eq("id", mid).execute()
                    clear_caches(); st.success("Saved."); st.rerun()

    # Admin: change formations (validated)
    if st.session_state.get("is_admin"):
        presets5 = ["1-2-1","1-3","2-2","3-1"]
        presets7 = ["2-1-2-1","3-2-1","2-3-1"]
        side_count = int(m.get("side_count") or 5)
        options = presets7 if side_count == 7 else presets5

        colf1, colf2, colf3 = st.columns([2,2,1])
        fa_pick = colf1.selectbox("Formation (Non-bibs)", options,
                    index=(options.index(m.get("formation_a")) if m.get("formation_a") in options else 0),
                    key=f"fa_{mid}")
        fb_pick = colf2.selectbox("Formation (Bibs)", options,
                    index=(options.index(m.get("formation_b")) if m.get("formation_b") in options else 0),
                    key=f"fb_{mid}")
        if colf3.button("Save formations", key=f"save_forms_{mid}"):
            s = service()
            if not s: st.error("Admin required.")
            else:
                fa_s = validate_formation(fa_pick, side_count)
                fb_s = validate_formation(fb_pick, side_count)
                s.table("matches").update({"formation_a":fa_s,"formation_b":fb_s}).eq("id", mid).execute()
                clear_caches(); st.success("Saved."); st.rerun()

    # Combined pitch render (validated)
    side_count = int(m.get("side_count") or 5)
    fa_render = validate_formation(m.get("formation_a"), side_count)
    fb_render = validate_formation(m.get("formation_b"), side_count)
    st.caption(f"{m['team_a']} (left)  vs  {m['team_b']} (right)")
    render_match_pitch_combined(a_rows, b_rows, fa_render, fb_render, m.get("motm_name"), m["team_a"], m["team_b"], show_stats=True)

    # Admin: Arrange lineup + goals/assists/GK/positions
    if st.session_state.get("is_admin"):
        with st.expander("🧲 Arrange lineup (admin)", expanded=False):
            pl = fetch_players()
            # helper picker
            def team_editor(team_name: str, df_team: pd.DataFrame, keypref: str, formation: str):
                st.markdown(f"**{team_name}**")
                existing_names = set(df_team["name"].fillna(df_team.get("player_name")).dropna().astype(str))
                choices = [n for n in pl["name"].tolist() if n not in existing_names]
                add_name = st.selectbox("Add player", ["—"]+choices, key=f"{keypref}_add")
                if st.button(f"Add to {team_name}", key=f"{keypref}_add_btn") and add_name and add_name!="—":
                    # insert new lineup row (default position centered)
                    s = service()
                    if s:
                        s.table("lineups").insert({
                            "id": str(uuid.uuid4()),
                            "match_id": mid,
                            "team": team_name,
                            "player_name": add_name,
                            "name": add_name,
                            "is_gk": False,
                            "goals": 0, "assists": 0
                        }).execute()
                        clear_caches(); st.rerun()

                # table-like editor for current rows
                df = _ensure_positions(df_team.copy(), formation)
                for i, r in df.reset_index(drop=True).iterrows():
                    cols = st.columns([2,1,1,1,1,1,1])
                    cols[0].markdown(f"**{r['name']}**")
                    is_gk = cols[1].toggle("GK", value=bool(r.get("is_gk")), key=f"{keypref}_gk_{i}")
                    goals = cols[2].number_input("G", 0, 50, int(r.get("goals") or 0), key=f"{keypref}_g_{i}")
                    assists = cols[3].number_input("A", 0, 50, int(r.get("assists") or 0), key=f"{keypref}_a_{i}")
                    line = cols[4].number_input("Line", 0, max(0,len(formation_to_lines(formation))-1), int(r.get("line") or 0), key=f"{keypref}_l_{i}")
                    slot = cols[5].number_input("Slot", 0, 8, int(r.get("slot") or 0), key=f"{keypref}_s_{i}")
                    if cols[6].button("Remove", key=f"{keypref}_rm_{i}"):
                        s = service()
                        if s: s.table("lineups").delete().eq("id", r["id"]).execute()
                        clear_caches(); st.rerun()
                    # Save each row when changed via small save button
                    if st.button("Save row", key=f"{keypref}_sv_{i}"):
                        s = service()
                        if s:
                            s.table("lineups").update({
                                "is_gk": bool(is_gk),
                                "goals": int(goals),
                                "assists": int(assists),
                                "line": int(line),
                                "slot": int(slot)
                            }).eq("id", r["id"]).execute()
                            clear_caches(); st.success("Row saved."); st.rerun()

            colA, colB = st.columns(2)
            with colA: team_editor("Non-bibs", a_rows, f"edA_{mid}", fa_render)
            with colB: team_editor("Bibs", b_rows, f"edB_{mid}", fb_render)

# -----------------------------------
# Add Match (admin wizard)
# -----------------------------------
def page_add_match():
    st.markdown("### Add Match")
    if not st.session_state.get("is_admin"):
        st.info("Admin required.")
        return
    with st.form("add_match_form", clear_on_submit=False):
        col1,col2,col3 = st.columns(3)
        season = col1.number_input("Season", 2023, 2100, datetime.now().year)
        gw = col2.number_input("Gameweek", 1, 500, 1)
        side_count = col3.selectbox("Side count", [5,7], index=0)
        d = st.date_input("Date", value=date.today())
        team_a = st.text_input("Team A", value="Non-bibs")
        team_b = st.text_input("Team B", value="Bibs")
        default_form = "1-2-1" if side_count==5 else "2-1-2-1"
        formation_a = st.text_input("Formation A", value=default_form)
        formation_b = st.text_input("Formation B", value=default_form)
        notes = st.text_area("Notes", value="")
        submit = st.form_submit_button("Create match")
        if submit:
            s = service()
            if not s: st.error("Admin required.")
            else:
                mid = str(uuid.uuid4())
                s.table("matches").insert({
                    "id": mid, "season": int(season), "gw": int(gw), "side_count": int(side_count),
                    "team_a": team_a, "team_b": team_b,
                    "score_a": 0, "score_b": 0,
                    "date": str(d), "motm_name": None, "is_draw": False,
                    "formation_a": validate_formation(formation_a, side_count),
                    "formation_b": validate_formation(formation_b, side_count),
                    "notes": notes
                }).execute()
                clear_caches(); st.success(f"Match GW{int(gw)} created."); st.rerun()

# -----------------------------------
# Players Manager (add/edit, upload avatar)
# -----------------------------------
def page_player_manager():
    st.markdown("### Player Manager")
    if not st.session_state.get("is_admin"):
        st.info("Admin required.")
        return

    pl = fetch_players()

    with st.expander("Add player", expanded=False):
        c1,c2 = st.columns([2,1])
        name_new = c1.text_input("Name", key="pm_name_new")
        photo = c2.file_uploader("Avatar (HEIC/JPG/PNG)", type=["heic","heif","jpg","jpeg","png"], key="pm_photo_up")
        notes_new = st.text_area("Notes", key="pm_notes_new")
        if st.button("Create player", key="pm_create"):
            if not name_new.strip():
                st.error("Name required."); st.stop()
            url = upload_avatar(photo) if photo else None
            s = service()
            if s:
                s.table("players").insert({
                    "id": str(uuid.uuid4()), "name": name_new.strip(),
                    "photo_url": url, "notes": notes_new
                }).execute()
                clear_caches(); st.success("Player created."); st.rerun()

    st.divider()
    st.markdown("#### Edit existing")
    for _, r in pl.iterrows():
        with st.expander(r["name"], expanded=False):
            c1,c2,c3 = st.columns([2,1,1])
            nm = c1.text_input("Name", value=r["name"], key=f"pm_nm_{r['id']}")
            nt = c1.text_area("Notes", value=r.get("notes") or "", key=f"pm_nt_{r['id']}")
            photo = c2.file_uploader("Replace photo", type=["heic","heif","jpg","jpeg","png"], key=f"pm_up_{r['id']}")
            if c2.button("Save", key=f"pm_save_{r['id']}"):
                s = service()
                if s:
                    url = upload_avatar(photo) if photo else r.get("photo_url")
                    s.table("players").update({"name":nm,"notes":nt,"photo_url":url}).eq("id", r["id"]).execute()
                    clear_caches(); st.success("Saved."); st.rerun()
            if c3.button("Delete", key=f"pm_del_{r['id']}"):
                s = service()
                if s:
                    s.table("players").delete().eq("id", r["id"]).execute()
                    clear_caches(); st.success("Deleted."); st.rerun()

# -----------------------------------
# Stats page
# -----------------------------------
def page_stats():
    players = fetch_players()
    matches = fetch_matches()
    lineups = fetch_lineups()
    lfact = build_fact(players, matches, lineups)
    if lfact.empty:
        st.info("No data yet.")
        return

    st.markdown("### Stats")

    # Filters
    seasons = [-1] + sorted(lfact["season"].dropna().astype(int).unique().tolist())
    c1,c2,c3,c4 = st.columns(4)
    sel_season = c1.selectbox("Season (or All)", seasons, format_func=lambda x: ("All" if x==-1 else x), index=len(seasons)-1, key="st_season")
    min_games = c2.number_input("Min games", 0, 100, 1, key="st_min")
    last_gw = c3.number_input("Last N GWs (0 = all)", 0, 200, 0, key="st_last")
    top_n = c4.number_input("Rows", 5, 200, 25, key="st_rows")

    metric = st.selectbox("Metric", [
        "Top Scorers","Top Assisters","Top G+A","Team Contribution%","MOTM Count","Best Duos","Nemesis"
    ], key="st_metric")

    if metric in ["Top Scorers","Top Assisters","Top G+A","Team Contribution%","MOTM Count"]:
        agg = player_agg(lfact, None if sel_season==-1 else int(sel_season), int(min_games), int(last_gw))
        if metric == "Top Scorers":
            out = agg.sort_values(["Goals","G+A","Win%"], ascending=[False,False,False]).head(int(top_n))
        elif metric == "Top Assisters":
            out = agg.sort_values(["Assists","G+A","Win%"], ascending=[False,False,False]).head(int(top_n))
        elif metric == "Top G+A":
            out = agg.sort_values(["G+A","Goals","Assists"], ascending=[False,False,False]).head(int(top_n))
        elif metric == "Team Contribution%":
            out = agg.sort_values(["Team Contrib%","G+A","GP"], ascending=[False,False,False]).head(int(top_n))
        else:  # MOTM Count derived from matches
            m = fetch_matches().copy()
            cnt = m["motm_name"].dropna().value_counts().rename_axis("name").reset_index(name="MOTM")
            out = agg.merge(cnt, on="name", how="left").fillna({"MOTM":0}).sort_values(["MOTM","G+A"], ascending=[False,False]).head(int(top_n))
        st.dataframe(out, use_container_width=True, hide_index=True)
    elif metric == "Best Duos":
        out = duos_table(lfact, None if sel_season==-1 else int(sel_season), int(min_games), int(last_gw)).head(int(top_n))
        st.dataframe(out, use_container_width=True, hide_index=True)
    else:  # Nemesis
        out = nemesis_table(lfact, None if sel_season==-1 else int(sel_season), int(min_games), int(last_gw)).head(int(top_n))
        st.dataframe(out, use_container_width=True, hide_index=True)

# -----------------------------------
# Player Profiles
# -----------------------------------
def page_players():
    players = fetch_players()
    matches = fetch_matches()
    lineups = fetch_lineups()
    lfact = build_fact(players, matches, lineups)

    if players.empty:
        st.info("No players yet. Add via Player Manager.")
        return

    names = players["name"].tolist()
    sel = st.selectbox("Player", names, key="pp_pick")

    mine = lfact[lfact["name"] == sel].copy().sort_values(["season","gw"])
    if mine.empty:
        st.info("No games recorded for this player yet.")
        return

    # Overview numbers
    gp = mine["match_id"].nunique()
    w = (mine["result"]=="W").sum()
    d = (mine["result"]=="D").sum()
    l = (mine["result"]=="L").sum()
    goals = mine["goals"].sum(); assists = mine["assists"].sum()
    ga = goals + assists
    gapg = (ga / gp) if gp else 0
    contrib = mine["contrib"].mean() if not mine.empty else 0

    # Form string last N (adjustable)
    n = st.number_input("Last N games", 1, max(1,gp), min(5,gp), key="pp_last")
    frm = form_string(mine["result"].tolist(), n=n)

    # Best teammate / Nemesis
    duo = duos_table(lfact[lfact["name"].isin([sel]) | (lfact["name"]==sel)], None, 1, 0)
    best_t = None
    if not duo.empty:
        best_t = duo[(duo["A"]==sel) | (duo["B"]==sel)].copy()
        best_t["Mate"] = np.where(best_t["A"]==sel, best_t["B"], best_t["A"])
        best_t = best_t.sort_values(["Win%","GP"], ascending=[False,False]).head(1)
    neme = nemesis_table(lfact, None, 1, 0)
    my_nem = None
    if not neme.empty:
        my_nem = neme[neme["Player"]==sel].sort_values(["Win%","GP"], ascending=[True,False]).head(1)

    # Avatar
    pr = players[players["name"]==sel].iloc[0]
    avatar = pr.get("photo_url") or None
    av_html = (
        f"<img src='{avatar}' style='width:96px;height:96px;border-radius:50%;object-fit:cover;border:2px solid rgba(255,255,255,.2)'>"
        if avatar else
        f"<div style='width:96px;height:96px;border-radius:50%;background:#1a2430;color:#e9eef3;display:flex;align-items:center;justify-content:center;font-weight:800'>{initials(sel)}</div>"
    )

    # Top panel
    st.markdown(f"""
    <div class='badge'>
      {av_html}
      <div style='display:flex;flex-direction:column;gap:.2rem'>
        <div style='font-weight:800;font-size:1.1rem'>{sel}</div>
        <div class='small'>GP {gp} • W-D-L {w}-{d}-{l} • Win% {(w/gp*100 if gp else 0):.1f}</div>
        <div class='small'>Goals {goals} • Assists {assists} • G+A {ga} • G+A/GP {gapg:.2f}</div>
        <div class='small'>Team Contribution% {contrib:.1f}</div>
        <div>Form: {frm}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Recent games
    st.markdown("#### Recent games")
    recent = mine.sort_values(["season","gw"], ascending=[False,False]).head(int(n))
    show = recent[["season","gw","team","for","against","result","goals","assists"]].rename(columns={
        "for":"For","against":"Ag","result":"Res","goals":"G","assists":"A"
    })
    st.dataframe(show, use_container_width=True, hide_index=True)

    # Best teammate / Nemesis rows
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("#### Best teammate")
        if best_t is not None and not best_t.empty:
            st.dataframe(best_t[["Mate","GP","W","D","L","Win%","G+A"]], use_container_width=True, hide_index=True)
        else:
            st.caption("—")
    with c2:
        st.markdown("#### Nemesis")
        if my_nem is not None and not my_nem.empty:
            st.dataframe(my_nem[["Nemesis","GP","W","D","L","Win%"]], use_container_width=True, hide_index=True)
        else:
            st.caption("—")

# -----------------------------------
# Awards page
# -----------------------------------
def page_awards():
    matches = fetch_matches()
    awards = fetch_awards()

    st.markdown("### Awards")
    st.markdown("**MOTM (from matches):**")
    if matches.empty:
        st.caption("No matches.")
    else:
        motm = matches["motm_name"].dropna().value_counts().rename_axis("name").reset_index(name="MOTM")
        st.dataframe(motm, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("**POTM/MOTM (manual):**")
    if not awards.empty:
        view = awards.copy()
        view = view.sort_values(["season","month","type","gw"])
        st.dataframe(view, use_container_width=True, hide_index=True)
    else:
        st.caption("No manual awards yet.")

    if st.session_state.get("is_admin"):
        with st.expander("Add manual award", expanded=False):
            c1,c2,c3,c4,c5,c6 = st.columns(6)
            season = c1.number_input("Season", 2023, 2100, datetime.now().year, key="aw_s")
            month = c2.number_input("Month", 1, 12, datetime.now().month, key="aw_m")
            typ = c3.selectbox("Type", ["MOTM","POTM"], key="aw_t")
            gw = c4.number_input("GW (optional)", 0, 500, 0, key="aw_gw")
            pl = fetch_players()
            pname = c5.selectbox("Player", pl["name"].tolist() if not pl.empty else [], key="aw_p")
            notes = c6.text_input("Notes", key="aw_n")
            if st.button("Save award", key="aw_save"):
                s = service()
                if s:
                    s.table("awards").insert({
                        "id": str(uuid.uuid4()), "season": int(season), "month": int(month),
                        "type": typ, "gw": (int(gw) if gw else None),
                        "player_id": None, "player_name": pname, "notes": notes
                    }).execute()
                    clear_caches(); st.success("Saved."); st.rerun()

# -----------------------------------
# Import / Export
# -----------------------------------
def page_import_export():
    st.markdown("### Import / Export")
    if st.session_state.get("is_admin"):
        st.caption("Import order: players → matches → lineups")

        with st.expander("Import players.csv", expanded=False):
            up = st.file_uploader("Upload CSV", type=["csv"], key="imp_p")
            if up and st.button("Import players", key="imp_p_btn"):
                df = pd.read_csv(up)
                rows = []
                for _, r in df.iterrows():
                    rows.append({
                        "id": r.get("id") or str(uuid.uuid4()),
                        "name": r["name"],
                        "photo_url": r.get("photo_url") or None,
                        "notes": r.get("notes") or None
                    })
                s = service(); s.table("players").upsert(rows, on_conflict="name").execute()
                clear_caches(); st.success("Players upserted.")

        with st.expander("Import matches.csv", expanded=False):
            up = st.file_uploader("Upload CSV", type=["csv"], key="imp_m")
            if up and st.button("Import matches", key="imp_m_btn"):
                df = pd.read_csv(up)
                df["side_count"] = pd.to_numeric(df.get("side_count"), errors="coerce").fillna(5).astype(int)
                df["formation_a"] = df.apply(lambda r: validate_formation(r.get("formation_a"), int(r["side_count"])), axis=1)
                df["formation_b"] = df.apply(lambda r: validate_formation(r.get("formation_b"), int(r["side_count"])), axis=1)
                rows = df.to_dict("records")
                s = service(); s.table("matches").upsert(rows, on_conflict="season,gw").execute()
                clear_caches(); st.success("Matches upserted.")

        with st.expander("Import lineups.csv", expanded=False):
            up = st.file_uploader("Upload CSV", type=["csv"], key="imp_l")
            if up and st.button("Import lineups", key="imp_l_btn"):
                df = pd.read_csv(up)
                # delete-then-insert per (match_id, team)
                s = service()
                if not s: st.error("Admin required.")
                else:
                    for (mid, team), grp in df.groupby(["match_id","team"]):
                        s.table("lineups").delete().eq("match_id", mid).eq("team", team).execute()
                        # insert in chunks
                        recs = grp.to_dict("records")
                        chunk = 500
                        for i in range(0, len(recs), chunk):
                            s.table("lineups").insert(recs[i:i+chunk]).execute()
                    clear_caches(); st.success("Lineups imported.")

    st.divider()
    st.markdown("#### Export")
    pl = fetch_players(); mt = fetch_matches(); ln = fetch_lineups()
    col1,col2,col3 = st.columns(3)
    col1.download_button("players.csv", pl.to_csv(index=False).encode("utf-8"), "players.csv", "text/csv")
    col2.download_button("matches.csv", mt.to_csv(index=False).encode("utf-8"), "matches.csv", "text/csv")
    col3.download_button("lineups.csv", ln.to_csv(index=False).encode("utf-8"), "lineups.csv", "text/csv")

# -----------------------------------
# Router
# -----------------------------------
def run_app():
    header()
    st.divider()
    page = st.sidebar.radio("Go to", ["Matches","Add Match","Players","Stats","Awards","Import/Export","Player Manager"], index=0, key="nav")
    if page == "Matches": page_matches()
    elif page == "Add Match": page_add_match()
    elif page == "Players": page_players()
    elif page == "Stats": page_stats()
    elif page == "Awards": page_awards()
    elif page == "Import/Export": page_import_export()
    elif page == "Player Manager": page_player_manager()

if __name__ == "__main__":
    run_app()
