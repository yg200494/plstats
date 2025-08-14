# app.py — Powerleague Stats (Drag & Drop Final)
# - Pitch: centered, FotMob-style, team-colored rings, modern chips, MOTM star
# - Matches: dropdown selector, add/update matches (past & future), inline G/A + MOTM editor
# - Drag & Drop lineup editor (GK + per-slot containers) -> saves is_gk/line/slot
# - Stats: dropdown metrics (incl. duos/nemesis) + Season/Min Games/Last N/Top N filters
# - Player Profile: hero, cards, streaks, last-N, duos & nemesis
# - Player Manager: add/edit/rename player, upload JPG/PNG/HEIC -> square PNG to Supabase Storage
# - Admin writes guarded by ADMIN_PASSWORD + SERVICE KEY

from __future__ import annotations
import io
import base64
import datetime as dt
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from supabase import create_client, Client

from PIL import Image
import pillow_heif

# Enable HEIC support via Pillow
try:
    pillow_heif.register_heif_opener()
except Exception:
    pass

# ---------------------------------
# App config + CSS
# ---------------------------------
st.set_page_config(page_title="Powerleague Stats", page_icon="⚽", layout="wide", initial_sidebar_state="collapsed")

def load_css():
    try:
        with open("styles/styles.css", "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass

load_css()

# ---------------------------------
# Secrets & Supabase clients
# ---------------------------------
REQ = ["SUPABASE_URL", "SUPABASE_ANON_KEY", "ADMIN_PASSWORD", "AVATAR_BUCKET"]
missing = [k for k in REQ if k not in st.secrets]
if missing:
    st.error(f"Missing required secrets: {', '.join(missing)}")
    st.stop()

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
SUPABASE_SERVICE_KEY = st.secrets.get("SUPABASE_SERVICE_KEY")  # required for writes
ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
AVATAR_BUCKET = st.secrets["AVATAR_BUCKET"]

sb_public: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

def service() -> Optional[Client]:
    if st.session_state.get("is_admin") and SUPABASE_SERVICE_KEY:
        return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return None

# ---------------------------------
# Cached reads
# ---------------------------------
@st.cache_data(ttl=20)
def fetch_players() -> pd.DataFrame:
    data = sb_public.table("players").select("*").execute().data or []
    df = pd.DataFrame(data)
    if "id" in df.columns: df["id"] = df["id"].astype(str)
    return df

@st.cache_data(ttl=20)
def fetch_matches() -> pd.DataFrame:
    data = sb_public.table("matches").select("*").order("season").order("gw").execute().data or []
    df = pd.DataFrame(data)
    if "id" in df.columns: df["id"] = df["id"].astype(str)
    return df

@st.cache_data(ttl=20)
def fetch_lineups() -> pd.DataFrame:
    data = sb_public.table("lineups").select("*").execute().data or []
    df = pd.DataFrame(data)
    if "id" in df.columns: df["id"] = df["id"].astype(str)
    return df

@st.cache_data(ttl=20)
def fetch_awards() -> pd.DataFrame:
    data = sb_public.table("awards").select("*").execute().data or []
    df = pd.DataFrame(data)
    if "id" in df.columns: df["id"] = df["id"].astype(str)
    return df

def clear_caches():
    fetch_players.clear(); fetch_matches.clear(); fetch_lineups.clear(); fetch_awards.clear()

# ---------------------------------
# Helpers
# ---------------------------------
TEAM_COLORS = {"Non-bibs": "#2f7ef0", "Bibs": "#ff8a1c"}

def initials(name: str) -> str:
    parts = [p[0] for p in str(name).split() if p]
    return "".join(parts[:2]).upper() or "?"

def _to_int(s):
    return pd.to_numeric(s, errors="coerce")

def formation_to_lines(form: str | None) -> List[int]:
    if not form: return [1,2,1]
    try:
        parts = [int(x) for x in str(form).split("-") if x.strip().isdigit()]
        return parts if parts else [1,2,1]
    except Exception:
        return [1,2,1]

def build_fact(players: pd.DataFrame, matches: pd.DataFrame, lineups: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Join lineup rows with match metadata; unify season/gw columns; derive results."""
    if matches.empty or lineups.empty:
        return pd.DataFrame(columns=[
            "match_id","season","gw","date","team","team_a","team_b","score_a","score_b","is_draw",
            "motm_name","formation_a","formation_b","player_id","player_name","name","photo",
            "goals","assists","ga","result","team_goals","opp_goals","is_gk","line","slot"
        ]), matches

    m = matches.copy(); l = lineups.copy(); p = players.copy()
    for c in ["season","gw","score_a","score_b","side_count"]:
        if c in m.columns: m[c] = _to_int(m[c])
    for c in ["season","gw","goals","assists","line","slot"]:
        if c in l.columns: l[c] = _to_int(l[c])

    keep = ["id","season","gw","date","team_a","team_b","score_a","score_b","is_draw","motm_name","formation_a","formation_b"]
    j = l.merge(m[keep].rename(columns={"id":"match_id"}), on="match_id", how="left", suffixes=("_lu","_ma"))

    def pick(a, b):
        if a is None and b is None: return pd.Series(dtype="float64")
        a = _to_int(a) if a is not None else None
        b = _to_int(b) if b is not None else None
        if a is None: return b
        if b is None: return a
        return a.where(a.notna(), b)

    j["season"] = pick(j.get("season_lu"), j.get("season_ma"))
    j["gw"]     = pick(j.get("gw_lu"),     j.get("gw_ma"))
    for c in ["season_lu","season_ma","gw_lu","gw_ma"]:
        if c in j.columns: j.drop(columns=c, inplace=True)

    if not p.empty:
        pp = p.rename(columns={"id":"player_id"}).copy()
        j = j.merge(pp[["player_id","name","photo_url"]], on="player_id", how="left")
        j["name"] = j["player_name"].where(
            j["player_name"].notna() & (j["player_name"].astype(str).str.strip()!=""),
            j["name"]
        )
        j["photo"] = j["photo_url"]
    else:
        j["name"] = j["player_name"]; j["photo"] = None

    j["side"] = np.where(j["team"]==j["team_a"], "A", np.where(j["team"]==j["team_b"], "B", None))
    j["team_goals"] = np.where(j["side"]=="A", j["score_a"], np.where(j["side"]=="B", j["score_b"], np.nan))
    j["opp_goals"]  = np.where(j["side"]=="A", j["score_b"], np.where(j["side"]=="B", j["score_a"], np.nan))
    j["result"] = np.where(j["is_draw"]==True, "D", np.where(_to_int(j["team_goals"])>_to_int(j["opp_goals"]), "W", "L"))
    j["goals"] = _to_int(j["goals"]).fillna(0).astype(int)
    j["assists"] = _to_int(j["assists"]).fillna(0).astype(int)
    j["ga"] = (j["goals"] + j["assists"]).astype(int)
    j["is_gk"] = j.get("is_gk", False).astype(bool)

    return j, matches

def df_filter_by(df: pd.DataFrame, season: Optional[int], last_gw: int) -> pd.DataFrame:
    d = df.copy()
    if season is not None:
        d = d[d["season"]==season]
    if last_gw and last_gw > 0 and not d.empty:
        max_gw = _to_int(d["gw"]).max()
        if pd.notna(max_gw):
            d = d[_to_int(d["gw"]) >= (int(max_gw)-int(last_gw)+1)]
    return d

def player_agg(l: pd.DataFrame, season=None, min_games=0, last_gw=None) -> pd.DataFrame:
    df = df_filter_by(l, season, last_gw or 0).copy()
    if df.empty:
        return pd.DataFrame(columns=["name","gp","w","d","l","win_pct","goals","assists","ga","ga_pg","g_pg","a_pg","team_contrib_pct","photo"])
    df["team_goals"] = _to_int(df["team_goals"])
    gp = df.groupby("name").size().rename("gp")
    w  = df[df["result"]=="W"].groupby("name").size().reindex(gp.index, fill_value=0)
    d  = df[df["result"]=="D"].groupby("name").size().reindex(gp.index, fill_value=0)
    l_ = df[df["result"]=="L"].groupby("name").size().reindex(gp.index, fill_value=0)
    goals = df.groupby("name")["goals"].sum().reindex(gp.index, fill_value=0)
    assists = df.groupby("name")["assists"].sum().reindex(gp.index, fill_value=0)
    ga = goals + assists
    ga_pg = (ga / gp.replace(0,np.nan)).round(2).fillna(0)
    g_pg  = (goals / gp.replace(0,np.nan)).round(2).fillna(0)
    a_pg  = (assists / gp.replace(0,np.nan)).round(2).fillna(0)
    team_goals_sum = df.groupby("name")["team_goals"].sum(min_count=1).reindex(gp.index)
    team_contrib = ((ga / team_goals_sum.replace(0,np.nan))*100).round(1).fillna(0)
    win_pct = ((w.values/np.maximum(gp.values,1))*100).round(1)
    photo = df.groupby("name")["photo"].last().reindex(gp.index)
    out = pd.DataFrame({
        "name": gp.index, "gp": gp.values, "w": w.values, "d": d.values, "l": l_.values,
        "win_pct": win_pct, "goals": goals.values.astype(int), "assists": assists.values.astype(int),
        "ga": ga.values.astype(int), "ga_pg": ga_pg.values, "g_pg": g_pg.values, "a_pg": a_pg.values,
        "team_contrib_pct": team_contrib.values, "photo": photo.values
    }).sort_values(["ga","goals","assists"], ascending=False).reset_index(drop=True)
    if min_games>0: out = out[out["gp"]>=min_games]
    return out

def duos_global(lfact: pd.DataFrame, min_gp: int = 3, season=None, last_gw=None):
    df = df_filter_by(lfact, season, last_gw or 0)
    a = df.merge(df, on=["match_id","team"], suffixes=("_a","_b"))
    a = a[a["name_a"] < a["name_b"]]
    if a.empty: return pd.DataFrame(columns=["duo","gp","w","d","l","win_pct"])
    gp = a.groupby(["name_a","name_b"]).size()
    w = a[a["result_a"]=="W"].groupby(["name_a","name_b"]).size().reindex(gp.index, fill_value=0)
    d = a[a["result_a"]=="D"].groupby(["name_a","name_b"]).size().reindex(gp.index, fill_value=0)
    l = a[a["result_a"]=="L"].groupby(["name_a","name_b"]).size().reindex(gp.index, fill_value=0)
    out = pd.DataFrame({"gp":gp,"w":w,"d":d,"l":l}).reset_index()
    out["duo"] = out["name_a"]+" + "+out["name_b"]
    out["win_pct"] = ((out["w"]/out["gp"])*100).round(1)
    out = out[out["gp"]>=min_gp].sort_values(["win_pct","gp"], ascending=[False,False])[["duo","gp","w","d","l","win_pct"]]
    return out

def nemesis_global(lfact: pd.DataFrame, min_gp: int = 3, season=None, last_gw=None):
    df = df_filter_by(lfact, season, last_gw or 0)
    a = df.merge(df, on=["match_id"], suffixes=("_a","_b"))
    a = a[a["team_a"]!=a["team_b"]]
    a = a[a["name_a"] < a["name_b"]]
    if a.empty: return pd.DataFrame(columns=["pair","gp","w_vs","d_vs","l_vs","win_pct_vs"])
    gp = a.groupby(["name_a","name_b"]).size()
    w_a = a[a["result_a"]=="W"].groupby(["name_a","name_b"]).size().reindex(gp.index, fill_value=0)
    d_a = a[a["result_a"]=="D"].groupby(["name_a","name_b"]).size().reindex(gp.index, fill_value=0)
    l_a = a[a["result_a"]=="L"].groupby(["name_a","name_b"]).size().reindex(gp.index, fill_value=0)
    out = pd.DataFrame({"gp":gp,"w_vs":w_a,"d_vs":d_a,"l_vs":l_a}).reset_index()
    out["pair"] = out["name_a"]+" vs "+out["name_b"]
    out["win_pct_vs"] = ((out["w_vs"]/out["gp"])*100).round(1)
    out = out[out["gp"]>=min_gp].sort_values(["win_pct_vs","gp"], ascending=[True,False])[["pair","gp","w_vs","d_vs","l_vs","win_pct_vs"]]
    return out

def compute_streak(series_vals, predicate):
    count=0
    for v in list(series_vals)[::-1]:
        if predicate(v): count+=1
        else: break
    return count

# ---------------------------------
# Pitch rendering
# ---------------------------------
def _pitch_svg() -> str:
    return """
    <svg viewBox="0 0 100 150" preserveAspectRatio="none">
      <defs>
        <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#1b7a47"/>
          <stop offset="100%" stop-color="#15653b"/>
        </linearGradient>
      </defs>
      <rect x="0" y="0" width="100" height="150" fill="url(#g)" />
      <rect x="2" y="3" width="96" height="144" fill="none" stroke="#cdebd8" stroke-width="1.1" rx="2"/>
      <line x1="2" y1="75" x2="98" y2="75" stroke="#cdebd8" stroke-width="1"/>
      <circle cx="50" cy="75" r="16" fill="none" stroke="#cdebd8" stroke-width="1"/>
      <rect x="12" y="3" width="76" height="24" fill="none" stroke="#cdebd8" stroke-width="1"/>
      <rect x="12" y="123" width="76" height="24" fill="none" stroke="#cdebd8" stroke-width="1"/>
      <rect x="30" y="3" width="40" height="12" fill="none" stroke="#cdebd8" stroke-width="1"/>
      <rect x="30" y="135" width="40" height="12" fill="none" stroke="#cdebd8" stroke-width="1"/>
    </svg>
    """

def _avatar_html(name: str, photo_url: Optional[str], team: str, motm: bool) -> str:
    border = TEAM_COLORS.get(team, "#ddd")
    if photo_url and str(photo_url).strip():
        img = f"<img src='{photo_url}' class='pl-avatar-img' />"
    else:
        init = initials(name)
        img = f"<div class='pl-avatar-init'>{init}</div>"
    star = "<div class='pl-motm'>★</div>" if motm else ""
    return f"<div class='pl-avatar' style='border-color:{border};'>{img}{star}</div>"

def _ensure_positions(team_df: pd.DataFrame, formation: str) -> pd.DataFrame:
    rows = team_df.copy()
    parts = formation_to_lines(formation)
    max_slots = max(parts+[1])
    if rows.empty: return rows

    rows["line"] = _to_int(rows.get("line")).astype("Int64")
    rows["slot"] = _to_int(rows.get("slot")).astype("Int64")

    if rows["line"].isna().all() and rows["slot"].isna().all():
        gk = rows[rows.get("is_gk", False)==True]
        others = rows.drop(index=gk.index if not gk.empty else [], errors="ignore")
        if not gk.empty:
            rows.loc[gk.index[0],["line","slot"]] = [len(parts), max_slots//2]
        cur=0; filled=[0]*len(parts)
        for idx,_ in others.iterrows():
            if cur>=len(parts): cur=0
            slots=parts[cur]; offset=(max_slots-slots)//2
            pos=filled[cur] % slots
            rows.loc[idx,["line","slot"]] = [cur, offset+pos]
            filled[cur]+=1; cur+=1

    for idx, r in rows.iterrows():
        if bool(r.get("is_gk")):
            rows.loc[idx,"line"] = len(parts)
            rows.loc[idx,"slot"] = max_slots//2
        else:
            li = 0 if pd.isna(r.get("line")) else int(r["line"])
            li = min(max(li,0), len(parts)-1)
            slots = parts[li]
            offset = (max_slots - slots)//2
            s = offset if pd.isna(r.get("slot")) else int(r["slot"])
            s = min(max(s, offset), offset + slots - 1)
            rows.loc[idx,"line"] = li
            rows.loc[idx,"slot"] = s
    return rows

def render_pitch(team_df: pd.DataFrame, formation: str, motm_name: Optional[str], team_label: str, show_stats=True, show_photos=True):
    rows = _ensure_positions(team_df, formation)
    parts = formation_to_lines(formation)
    max_slots = max(parts+[1])
    total_rows = len(parts)+1  # + GK row

    TOP_PAD = 18
    BOT_PAD = 18

    def y_for(i: int) -> float:
        if total_rows <= 1:
            return 50.0
        usable = 100.0 - TOP_PAD - BOT_PAD
        return TOP_PAD + usable * (i / (total_rows - 1))

    html = [f"<div class='pl-pitch'>{_pitch_svg()}"]
    for _, r in rows.iterrows():
        is_gk = bool(r.get("is_gk"))
        row_idx = len(parts) if is_gk else int(r.get("line"))
        slots_row = 1 if is_gk else parts[row_idx]
        offset_row = 0 if is_gk else (max_slots - slots_row)//2
        abs_slot = int(r.get("slot") if pd.notna(r.get("slot")) else (max_slots//2 if is_gk else offset_row))
        rel_slot = max(0, min(abs_slot - offset_row, slots_row-1))
        x = (100.0 * (rel_slot + 1) / (slots_row + 1))
        y = y_for(row_idx)

        name = r.get("name","")
        g = int(r.get("goals") or 0); a = int(r.get("assists") or 0)
        is_m = (motm_name and str(name).strip()==str(motm_name).strip())
        photo = (r.get("photo") if show_photos else None)
        avatar = _avatar_html(name, photo, team_label, is_m)

        chips = []
        if show_stats and g>0: chips.append(f"<div class='pl-chip pl-chip-g'>⚽ {g}</div>")
        if show_stats and a>0: chips.append(f"<div class='pl-chip pl-chip-a'>🅰 {a}</div>")
        chips_html = f"<div class='pl-chips'>{''.join(chips)}</div>" if chips else ""

        html.append(
            f"<div class='pl-spot' style='left:{x}%;top:{y}%;'>"
            f"{avatar}"
            f"<div class='pl-name'>{name}</div>"
            f"{chips_html}"
            f"</div>"
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)

# ---------------------------------
# Drag & Drop support (optional)
# ---------------------------------
try:
    from streamlit_sortables import sort_items as _sort_items
    _DND_AVAILABLE = True
except Exception:
    _DND_AVAILABLE = False

def _encode_item(row: pd.Series) -> str:
    return f"{row['id']}|{row.get('name') or row.get('player_name') or ''}"

def _decode_item(token: str) -> tuple[str, str]:
    pid, name = token.split("|", 1)
    return pid, name

def _containers_from_team_rows(team_rows: pd.DataFrame, formation: str, keyprefix: str):
    rows = _ensure_positions(team_rows, formation)
    parts = formation_to_lines(formation)
    max_slots = max(parts+[1])

    gk_items, non_gk = [], []
    for _, r in rows.iterrows():
        if bool(r.get("is_gk")): gk_items.append(_encode_item(r))
        else: non_gk.append(r)

    containers = []
    containers.append({"header": "GK", "items": gk_items[:1]})

    placed_ids = set()
    for i, slots in enumerate(parts):
        offset = (max_slots - slots)//2
        rel_to_abs = [offset + j for j in range(slots)]
        for j, abs_slot in enumerate(rel_to_abs):
            header = f"R{i+1} S{j+1}"
            items = []
            for r in non_gk:
                if int(r.get("line") or -1) == i and int(r.get("slot") or -999) == abs_slot:
                    items.append(_encode_item(r)); placed_ids.add(r["id"])
            containers.append({"header": header, "items": items})

    bench_items = []
    if len(gk_items) > 1:
        bench_items.extend(gk_items[1:])
    for _, r in rows.iterrows():
        tok = _encode_item(r)
        if (r["id"] not in placed_ids) and (tok not in bench_items) and (not bool(r.get("is_gk"))):
            bench_items.append(tok)
    containers.append({"header":"Bench","items":bench_items})
    return containers

def _apply_dnd_result(containers: list[dict], formation: str) -> list[dict]:
    parts = formation_to_lines(formation)
    max_slots = max(parts+[1])

    updates = []

    def header_to_pos(h: str) -> tuple[int|None, int|None]:
        if h == "GK": return None, None
        if not h.startswith("R"): return None, None
        try:
            row_txt, slot_txt = h.split()
            row_i = int(row_txt[1:]) - 1
            rel_j = int(slot_txt[1:]) - 1
            slots = parts[row_i]; offset = (max_slots - slots)//2
            return row_i, offset + rel_j
        except Exception:
            return None, None

    gk_set = set()
    for c in containers:
        if c["header"] == "GK":
            if c["items"]:
                pid, _ = _decode_item(c["items"][0])
                updates.append({"id": pid, "is_gk": True, "line": None, "slot": None})
                gk_set.add(pid)

    for c in containers:
        h = c["header"]
        if h in ("GK","Bench"): continue
        line, abs_slot = header_to_pos(h)
        for tok in c["items"]:
            pid, _ = _decode_item(tok)
            updates.append({"id": pid, "is_gk": False, "line": int(line), "slot": int(abs_slot)})

    seen = {}
    for u in updates:
        seen[u["id"]] = u
    return list(seen.values())

def dnd_lineup_editor(team_rows: pd.DataFrame, formation: str, team_label: str, keypref: str, mid: str):
    if not _DND_AVAILABLE:
        st.info("Install `streamlit-sortables` to enable drag & drop. Falling back to manual editing.")
        return None
    containers = _containers_from_team_rows(team_rows, formation, keypref)
    st.caption(f"Drag players into **GK** or any slot (formation {formation}).")
    out = _sort_items(containers, multi_containers=True, key=f"{keypref}_{mid}")
    bench = next((c for c in out if c["header"]=="Bench"), {"items":[]})
    if bench["items"]:
        st.warning("Players in **Bench** keep their previous positions unless you assign them to a slot.")
    return _apply_dnd_result(out, formation)

# ---------------------------------
# Admin/auth header  (popover-safe)
# ---------------------------------
def header():
    left, right = st.columns([1,1])
    with left:
        st.title("⚽ Powerleague Stats")

    with right:
        # ensure state exists
        if "is_admin" not in st.session_state:
            st.session_state["is_admin"] = False

        if st.session_state["is_admin"]:
            st.success("Admin mode", icon="🔐")
            if st.button("Logout", key="btn_logout"):
                st.session_state["is_admin"] = False
                st.rerun()
        else:
            # Use popover if available, otherwise fall back to expander (works on older Streamlit)
            if hasattr(st, "popover"):
                with st.popover("🔑 Admin login"):  # no key to avoid older builds tripping on element id
                    pw = st.text_input("Password", type="password", key="admin_pw")
                    if st.button("Login", key="admin_login"):
                        if pw == ADMIN_PASSWORD:
                            st.session_state["is_admin"] = True
                            st.rerun()
                        else:
                            st.error("Invalid password")
            else:
                with st.expander("🔑 Admin login", expanded=False):
                    pw = st.text_input("Password", type="password", key="admin_pw")
                    if st.button("Login", key="admin_login"):
                        if pw == ADMIN_PASSWORD:
                            st.session_state["is_admin"] = True
                            st.rerun()
                        else:
                            st.error("Invalid password")


# ---------------------------------
# Matches Page
# ---------------------------------
def add_match_wizard():
    st.markdown("### ➕ Add / Update Match")
    with st.form("add_match"):
        c1,c2,c3 = st.columns(3)
        season = int(c1.number_input("Season", min_value=1, value=1, key="am_season"))
        gw = int(c2.number_input("Gameweek", min_value=1, value=1, key="am_gw"))
        date = c3.date_input("Date", value=dt.date.today(), key="am_date")
        side_count = st.selectbox("Side count", [5,7], index=0, key="am_side")
        team_a, team_b = "Non-bibs", "Bibs"
        sc1, sc2 = st.columns(2)
        score_a = int(sc1.number_input(f"{team_a} goals", min_value=0, value=0, key="am_sa"))
        score_b = int(sc2.number_input(f"{team_b} goals", min_value=0, value=0, key="am_sb"))
        motm_name = st.text_input("Man of the Match (optional)", key="am_motm")
        presets5 = ["1-2-1","1-3","2-2","3-1"]; presets7 = ["2-1-2-1","3-2-1","2-3-1"]
        fa, fb = st.columns(2)
        form_a = fa.selectbox("Formation (Non-bibs)", presets7 if side_count==7 else presets5, index=0, key="am_fa")
        form_b = fb.selectbox("Formation (Bibs)", presets7 if side_count==7 else presets5, index=0, key="am_fb")
        notes = st.text_area("Notes", key="am_notes")
        submit = st.form_submit_button("Save match", type="primary")
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
    presets = ["1-2-1","1-3","2-2","3-1","2-1-2-1","3-2-1","2-3-1"]
    for _, m in matches.sort_values(["season","gw"]).iterrows():
        with st.expander(f"S{int(m['season'])} GW{int(m['gw'])} — {m['team_a']} {int(m['score_a'] or 0)}–{int(m['score_b'] or 0)} {m['team_b']}", expanded=False):
            d1,d2,d3,d4 = st.columns([2,2,1,1])
            date = d1.date_input("Date", value=pd.to_datetime(m["date"]).date() if pd.notna(m["date"]) else dt.date.today(), key=f"dt_{m['id']}")
            sa = int(d2.number_input("Score A", min_value=0, value=int(m.get("score_a") or 0), key=f"sa_{m['id']}"))
            sb = int(d3.number_input("Score B", min_value=0, value=int(m.get("score_b") or 0), key=f"sb_{m['id']}"))
            motm = d4.text_input("MOTM", value=m.get("motm_name") or "", key=f"mm_{m['id']}")
            fa, fb = st.columns(2)
            idxa = presets.index(m.get("formation_a")) if m.get("formation_a") in presets else 0
            idxb = presets.index(m.get("formation_b")) if m.get("formation_b") in presets else 0
            forma = fa.selectbox("Formation A", presets, index=idxa, key=f"fix_fa_{m['id']}")
            formb = fb.selectbox("Formation B", presets, index=idxb, key=f"fix_fb_{m['id']}")
            if st.button("Save", key=f"fixsave_{m['id']}"):
                s.table("matches").update({
                    "date": str(date), "score_a": sa, "score_b": sb, "is_draw": (sa==sb),
                    "motm_name": motm or None, "formation_a": forma, "formation_b": formb
                }).eq("id", m["id"]).execute()
                clear_caches(); st.success("Updated."); st.rerun()

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
    sel_season = st.selectbox("Season", seasons, index=len(seasons)-1 if seasons else 0, key="m_season")
    opts = matches[matches["season"]==sel_season].sort_values("gw")
    labels = opts.apply(lambda r: f"GW {int(r['gw'])} — {r['team_a']} {int(r.get('score_a') or 0)}–{int(r.get('score_b') or 0)} {r['team_b']}", axis=1).tolist()
    id_map = {labels[i]: str(opts.iloc[i]["id"]) for i in range(len(opts))}
    sel_label = st.selectbox("Match", labels, key="m_label")
    mid = id_map[sel_label]

    m = matches[matches["id"].astype(str)==mid].iloc[0]
    show_photos = st.toggle("Show photos", True, key=f"sp_{mid}")

    st.markdown(
        f"<div class='pl-banner'>"
        f"<div><div class='pl-title'>Season {int(m['season'])} · GW {int(m['gw'])}</div>"
        f"<div class='pl-sub'>{m.get('date') or ''}</div></div>"
        f"<div class='pl-title'>{m['team_a']} {int(m.get('score_a') or 0)} – {int(m.get('score_b') or 0)} {m['team_b']}</div>"
        f"</div>", unsafe_allow_html=True
    )
    if m.get("motm_name"):
        st.markdown(f"<div class='pl-banner slim'><span>Man of the Match</span><span class='pl-badge'>🏅 {m['motm_name']}</span></div>", unsafe_allow_html=True)

    g = lfact[lfact["match_id"]==mid]
    a_rows = g[g["team"]==m["team_a"]].copy()
    b_rows = g[g["team"]==m["team_b"]].copy()

    # Formations (admin can change)
    if st.session_state.get("is_admin"):
        presets5 = ["1-2-1","1-3","2-2","3-1"]; presets7 = ["2-1-2-1","3-2-1","2-3-1"]
        preset_list = presets7 if int(m.get("side_count") or 5)==7 else presets5
        colf1, colf2, colf3 = st.columns([2,2,1])
        fa = colf1.selectbox("Formation (Non-bibs)", preset_list, index=(preset_list.index(m.get("formation_a")) if m.get("formation_a") in preset_list else 0), key=f"view_fa_{mid}")
        fb = colf2.selectbox("Formation (Bibs)", preset_list, index=(preset_list.index(m.get("formation_b")) if m.get("formation_b") in preset_list else 0), key=f"view_fb_{mid}")
        if colf3.button("Save formations", key=f"save_forms_{mid}"):
            s = service()
            if not s: st.error("Admin required.")
            else:
                s.table("matches").update({"formation_a": fa, "formation_b": fb}).eq("id", mid).execute()
                clear_caches(); st.success("Formations updated."); st.rerun()
    else:
        fa = m.get("formation_a") or ("2-1-2-1" if int(m.get("side_count") or 5)==7 else "1-2-1")
        fb = m.get("formation_b") or ("2-1-2-1" if int(m.get("side_count") or 5)==7 else "1-2-1")

    c1, c2 = st.columns(2)
    with c1:
        st.caption(m["team_a"])
        render_pitch(a_rows, fa, m.get("motm_name"), m["team_a"], show_stats=True, show_photos=show_photos)
    with c2:
        st.caption(m["team_b"])
        render_pitch(b_rows, fb, m.get("motm_name"), m["team_b"], show_stats=True, show_photos=show_photos)

    # Inline G/A + MOTM editor (admin)
    if st.session_state.get("is_admin"):
        with st.expander("📊 Goals / Assists & MOTM", expanded=False):
            s = service()
            if not s:
                st.info("Login as admin.")
            else:
                all_names = g["name"].dropna().astype(str).unique().tolist()
                default_idx = all_names.index(m.get("motm_name")) if m.get("motm_name") in all_names else (0 if all_names else None)
                colm1, colm2 = st.columns([3,1])
                with colm1:
                    motm_pick = st.selectbox("Man of the Match", all_names if all_names else [""], index=(default_idx if default_idx is not None else 0), key=f"motm_{mid}")
                with colm2:
                    if st.button("Save MOTM", key=f"motm_save_{mid}"):
                        s.table("matches").update({"motm_name": motm_pick or None}).eq("id", mid).execute()
                        clear_caches(); st.success("MOTM saved."); st.rerun()

                st.markdown("#### Non-bibs")
                for _, r in a_rows.sort_values("name").iterrows():
                    c1,c2,c3,c4 = st.columns([3,1,1,1])
                    c1.write(r["name"])
                    g_in = int(c2.number_input("⚽", min_value=0, value=int(r.get("goals") or 0), key=f"ga_{r['id']}"))
                    a_in = int(c3.number_input("🅰️", min_value=0, value=int(r.get("assists") or 0), key=f"as_{r['id']}"))
                    if c4.button("Save", key=f"save_{r['id']}"):
                        s.table("lineups").update({"goals": g_in, "assists": a_in}).eq("id", r["id"]).execute()
                        clear_caches(); st.success(f"Saved {r['name']}"); st.rerun()

                st.markdown("#### Bibs")
                for _, r in b_rows.sort_values("name").iterrows():
                    c1,c2,c3,c4 = st.columns([3,1,1,1])
                    c1.write(r["name"])
                    g_in = int(c2.number_input("⚽", min_value=0, value=int(r.get("goals") or 0), key=f"ga_{r['id']}"))
                    a_in = int(c3.number_input("🅰️", min_value=0, value=int(r.get("assists") or 0), key=f"as_{r['id']}"))
                    if c4.button("Save", key=f"save_{r['id']}"):
                        s.table("lineups").update({"goals": g_in, "assists": a_in}).eq("id", r["id"]).execute()
                        clear_caches(); st.success(f"Saved {r['name']}"); st.rerun()

        # Drag & Drop positions (per team)
        with st.expander("🧲 Drag & drop lineup (beta)", expanded=False):
            s = service()
            if not s:
                st.info("Login as admin.")
            else:
                updates = []
                colA, colB = st.columns(2)
                with colA:
                    st.markdown(f"**{m['team_a']}**")
                    upd_a = dnd_lineup_editor(a_rows, fa, m["team_a"], keypref=f"dndA", mid=mid)
                    if upd_a: updates.extend(upd_a)
                with colB:
                    st.markdown(f"**{m['team_b']}**")
                    upd_b = dnd_lineup_editor(b_rows, fb, m["team_b"], keypref=f"dndB", mid=mid)
                    if upd_b: updates.extend(upd_b)

                if updates and st.button("💾 Save all positions", type="primary", key=f"save_dnd_{mid}"):
                    try:
                        CHUNK = 20
                        for i in range(0, len(updates), CHUNK):
                            s.table("lineups").upsert(updates[i:i+CHUNK], on_conflict="id").execute()
                        clear_caches()
                        st.success("Positions saved."); st.rerun()
                    except Exception as e:
                        st.error(f"Save failed: {e}")

# ---------------------------------
# Player Manager (Add/Edit + Photo upload)
# ---------------------------------
def process_avatar_to_png_square(file) -> bytes:
    """Read uploaded image (JPG/PNG/HEIC) and return square PNG bytes (512x512)."""
    data = file.read()
    img = Image.open(io.BytesIO(data))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    w,h = img.size
    m = min(w,h)
    left = (w-m)//2; top = (h-m)//2
    img = img.crop((left, top, left+m, top+m)).resize((512,512))
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()

def upload_avatar_png(player_id: str, png_bytes: bytes) -> str:
    s = service()
    if not s: raise RuntimeError("Admin required to upload photos.")
    path = f"players/{player_id}.png"
    s.storage.from_(AVATAR_BUCKET).upload(path, png_bytes, {"content-type": "image/png", "x-upsert": "true"})
    url = s.storage.from_(AVATAR_BUCKET).get_public_url(path)
    return url

def page_player_manager():
    st.subheader("Player Manager")
    players = fetch_players()
    s = service()
    if not st.session_state.get("is_admin"):
        st.info("Login as admin to add/edit players.")
        st.dataframe(players[["name","photo_url","notes"]].rename(columns={"photo_url":"Photo"}), use_container_width=True, hide_index=True)
        return

    tab1, tab2 = st.tabs(["Edit Existing", "Add New"])
    with tab1:
        if players.empty:
            st.info("No players yet.")
        else:
            sel = st.selectbox("Select player", players["name"].tolist(), key="pm_sel")
            prow = players[players["name"]==sel].iloc[0]
            new_name = st.text_input("Name", value=prow["name"], key=f"pm_name_{prow['id']}")
            notes = st.text_area("Notes", value=prow.get("notes") or "", key=f"pm_notes_{prow['id']}")
            up = st.file_uploader("Upload photo (JPG/PNG/HEIC)", type=["jpg","jpeg","png","heic","heif"], key=f"pm_upload_{prow['id']}")
            colb1, colb2, colb3 = st.columns([1,1,1])
            if colb1.button("Save", type="primary", key=f"pm_save_{prow['id']}"):
                if not s: st.error("Admin required.")
                else:
                    update = {"name": new_name, "notes": (notes or None)}
                    s.table("players").update(update).eq("id", prow["id"]).execute()
                    clear_caches(); st.success("Player updated."); st.rerun()
            if up is not None and colb2.button("Save photo", key=f"pm_save_photo_{prow['id']}"):
                try:
                    png = process_avatar_to_png_square(up)
                    url = upload_avatar_png(prow["id"], png)
                    s.table("players").update({"photo_url": url}).eq("id", prow["id"]).execute()
                    clear_caches(); st.success("Photo updated."); st.rerun()
                except Exception as e:
                    st.error(f"Upload failed: {e}")
            if colb3.button("Delete player", type="secondary", key=f"pm_del_{prow['id']}"):
                s.table("players").delete().eq("id", prow["id"]).execute()
                clear_caches(); st.success("Deleted."); st.rerun()

    with tab2:
        nm = st.text_input("Name (unique)", key="pm_new_name")
        nt = st.text_area("Notes", key="pm_new_notes")
        up2 = st.file_uploader("Upload photo (optional, JPG/PNG/HEIC)", type=["jpg","jpeg","png","heic","heif"], key="pm_new_upload")
        if st.button("Add player", type="primary", key="pm_new_add"):
            if not s: st.error("Admin required.")
            elif not nm.strip():
                st.error("Name is required.")
            else:
                res = s.table("players").insert({"name": nm.strip(), "notes": (nt or None)}).execute()
                new = pd.DataFrame(res.data)
                if not new.empty:
                    pid = new.iloc[0]["id"]
                    if up2 is not None:
                        try:
                            png = process_avatar_to_png_square(up2)
                            url = upload_avatar_png(pid, png)
                            s.table("players").update({"photo_url": url}).eq("id", pid).execute()
                        except Exception as e:
                            st.error(f"Photo upload failed: {e}")
                clear_caches(); st.success("Player added."); st.rerun()

# ---------------------------------
# Players (profile)
# ---------------------------------
def page_players():
    players = fetch_players(); matches = fetch_matches(); lineups = fetch_lineups(); awards = fetch_awards()
    lfact, _ = build_fact(players, matches, lineups)

    st.subheader("Players")
    names = sorted(players["name"].dropna().astype(str).unique().tolist())
    sel = st.selectbox("Select player", [None]+names, index=0, key="pp_sel")
    if not sel: st.info("Choose a player"); return

    prow = players[players["name"]==sel]
    p = prow.iloc[0].to_dict() if not prow.empty else {"id":None,"name":sel,"photo_url":None,"notes":None}
    mine = lfact[lfact["name"]==sel].copy().sort_values(["season","gw"])

    c1, c2 = st.columns([1,2])
    with c1:
        if p.get("photo_url"):
            st.image(p["photo_url"], width=160)
        else:
            st.markdown(f"<div class='pl-avatar pl-large'><div class='pl-avatar-init'>{initials(sel)}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"## {sel}")
        if p.get("notes"): st.caption(p["notes"])

    agg_all = player_agg(lfact)
    me = agg_all[agg_all["name"]==sel]
    if not me.empty:
        me_row = me.iloc[0]
        cards = [
            ("Games", int(me_row["gp"])),
            ("W-D-L", f"{int(me_row['w'])}-{int(me_row['d'])}-{int(me_row['l'])}"),
            ("Win %", f"{me_row['win_pct']}%"),
            ("Goals", int(me_row["goals"])),
            ("Assists", int(me_row["assists"])),
            ("G+A", int(me_row["ga"])),
            ("G/PG", me_row["g_pg"]),
            ("A/PG", me_row["a_pg"]),
            ("G+A/PG", me_row["ga_pg"]),
            ("Team Contrib %", f"{me_row['team_contrib_pct']}%"),
        ]
        st.markdown("<div class='pl-cards'>"+ "".join([f"<div class='pl-card'><div class='pl-card-val'>{v}</div><div class='pl-card-lbl'>{k}</div></div>" for k,v in cards]) +"</div>", unsafe_allow_html=True)

    if not mine.empty:
        mine_ord = mine.sort_values(["season","gw"])
        res_streak = compute_streak(mine_ord["result"].tolist(), lambda r: r=="W")
        ga_streak  = compute_streak(mine_ord["ga"].tolist(), lambda x: int(x)>0)
        g_streak   = compute_streak(mine_ord["goals"].tolist(), lambda x: int(x)>0)
        a_streak   = compute_streak(mine_ord["assists"].tolist(), lambda x: int(x)>0)
        st.markdown("<div class='pl-cards'>"+ "".join([
            f"<div class='pl-card'><div class='pl-card-val'>{res_streak}</div><div class='pl-card-lbl'>Win streak</div></div>",
            f"<div class='pl-card'><div class='pl-card-val'>{ga_streak}</div><div class='pl-card-lbl'>G+A streak</div></div>",
            f"<div class='pl-card'><div class='pl-card-val'>{g_streak}</div><div class='pl-card-lbl'>Goals streak</div></div>",
            f"<div class='pl-card'><div class='pl-card-val'>{a_streak}</div><div class='pl-card-lbl'>Assists streak</div></div>",
        ]) +"</div>", unsafe_allow_html=True)

    st.markdown("### Match Log")
    last_n = int(st.number_input("Show last N games (0 = all)", min_value=0, value=5, step=1, key="pp_lastN"))
    mine2 = mine.sort_values(["season","gw"], ascending=[False,False])
    if last_n>0: mine2 = mine2.head(last_n)
    cols = ["season","gw","team_a","score_a","score_b","team_b","team","goals","assists","ga","result"]
    mine2 = mine2[cols].rename(columns={
        "season":"Season","gw":"GW","team_a":"Team A","score_a":"A",
        "score_b":"B","team_b":"Team B","team":"Side","goals":"Goals","assists":"Assists","ga":"G+A","result":"Result"
    })
    st.dataframe(mine2, use_container_width=True, hide_index=True)

    st.markdown("### Duos & Nemesis")
    mg = int(st.number_input("Min games together/against", min_value=1, value=3, step=1, key="pp_min_gp"))
    last_x = int(st.number_input("Last N GWs (0 = all)", min_value=0, value=0, step=1, key="pp_lastX"))

    filt_me = df_filter_by(lfact[lfact["name"]==sel], None, last_x)
    a = filt_me.merge(lfact, on=["match_id","team"], suffixes=("_me","_tm"))
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

    b = filt_me.merge(lfact, on=["match_id"], suffixes=("_me","_op"))
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

# ---------------------------------
# Stats
# ---------------------------------
def page_stats():
    players = fetch_players(); matches = fetch_matches(); lineups = fetch_lineups(); awards = fetch_awards()
    lfact, _ = build_fact(players, matches, lineups)
    st.subheader("Stats")

    c1,c2,c3,c4 = st.columns(4)
    seasons = sorted(matches["season"].dropna().unique().tolist()) if not matches.empty else []
    season = c1.selectbox("Season", [None]+seasons, index=0, key="st_season")
    min_games = int(c2.number_input("Min games", min_value=0, value=0, step=1, key="st_mingp"))
    last_x = int(c3.number_input("Last N GWs (0 = all)", min_value=0, value=0, step=1, key="st_lastx"))
    top_n = int(c4.number_input("Rows (Top N, 0=all)", min_value=0, value=10, step=1, key="st_topn"))

    metric = st.selectbox(
        "Metric",
        ["G+A","Goals","Assists","Goals per Game","Assists per Game","G+A per Game","Win %","Team Contribution %","MOTM","Best Duos","Nemesis Pairs"],
        index=0, key="st_metric"
    )

    agg = player_agg(lfact, season=season, min_games=min_games, last_gw=last_x)

    motm = awards[awards["type"]=="MOTM"] if "type" in awards.columns else pd.DataFrame(columns=["player_name","type","season","gw"])
    if season is not None and not motm.empty and "season" in motm.columns:
        motm = motm[motm["season"]==season]
    if not motm.empty:
        motm_cnt = motm.groupby("player_name").size().rename("MOTM")
        agg = agg.merge(motm_cnt, left_on="name", right_index=True, how="left")
        agg["MOTM"] = agg["MOTM"].fillna(0).astype(int)
    else:
        agg["MOTM"] = 0

    rename = {
        "name":"Player","gp":"GP","w":"W","d":"D","l":"L","win_pct":"Win %","goals":"Goals",
        "assists":"Assists","ga":"G+A","g_pg":"G/PG","a_pg":"A/PG","ga_pg":"G+A/PG",
        "team_contrib_pct":"Team Contrib %","MOTM":"MOTM"
    }
    def show(df: pd.DataFrame, cols: list, sort_cols: list, asc: list):
        if df.empty:
            st.info("No data for current filters."); return
        out = df.sort_values(sort_cols, ascending=asc)
        if top_n>0: out = out.head(top_n)
        st.dataframe(out[cols].rename(columns=rename), use_container_width=True, hide_index=True)

    if metric == "Goals":
        show(agg, ["name","gp","goals","g_pg","assists","ga","win_pct","team_contrib_pct","MOTM"], ["goals","ga","assists"], [False,False,False])
    elif metric == "Assists":
        show(agg, ["name","gp","assists","a_pg","goals","ga","win_pct","team_contrib_pct","MOTM"], ["assists","ga","goals"], [False,False,False])
    elif metric == "G+A":
        show(agg, ["name","gp","ga","ga_pg","goals","assists","win_pct","team_contrib_pct","MOTM"], ["ga","goals","assists"], [False,False,False])
    elif metric == "Goals per Game":
        show(agg, ["name","gp","g_pg","goals","assists","ga","win_pct","team_contrib_pct","MOTM"], ["g_pg","goals"], [False,False])
    elif metric == "Assists per Game":
        show(agg, ["name","gp","a_pg","assists","goals","ga","win_pct","team_contrib_pct","MOTM"], ["a_pg","assists"], [False,False])
    elif metric == "G+A per Game":
        show(agg, ["name","gp","ga_pg","ga","goals","assists","win_pct","team_contrib_pct","MOTM"], ["ga_pg","ga","goals"], [False,False,False])
    elif metric == "Win %":
        show(agg, ["name","gp","win_pct","w","d","l","goals","assists","ga","team_contrib_pct","MOTM"], ["win_pct","ga","goals"], [False,False,False])
    elif metric == "Team Contribution %":
        show(agg, ["name","gp","team_contrib_pct","ga","goals","assists","win_pct","MOTM"], ["team_contrib_pct","ga","goals"], [False,False,False])
    elif metric == "MOTM":
        df = agg.sort_values(["MOTM","ga","goals"], ascending=[False,False,False])
        if top_n>0: df = df.head(top_n)
        st.dataframe(df[["name","MOTM","gp","ga","goals","assists","win_pct"]].rename(columns=rename), use_container_width=True, hide_index=True)
    elif metric == "Best Duos":
        df = duos_global(lfact, min_gp=max(1, min_games), season=season, last_gw=last_x)
        if top_n>0: df = df.head(top_n)
        st.dataframe(df.rename(columns={"duo":"Duo","gp":"GP","w":"W","d":"D","l":"L","win_pct":"Win %"}), use_container_width=True, hide_index=True)
    elif metric == "Nemesis Pairs":
        df = nemesis_global(lfact, min_gp=max(1, min_games), season=season, last_gw=last_x)
        if top_n>0: df = df.head(top_n)
        st.dataframe(df.rename(columns={"pair":"Pair","gp":"GP","w_vs":"W","d_vs":"D","l_vs":"L","win_pct_vs":"Win % vs"}), use_container_width=True, hide_index=True)

# ---------------------------------
# Awards
# ---------------------------------
def page_awards():
    aw = fetch_awards()
    st.subheader("Awards")
    if st.session_state.get("is_admin"):
        with st.form("add_award"):
            atype = st.selectbox("Type", ["MOTM","POTM"], key="aw_type")
            season = st.number_input("Season", min_value=1, value=1, step=1, key="aw_season")
            gw = st.number_input("GW (for MOTM)", min_value=1, value=1, step=1, key="aw_gw")
            month = st.number_input("Month (1-12 for POTM)", min_value=1, max_value=12, value=1, step=1, key="aw_month")
            player_name = st.text_input("Player name", key="aw_player")
            notes = st.text_input("Notes", key="aw_notes")
            if st.form_submit_button("Add", type="primary"):
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
    potm = aw[aw["type"]=="POTM"] if "type" in aw.columns else pd.DataFrame()
    if potm.empty: st.caption("No POTM yet.")
    else:
        for _, r in potm.sort_values(["season","month"]).iterrows():
            st.write(f"🏆 S{int(r['season'])} · M{int(r['month'])}: {r.get('player_name','')}")

    st.markdown("#### Man of the Match (History)")
    motm = aw[aw["type"]=="MOTM"] if "type" in aw.columns else pd.DataFrame()
    if motm.empty: st.caption("No MOTM yet.")
    else:
        for _, r in motm.sort_values(["season","gw"]).iterrows():
            st.write(f"🎖️ S{int(r['season'])} GW{int(r['gw'])}: {r.get('player_name','')}")

# ---------------------------------
# Router
# ---------------------------------
def run_app():
    header()

    Page = getattr(st, "Page", None); nav = getattr(st, "navigation", None)
    pages = {
        "Matches": page_matches,
        "Players": page_players,
        "Player Manager": page_player_manager,
        "Stats": page_stats,
        "Awards": page_awards,
    }

    if Page and nav:
        sections = {"Main":[
            Page(page_matches, title="Matches", icon="📋"),
            Page(page_players, title="Players", icon="👤"),
            Page(page_player_manager, title="Player Manager", icon="🛠️"),
            Page(page_stats, title="Stats", icon="📊"),
            Page(page_awards, title="Awards", icon="🏆"),
        ]}
        n = nav(sections); n.run()
    else:
        sel = st.sidebar.radio("Go to", list(pages.keys()), index=0, key="router_radio")
        pages[sel]()

if __name__ == "__main__":
    run_app()
