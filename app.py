# Powerleague Stats — Black & Gold (Phase 0+1 complete)
import streamlit as st
from supabase import create_client
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import date

# ---------- App config ----------
st.set_page_config(page_title="Powerleague Stats", layout="wide", initial_sidebar_state="collapsed")

# ---------- Secrets ----------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
SUPABASE_SERVICE_KEY = st.secrets["SUPABASE_SERVICE_KEY"]
ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
AVATAR_BUCKET = st.secrets.get("AVATAR_BUCKET", "avatars")

# ---------- Supabase clients ----------
sb = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
sb_service = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ---------- CSS loader ----------
# ---------- CSS loader with inline fallback ----------
PITCH_MIN_CSS = """
<style>
/* minimal, self-contained styles for the pitch & pills */
.pitch{position:relative;width:100%;padding-top:150%;
  background: radial-gradient(1200px 800px at 50% -20%, #162417 0%, #0e1711 45%, #0b120d 100%);
  border-radius:20px;border:1px solid #1b3b2d;overflow:hidden}
.pitch-inner{position:absolute;inset:10px;border-radius:16px}
.pitch-line{position:absolute;left:6%;right:6%;border-top:1px solid rgba(255,255,255,.06)}
.p-slot{position:absolute;transform:translate(-50%,-50%);display:flex;flex-direction:column;align-items:center;gap:.25rem}
.p-bubble{width:56px;height:56px;border-radius:999px;display:flex;align-items:center;justify-content:center;
  background:#0c2017;border:2px solid #274a3a;box-shadow:0 2px 10px rgba(0,0,0,.25)}
.p-bubble.motm{border-color:#D4AF37;box-shadow:0 0 0 2px rgba(212,175,55,.25),0 6px 18px rgba(212,175,55,.2)}
.p-init{font-weight:800;letter-spacing:.3px;color:#dff7ec}
.p-name{font-size:.85rem;font-weight:800;color:#E6EBF1;text-shadow:0 1px 0 rgba(0,0,0,.6)}
.p-pill{display:inline-flex;align-items:center;gap:.35rem;padding:.18rem .5rem;
  background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.12);border-radius:999px;font-size:.78rem}
.p-ico{display:inline-flex;align-items:center;justify-content:center;width:14px;height:14px}
</style>
"""

def load_css():
    """Load full theme CSS if available; otherwise inject minimal pitch CSS so layout never breaks."""
    # Try full theme
    try:
        with open("css/style.css", "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
            st.session_state["__css_theme_loaded__"] = True
            return
    except Exception:
        pass
    # Fallback (once per session)
    if not st.session_state.get("__pitch_css_injected__", False):
        st.markdown(PITCH_MIN_CSS, unsafe_allow_html=True)
        st.session_state["__pitch_css_injected__"] = True

def ensure_pitch_css():
    """Always ensure minimal pitch CSS exists (in case theme CSS wasn’t loaded)."""
    if not (st.session_state.get("__css_theme_loaded__") or st.session_state.get("__pitch_css_injected__")):
        st.markdown(PITCH_MIN_CSS, unsafe_allow_html=True)
        st.session_state["__pitch_css_injected__"] = True

# call once at import
load_css()


# ---------- Helpers: icons / badges (SVG, no emojis) ----------
def icon_svg(name: str, size: int = 14, color: str = "#E6EBF1") -> str:
    paths = {
        "ball": "M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Zm0 2 3 2-.5 3-2.5 2-2.5-2L9 6l3-2Zm-7 8 2-3 3 .5 1 2.5-1.5 2L7 14l-2-2Zm14 0-2 2-2.5 0-1.5-2 1-2.5 3-.5 2 3Zm-7 6 2-1 .5-3-2.5-2-2.5 2 .5 3 2 1Z",
        "assist": "M4 4h4v4H4V4Zm6 0h10v2H10V4Zm0 5h10v2H10V9ZM4 12h4v4H4v-4Zm6 4h10v2H10v-2Z",
        "crown": "M3 6l3 4 4-3 4 3 3-4v12H3V6Z",
        "trophy": "M4 4h16v2h-2c0 2.8-2.2 5-5 5S8 8.8 8 6H6V4Zm0 4h2c.2 2 1.9 3.6 4 3.9V17H8v3h8v-3h-2v-5.1c2.1-.3 3.8-1.9 4-3.9h2V8c0 3-2.2 5.5-5 5.9V17h-6v-3.1C6.2 13.5 4 11 4 8Z",
        "calendar": "M7 2v2H5a2 2 0 0 0-2 2v11a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2h-2V2h-2v2H9V2H7Zm12 8H5v9h14v-9Z",
        "edit": "M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25Zm18.71-10.04a1 1 0 0 0 0-1.41L19.2 3.29a1 1 0 0 0-1.41 0l-1.83 1.83 3.75 3.75 1.99-1.9Z",
        "upload": "M12 3l4 4h-3v6h-2V7H8l4-4Zm-7 12h14v2H5v-2Z",
    }
    p = paths.get(name)
    if not p:
        return ""
    return f"<svg class='p-ico' width='{size}' height='{size}' viewBox='0 0 24 24' fill='{color}' xmlns='http://www.w3.org/2000/svg'><path d='{p}'/></svg>"

def ga_inline(g: int, a: int) -> str:
    parts = []
    if (g or 0) > 0:
        parts.append(f"{icon_svg('ball', 14)}<span>{int(g)}</span>")
    if (a or 0) > 0:
        parts.append(f"{icon_svg('assist', 14)}<span>{int(a)}</span>")
    if not parts:
        return ""
    return f"<span class='p-pill'>{'&nbsp;&nbsp;'.join(parts)}</span>"

def motm_badge(name: str) -> str:
    return f"<span class='pl-badge'>{icon_svg('crown',16,'#D4AF37')} <b>MOTM</b>: {name}</span>"

def initials(name: str) -> str:
    return "".join([t[0] for t in str(name).split()[:2]]).upper() or "?"

# ---------- Caching helpers ----------
@st.cache_data(ttl=60, show_spinner=False)
def fetch_players() -> pd.DataFrame:
    data = sb.table("players").select("*").execute().data
    df = pd.DataFrame(data or [])
    if not df.empty:
        df["id"] = df["id"].astype(str)
        if "name" in df.columns:
            df["name"] = df["name"].astype(str)
    return df

@st.cache_data(ttl=60, show_spinner=False)
def fetch_matches() -> pd.DataFrame:
    data = sb.table("matches").select("*").order("season").order("gw").execute().data
    df = pd.DataFrame(data or [])
    if not df.empty:
        df["id"] = df["id"].astype(str)
    return df

@st.cache_data(ttl=60, show_spinner=False)
def fetch_lineups() -> pd.DataFrame:
    data = sb.table("lineups").select("*").execute().data
    df = pd.DataFrame(data or [])
    if not df.empty:
        df["id"] = df["id"].astype(str)
        if "match_id" in df.columns:
            df["match_id"] = df["match_id"].astype(str)
    return df

def clear_caches():
    try:
        fetch_players.clear(); fetch_matches.clear(); fetch_lineups.clear()
        st.cache_data.clear()
    except Exception:
        pass

def service():
    return sb_service if st.session_state.get("is_admin") else None

# ---------- Formation & fact ----------
def formation_to_lines(formation: str) -> List[int]:
    try:
        parts = [int(x) for x in str(formation or "").split("-") if str(x).isdigit()]
        return parts if parts else [1,2,1]
    except Exception:
        return [1,2,1]
def _ensure_positions(df: pd.DataFrame, formation: str) -> pd.DataFrame:
    """
    Guarantee sane values for is_gk/line/slot for a given formation.
    - Non-GK with missing line/slot get centered defaults.
    - All values clamped to formation bounds.
    """
    rows = df.copy()
    parts = formation_to_lines(formation)
    n_lines = max(1, len(parts))
    max_slots = max(parts + [1])

    # Ensure required columns exist
    for c in ["is_gk", "line", "slot", "goals", "assists"]:
        if c not in rows.columns:
            rows[c] = None

    rows["is_gk"] = rows["is_gk"].fillna(False).astype(bool)
    rows["goals"] = pd.to_numeric(rows["goals"], errors="coerce").fillna(0).astype(int)
    rows["assists"] = pd.to_numeric(rows["assists"], errors="coerce").fillna(0).astype(int)

    # Normalize to nullable integers
    rows["line"] = pd.to_numeric(rows["line"], errors="coerce").astype("Int64")
    rows["slot"] = pd.to_numeric(rows["slot"], errors="coerce").astype("Int64")

    center_line = (n_lines - 1) // 2  # middle row

    for i in rows.index:
        if bool(rows.at[i, "is_gk"]):
            rows.at[i, "line"] = pd.NA
            rows.at[i, "slot"] = pd.NA
            continue

        # LINE: default to center, then clamp
        ln = rows.at[i, "line"]
        if pd.isna(ln):
            ln = center_line
        else:
            ln = int(ln)
        ln = max(0, min(ln, n_lines - 1))
        rows.at[i, "line"] = ln

        # SLOT: default to centered within that line, then clamp
        slots = int(parts[ln])
        offset = (max_slots - slots) // 2
        sl = rows.at[i, "slot"]
        if pd.isna(sl):
            sl = offset + (slots - 1) // 2
        else:
            sl = int(sl)
        sl = max(offset, min(sl, offset + slots - 1))
        rows.at[i, "slot"] = sl

    return rows



def build_fact(players: pd.DataFrame, matches: pd.DataFrame, lineups: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Create lineup fact with joined player names/photos, per-match rows."""
    p = players.copy()
    m = matches.copy()
    l = lineups.copy()
    if p.empty or l.empty:
        return pd.DataFrame(columns=[
            "id","player_id","name","player_name","team","goals","assists","is_gk","line","slot",
            "match_id","season","gw"
        ]), {}
    # Left join names/photos onto lineups
    p_small = p[["id","name","photo_url"]].rename(columns={"id":"player_id"})
    l = l.merge(p_small, on="player_id", how="left")
    l["name"] = l["name"].fillna(l.get("player_name"))
    # ensure basics
    for c in ["goals","assists"]:
        l[c] = pd.to_numeric(l[c], errors="coerce").fillna(0).astype(int)
    if not m.empty:
        m_small = m[["id","season","gw","team_a","team_b","formation_a","formation_b","motm_name","side_count"]].rename(columns={"id":"match_id"})
        l = l.merge(m_small, on="match_id", how="left")
    else:
        l["season"] = None; l["gw"] = None
    return l, {}

# ---------- UI: header ----------
def header():
    c1, c2 = st.columns([1,1])
    with c1:
        st.title("⚽ Powerleague Stats")
    # defaults
    if "is_admin" not in st.session_state:
        st.session_state["is_admin"] = False
    with c2:
        if st.session_state["is_admin"]:
            b1, b2, b3 = st.columns([1,1,1])
            b1.success("Admin", icon="🔐")
            if b2.button("Clear cache"):
                clear_caches()
                st.success("Cache cleared.")
            if b3.button("Logout"):
                st.session_state["is_admin"] = False
                st.rerun()
        else:
            with st.expander("🔑 Admin login", expanded=False):
                pw = st.text_input("Password", type="password", key="pw_admin")
                if st.button("Login"):
                    if pw == ADMIN_PASSWORD:
                        st.session_state["is_admin"] = True
                        st.rerun()
                    else:
                        st.error("Invalid password")

# ---------- Pitch (no overlap) ----------
def stat_pill(goals: int, assists: int) -> str:
    """Premium GA chip: gold-accented monogram badges that always read well."""
    chunks = []
    if (goals or 0) > 0:
        chunks.append(f"<span class='i i-g'>G</span><strong>{int(goals)}</strong>")
    if (assists or 0) > 0:
        chunks.append(f"<span class='i i-a'>A</span><strong>{int(assists)}</strong>")
    if not chunks:
        return ""
    return f"<span class='p-pill'>{'&nbsp;&nbsp;&nbsp;'.join(chunks)}</span>"


def slot_html(x_pct: float, y_pct: float, name: str, motm: bool=False, pill: str = "") -> str:
    """Player bubble + name + pill at fixed vertical stack."""
    bubble_cls = "p-bubble motm" if motm else "p-bubble"
    init = "".join([t[0] for t in name.split()[:2]]).upper() or "?"
    return (
        f"<div class='p-slot' style='left:{x_pct}%;top:{y_pct}%;'>"
        f"  <div class='{bubble_cls}'><span class='p-init'>{init}</span></div>"
        f"  <div class='p-name'>{name}</div>"
        f"  {pill}"
        f"</div>"
    )


def render_pitch(rows: pd.DataFrame, formation: str, motm_name: Optional[str], team_label: str,
                 show_stats: bool = True, show_photos: bool = True):
    """
    FotMob-style pitch: lighter, premium look + scoped CSS so it never breaks.
    Uses _ensure_positions(...) so line/slot are always valid.
    """
    # Scoped inline CSS (lighter pitch + gold accents)
    pitch_css = """
    <style>
    .pitch{
      position:relative;width:100%;padding-top:150%;
      background:
        radial-gradient(1200px 800px at 50% -20%, #12161b 0%, #10151a 45%, #0f1418 100%),
        repeating-linear-gradient(180deg, rgba(255,255,255,.03) 0 10px, rgba(255,255,255,0) 10px 28px);
      border-radius:20px;border:1px solid #233041;overflow:hidden
    }
    .pitch-inner{position:absolute;inset:10px;border-radius:16px}
    .pitch-line{position:absolute;left:6%;right:6%;border-top:1px solid rgba(255,255,255,.10)}
    .p-slot{position:absolute;transform:translate(-50%,-50%);display:flex;flex-direction:column;align-items:center;gap:.34rem; text-align:center}
    .p-bubble{
      width:66px;height:66px;border-radius:999px;display:flex;align-items:center;justify-content:center;
      background:linear-gradient(180deg,#0f1720,#0b1219);border:2px solid #2c3e52;box-shadow:0 4px 16px rgba(0,0,0,.35)
    }
    .p-bubble.motm{border-color:#D4AF37;box-shadow:0 0 0 2px rgba(212,175,55,.22),0 10px 24px rgba(212,175,55,.18)}
    .p-init{font-weight:800;letter-spacing:.3px;color:#eef6ff;font-size:1.02rem}
    .p-name{font-size:.95rem;font-weight:800;color:#E9EEF3;text-shadow:0 1px 0 rgba(0,0,0,.6); max-width:120px}
    .p-pill{
      display:inline-flex;align-items:center;gap:.55rem;padding:.26rem .6rem;
      background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.16);border-radius:999px;
      font-size:.95rem
    }
    .i{
      display:inline-flex;align-items:center;justify-content:center;
      width:18px;height:18px;border-radius:4px;font-size:.75rem;font-weight:900;
      border:1px solid rgba(255,255,255,.25)
    }
    .i-g{color:#D4AF37;background:rgba(212,175,55,.12);border-color:rgba(212,175,55,.45)}
    .i-a{color:#86c7ff;background:rgba(134,199,255,.12);border-color:rgba(134,199,255,.45)}
    </style>
    """

    # Sanitize positions
    rows = _ensure_positions(rows, formation)
    parts = formation_to_lines(formation)
    if not parts:
        parts = [1, 2, 1]

    # layout constants
    top_margin = 12
    bottom_margin = 10
    inner_h = 100 - top_margin - bottom_margin
    n_lines = max(1, len(parts))
    max_slots = max(parts + [1])

    # Build HTML
    html = [pitch_css, "<div class='pitch'><div class='pitch-inner'>"]

    # guide lines
    for i in range(n_lines + 1):
        y = top_margin + (inner_h / (n_lines + 1)) * (i + 0.5)
        html.append(f"<div class='pitch-line' style='top:{y}%'></div>")

    # GK at top center (if any)
    gk = rows[rows.get("is_gk") == True] if "is_gk" in rows.columns else pd.DataFrame()
    if not gk.empty:
        r = gk.iloc[0]
        nm = str(r.get("name") or r.get("player_name") or "")
        y = top_margin * 0.45
        x = 50
        pill = stat_pill(int(r.get("goals") or 0), int(r.get("assists") or 0)) if show_stats else ""
        html.append(slot_html(x, y, nm, motm=(motm_name == nm), pill=pill))

    # Outfield rows
    for line_idx, slots in enumerate(parts):
        y = top_margin + inner_h * ((line_idx + 0.5) / n_lines)
        x_gap = 100 / (slots + 1)
        line_df = rows[(rows.get("is_gk") != True) & (rows.get("line") == line_idx)]
        for j in range(slots):
            x = (j + 1) * x_gap
            abs_offset = (max_slots - slots) // 2
            abs_slot = abs_offset + j
            p = line_df[line_df.get("slot") == abs_slot].head(1)
            if len(p) == 0:
                continue
            r = p.iloc[0]
            nm = str(r.get("name") or r.get("player_name") or "")
            pill = stat_pill(int(r.get("goals") or 0), int(r.get("assists") or 0)) if show_stats else ""
            html.append(slot_html(x, y, nm, motm=(motm_name == nm), pill=pill))

    html.append("</div></div>")
    st.markdown("".join(html), unsafe_allow_html=True)


# ---------- Tap-to-place editor (stable, no rerun thrash) ----------
def tap_pitch_editor(team_rows: pd.DataFrame, formation: str, team_label: str, keypref: str, mid: str):
    rows = _ensure_positions(team_rows, formation).copy()
    parts = formation_to_lines(formation)
    max_slots = max(parts + [1])

    pos_key = f"posmap_{keypref}_{mid}"
    sel_key = f"sel_{keypref}_{mid}"
    if pos_key not in st.session_state:
        pos = {}
        for _, r in rows.iterrows():
            pid = str(r["id"])
            is_gk = bool(r.get("is_gk"))
            pos[pid] = {
                "name": str(r.get("name") or r.get("player_name") or ""),
                "is_gk": is_gk,
                "line": None if is_gk else (int(r.get("line")) if pd.notna(r.get("line")) else None),
                "slot": None if is_gk else (int(r.get("slot")) if pd.notna(r.get("slot")) else None),
            }
        st.session_state[pos_key] = pos
    pos: Dict[str, dict] = st.session_state[pos_key]
    selected_pid = st.session_state.get(sel_key)

    # Clamp to formation
    for v in pos.values():
        if v["is_gk"]:
            v["line"] = v["slot"] = None
        else:
            if v["line"] is None: continue
            v["line"] = min(max(int(v["line"]), 0), len(parts)-1)
            slots = parts[v["line"]]
            offset = (max_slots - slots)//2
            if v["slot"] is None: v["slot"] = offset + (slots-1)//2
            v["slot"] = min(max(int(v["slot"]), offset), offset+slots-1)

    def occupant_at(line_i:int, abs_slot:int)->Optional[str]:
        for pid, v in pos.items():
            if (not v["is_gk"]) and v["line"]==line_i and v["slot"]==abs_slot:
                return pid
        return None

    def gk_pid()->Optional[str]:
        for pid, v in pos.items():
            if v["is_gk"]: return pid
        return None

    # player pick
    with st.container(border=True):
        st.markdown(f"**Select player to place ({team_label})**")
        ids = sorted(list(pos.keys()), key=lambda k: pos[k]["name"].lower())
        cols = st.columns(4)
        for i, pid in enumerate(ids):
            name = pos[pid]["name"]
            is_sel = (selected_pid == pid)
            label = ("✅ " if is_sel else "• ") + name
            if cols[i%4].button(label, key=f"{keypref}_{mid}_pick_{pid}"):
                st.session_state[sel_key] = None if is_sel else pid

    # GK actions
    gk_current = gk_pid()
    c1, c2, c3 = st.columns([2,1,1])
    c1.info(f"Current GK: {pos[gk_current]['name'] if gk_current else '— none —'}")
    if c2.button("Set selected as GK", key=f"{keypref}_{mid}_setgk"):
        if selected_pid:
            if gk_current and gk_current in pos:
                pos[gk_current]["is_gk"] = False
            pos[selected_pid]["is_gk"] = True
            pos[selected_pid]["line"] = pos[selected_pid]["slot"] = None
        else:
            st.warning("Pick a player first.")
    if c3.button("Bench GK", key=f"{keypref}_{mid}_benchgk") and gk_current:
        pos[gk_current]["is_gk"] = False
        pos[gk_current]["line"] = pos[gk_current]["slot"] = None

    # Pitch slots
    for i, slots in enumerate(parts):
        st.write(f"Row {i+1}")
        cols = st.columns(slots)
        offset = (max_slots - slots)//2
        for j in range(slots):
            abs_slot = offset + j
            occ = occupant_at(i, abs_slot)
            label = f"{pos[occ]['name']}" if occ else "＋ Empty"
            if cols[j].button(label, key=f"{keypref}_{mid}_r{i}_s{j}"):
                if selected_pid:
                    if occ:
                        pos[occ]["is_gk"] = False
                        pos[occ]["line"] = pos[occ]["slot"] = None
                    pos[selected_pid]["is_gk"] = False
                    pos[selected_pid]["line"] = i
                    pos[selected_pid]["slot"] = abs_slot
                else:
                    if occ:
                        pos[occ]["is_gk"] = False
                        pos[occ]["line"] = pos[occ]["slot"] = None

    bench = [pid for pid,v in pos.items() if (not v["is_gk"]) and (v["line"] is None or v["slot"] is None)]
    if bench:
        st.caption("Bench: " + ", ".join(pos[x]["name"] for x in bench))

    cc1, cc2 = st.columns(2)
    if cc1.button("Auto-spread", key=f"{keypref}_{mid}_auto"):
        pool = bench[:]
        for i, slots in enumerate(parts):
            offset = (max_slots - slots)//2
            for j in range(slots):
                abs_slot = offset + j
                if occupant_at(i, abs_slot): continue
                if not pool: break
                pid = pool.pop(0)
                pos[pid]["is_gk"] = False
                pos[pid]["line"] = i
                pos[pid]["slot"] = abs_slot
    if cc2.button("Bench all", key=f"{keypref}_{mid}_benchall"):
        for pid, v in pos.items():
            if not v["is_gk"]:
                v["line"] = v["slot"] = None

    # Build updates
    updates = []
    for _, r in rows.iterrows():
        pid = str(r["id"])
        v = pos[pid]
        updates.append({
            "id": pid,
            "is_gk": bool(v["is_gk"]),
            "line": None if v["is_gk"] else (None if v["line"] is None else int(v["line"])),
            "slot": None if v["is_gk"] else (None if v["slot"] is None else int(v["slot"])),
        })
    return updates

# ---------- Admin tools (minimal, solid) ----------
def add_match_wizard():
    s = service()
    if not s:
        st.info("Login as admin.")
        return

    k = "amw"  # key prefix for this wizard

    col1, col2, col3 = st.columns(3)
    season = int(col1.number_input("Season", min_value=2020, max_value=2100,
                                   value=date.today().year, key=f"{k}_season"))
    gw = int(col2.number_input("Gameweek", min_value=1, value=1, key=f"{k}_gw"))
    side_count = int(col3.selectbox("Side count", [5, 7], index=0, key=f"{k}_sidecount"))
    d = st.date_input("Date", value=date.today(), key=f"{k}_date")

    col4, col5 = st.columns(2)
    team_a = col4.text_input("Team A", value="Non-bibs", key=f"{k}_teama")
    team_b = col5.text_input("Team B", value="Bibs", key=f"{k}_teamb")

    col6, col7 = st.columns(2)
    score_a = int(col6.number_input("Score A", min_value=0, value=0, key=f"{k}_scorea"))
    score_b = int(col7.number_input("Score B", min_value=0, value=0, key=f"{k}_scoreb"))

    presets5 = ["1-2-1", "1-3", "2-2", "3-1"]
    presets7 = ["2-1-2-1", "3-2-1", "2-3-1"]
    preset_list = presets7 if side_count == 7 else presets5

    col8, col9 = st.columns(2)
    fa = col8.selectbox("Formation A", preset_list, index=0, key=f"{k}_fa")
    fb = col9.selectbox("Formation B", preset_list, index=0, key=f"{k}_fb")

    motm = st.text_input("MOTM (optional)", value="", key=f"{k}_motm")

    if st.button("Save match", key=f"{k}_save"):
        try:
            payload = {
                "season": season, "gw": gw, "side_count": side_count,
                "team_a": team_a, "team_b": team_b,
                "score_a": score_a, "score_b": score_b,
                "date": str(d), "motm_name": (motm or None),
                "formation_a": fa, "formation_b": fb,
                "is_draw": (score_a == score_b)
            }
            # upsert by (season, gw)
            existing = sb.table("matches").select("id").eq("season", season).eq("gw", gw).limit(1).execute().data
            if existing:
                mid = existing[0]["id"]
                s.table("matches").update(payload).eq("id", mid).execute()
            else:
                s.table("matches").insert(payload).execute()

            clear_caches()
            st.success("Match saved.")
        except Exception as e:
            st.error(f"Save failed: {e}")
def fixtures_admin_table():
    s = service()
    if not s:
        st.info("Login as admin.")
        return

    k = "fx"  # key prefix for fixtures editor

    matches = fetch_matches()
    if matches.empty:
        st.info("No matches yet.")
        return

    seasons = sorted(matches["season"].dropna().unique().tolist())
    sel_season = st.selectbox("Season", seasons,
                              index=(len(seasons) - 1 if seasons else 0),
                              key=f"{k}_season")
    subset = matches[matches["season"] == sel_season].sort_values("gw")

    labels = subset.apply(
        lambda r: f"GW {int(r['gw'])} — {r['team_a']} {int(r.get('score_a') or 0)}–{int(r.get('score_b') or 0)} {r['team_b']}",
        axis=1
    )
    id_map = {labels.iloc[i]: subset.iloc[i]["id"] for i in range(len(subset))}
    sel = st.selectbox("Edit match", list(id_map.keys()), key=f"{k}_pick")
    mid = id_map[sel]
    m = subset[subset["id"] == mid].iloc[0]

    c1, c2, c3 = st.columns(3)
    score_a = int(c1.number_input("Score A", min_value=0,
                                  value=int(m.get("score_a") or 0),
                                  key=f"{k}_sa_{mid}"))
    score_b = int(c2.number_input("Score B", min_value=0,
                                  value=int(m.get("score_b") or 0),
                                  key=f"{k}_sb_{mid}"))
    motm = c3.text_input("MOTM", value=str(m.get("motm_name") or ""),
                         key=f"{k}_motm_{mid}")

    presets5 = ["1-2-1", "1-3", "2-2", "3-1"]
    presets7 = ["2-1-2-1", "3-2-1", "2-3-1"]
    is7 = int(m.get("side_count") or 5) == 7
    options = presets7 if is7 else presets5

    # Choose index based on current value if present
    fa_current = str(m.get("formation_a") or (options[0] if options else ""))
    fb_current = str(m.get("formation_b") or (options[0] if options else ""))
    fa_idx = options.index(fa_current) if fa_current in options else 0
    fb_idx = options.index(fb_current) if fb_current in options else 0

    fa = st.selectbox("Formation A", options, index=fa_idx, key=f"{k}_fa_{mid}")
    fb = st.selectbox("Formation B", options, index=fb_idx, key=f"{k}_fb_{mid}")

    d_val = pd.to_datetime(m.get("date") or date.today()).date()
    d = st.date_input("Date", value=d_val, key=f"{k}_date_{mid}")

    if st.button("Update match", key=f"{k}_update_{mid}"):
        try:
            s.table("matches").update({
                "score_a": score_a, "score_b": score_b,
                "is_draw": (score_a == score_b),
                "motm_name": (motm or None),
                "formation_a": fa, "formation_b": fb,
                "date": str(d)
            }).eq("id", mid).execute()
            clear_caches()
            st.success("Updated.")
        except Exception as e:
            st.error(f"Update failed: {e}")


# ---------- Page: Matches ----------
def page_matches():
    players = fetch_players()
    matches = fetch_matches()
    lineups = fetch_lineups()
    lfact, _ = build_fact(players, matches, lineups)

    st.subheader("Matches")

    # Admin tools
    if st.session_state.get("is_admin"):
        with st.expander("Admin: Add / Update Match", expanded=False):
            add_match_wizard()
        with st.expander("Admin: All Fixtures (edit dates/scores/formations/MOTM)", expanded=False):
            fixtures_admin_table()

    if matches.empty:
        st.info("No matches yet.")
        return

    # picker
    seasons = sorted(matches["season"].dropna().unique().tolist())
    sel_season = st.selectbox("Season", seasons, index=len(seasons)-1 if seasons else 0, key="m_season")
    opts = matches[matches["season"]==sel_season].sort_values("gw")
    labels = opts.apply(
        lambda r: f"GW {int(r['gw'])} — {r['team_a']} {int(r.get('score_a') or 0)}–{int(r.get('score_b') or 0)} {r['team_b']}",
        axis=1
    ).tolist()
    id_map = {labels[i]: str(opts.iloc[i]["id"]) for i in range(len(opts))}
    sel_label = st.selectbox("Match", labels, key="m_label")
    mid = id_map[sel_label]
    m = matches[matches["id"].astype(str)==mid].iloc[0]

    show_photos = st.toggle("Show photos", True, key=f"sp_{mid}")

    # banner
    st.markdown(
        f"<div class='pl-banner'>"
        f"<div><div class='pl-title'>Season {int(m['season'])} · GW {int(m['gw'])}</div>"
        f"<div class='pl-sub'>{m.get('date') or ''}</div></div>"
        f"<div class='pl-title'>{m['team_a']} {int(m.get('score_a') or 0)} – {int(m.get('score_b') or 0)} {m['team_b']}</div>"
        f"</div>",
        unsafe_allow_html=True
    )
    if m.get("motm_name"):
        st.markdown(f"<div class='pl-banner slim'>{motm_badge(m['motm_name'])}</div>", unsafe_allow_html=True)

    g = lfact[lfact["match_id"]==mid]
    a_rows = g[g["team"]==m["team_a"]].copy()
    b_rows = g[g["team"]==m["team_b"]].copy()

    # formations
    if st.session_state.get("is_admin"):
        presets5 = ["1-2-1", "1-3", "2-2", "3-1"]
        presets7 = ["2-1-2-1", "3-2-1", "2-3-1"]
        preset_list = presets7 if int(m.get("side_count") or 5)==7 else presets5
        colf1, colf2, colf3 = st.columns([2,2,1])
        fa = colf1.selectbox("Formation (Non-bibs)", preset_list,
                             index=(preset_list.index(m.get("formation_a")) if m.get("formation_a") in preset_list else 0),
                             key=f"view_fa_{mid}")
        fb = colf2.selectbox("Formation (Bibs)", preset_list,
                             index=(preset_list.index(m.get("formation_b")) if m.get("formation_b") in preset_list else 0),
                             key=f"view_fb_{mid}")
        if colf3.button("Save formations", key=f"save_forms_{mid}"):
            s = service()
            if not s:
                st.error("Admin required.")
            else:
                s.table("matches").update({"formation_a": fa, "formation_b": fb}).eq("id", mid).execute()
                clear_caches()
                st.success("Formations updated.")
                st.rerun()
    else:
        fa = m.get("formation_a") or ("2-1-2-1" if int(m.get("side_count") or 5)==7 else "1-2-1")
        fb = m.get("formation_b") or ("2-1-2-1" if int(m.get("side_count") or 5)==7 else "1-2-1")

    # read-only pitches
    c1, c2 = st.columns(2)
    with c1:
        st.caption(m["team_a"])
        render_pitch(a_rows, fa, m.get("motm_name"), m["team_a"], show_stats=True, show_photos=show_photos)
    with c2:
        st.caption(m["team_b"])
        render_pitch(b_rows, fb, m.get("motm_name"), m["team_b"], show_stats=True, show_photos=show_photos)

    # GA & MOTM editor
    if st.session_state.get("is_admin"):
        with st.expander("Goals / Assists & MOTM", expanded=False):
            s = service()
            if not s:
                st.info("Login as admin.")
            else:
                all_names = g["name"].dropna().astype(str).unique().tolist()
                default_idx = all_names.index(m.get("motm_name")) if m.get("motm_name") in all_names else (0 if all_names else None)
                colm1, colm2 = st.columns([3,1])
                with colm1:
                    motm_pick = st.selectbox(
                        "Man of the Match", all_names if all_names else [""],
                        index=(default_idx if default_idx is not None else 0),
                        key=f"motm_{mid}"
                    )
                with colm2:
                    if st.button("Save MOTM", key=f"motm_save_{mid}"):
                        s.table("matches").update({"motm_name": motm_pick or None}).eq("id", mid).execute()
                        clear_caches()
                        st.success("MOTM saved."); st.rerun()

                st.markdown("#### Non-bibs")
                for _, r in a_rows.sort_values("name").iterrows():
                    c1_, c2_, c3_, c4_ = st.columns([3,1,1,1])
                    c1_.write(r["name"])
                    c2_.markdown(icon_svg("ball", 14), unsafe_allow_html=True)
                    g_in = int(c2_.number_input(" ", min_value=0, value=int(r.get("goals") or 0),
                                                key=f"ga_{r['id']}", label_visibility="hidden"))
                    c3_.markdown(icon_svg("assist", 14), unsafe_allow_html=True)
                    a_in = int(c3_.number_input(" ", min_value=0, value=int(r.get("assists") or 0),
                                                key=f"as_{r['id']}", label_visibility="hidden"))
                    if c4_.button("Save", key=f"save_{r['id']}"):
                        s.table("lineups").update({"goals": g_in, "assists": a_in}).eq("id", r["id"]).execute()
                        clear_caches()
                        st.success(f"Saved {r['name']}"); st.rerun()

                st.markdown("#### Bibs")
                for _, r in b_rows.sort_values("name").iterrows():
                    c1_, c2_, c3_, c4_ = st.columns([3,1,1,1])
                    c1_.write(r["name"])
                    c2_.markdown(icon_svg("ball", 14), unsafe_allow_html=True)
                    g_in = int(c2_.number_input(" ", min_value=0, value=int(r.get("goals") or 0),
                                                key=f"ga_{r['id']}_b", label_visibility="hidden"))
                    c3_.markdown(icon_svg("assist", 14), unsafe_allow_html=True)
                    a_in = int(c3_.number_input(" ", min_value=0, value=int(r.get("assists") or 0),
                                                key=f"as_{r['id']}_b", label_visibility="hidden"))
                    if c4_.button("Save", key=f"save_{r['id']}_b"):
                        s.table("lineups").update({"goals": g_in, "assists": a_in}).eq("id", r["id"]).execute()
                        clear_caches()
                        st.success(f"Saved {r['name']}"); st.rerun()

    # Arrange lineup (tap-to-place)
    if st.session_state.get("is_admin"):
        with st.expander("Arrange lineup", expanded=False):
            s = service()
            if not s:
                st.info("Login as admin.")
            else:
                updates = []
                colA, colB = st.columns(2)
                with colA:
                    st.markdown(f"### {m['team_a']}")
                    upd_a = tap_pitch_editor(a_rows, fa, m["team_a"], keypref="A", mid=mid)
                    if upd_a: updates.extend(upd_a)
                with colB:
                    st.markdown(f"### {m['team_b']}")
                    upd_b = tap_pitch_editor(b_rows, fb, m["team_b"], keypref="B", mid=mid)
                    if upd_b: updates.extend(upd_b)

                if updates and st.button("💾 Save all positions", type="primary", key=f"save_tap_{mid}"):
                    try:
                        CHUNK = 20
                        for i in range(0, len(updates), CHUNK):
                            s.table("lineups").upsert(updates[i:i+CHUNK], on_conflict="id").execute()
                        clear_caches()
                        st.success("Positions saved."); st.rerun()
                    except Exception as e:
                        st.error(f"Save failed: {e}")

# ---------- Run ----------
def run_app():
    header()
    # Single page for now (stable)
    page_matches()

if __name__ == "__main__":
    run_app()
