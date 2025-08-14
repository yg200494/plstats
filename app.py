# app.py — Powerleague Stats (final)
# Streamlit + Supabase | Mobile-first | Black & Gold UI | 5s/7s | Tap & Slot lineup editors

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime
from typing import Optional, List, Tuple, Dict
from supabase import create_client
import uuid
import io

# HEIC -> PNG conversion
try:
    import pillow_heif
    HEIF_OK = True
except Exception:
    HEIF_OK = False

from PIL import Image, ImageDraw

# Optional tap-to-place editor
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_OK = True
except Exception:
    CANVAS_OK = False

# ------------------------------
# Streamlit config
# ------------------------------
st.set_page_config(page_title="Powerleague Stats", layout="wide", initial_sidebar_state="collapsed")

# ------------------------------
# Secrets / Supabase
# ------------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
SUPABASE_SERVICE_KEY = st.secrets.get("SUPABASE_SERVICE_KEY", SUPABASE_ANON_KEY)
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "")
AVATAR_BUCKET = st.secrets.get("AVATAR_BUCKET", "avatars")

sb = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
sb_service = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

def service():
    return sb_service if st.session_state.get("is_admin") else None

# ------------------------------
# Global styles (black & gold)
# ------------------------------
st.markdown("""
<style>
:root{--gold:#D4AF37;--ink:#0b0f14;--panel:#0f141a;--text:#e9eef3}
html,body,.stApp{background:#0b0f14;}
.block-container{padding-top:.6rem !important;}
h1,h2,h3,h4,h5{color:var(--text)}
.small{opacity:.85;font-size:.9rem}
.card{padding:14px;border-radius:16px;background:linear-gradient(180deg,#111722,#0d131c);border:1px solid rgba(255,255,255,.12)}
.card-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
.metric{display:flex;flex-direction:column;gap:6px;align-items:flex-start;padding:12px;border-radius:14px;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.14)}
.metric .k{opacity:.85}
.metric .v{font-weight:900;font-size:1.15rem;color:#fff}
.badge{display:flex;align-items:center;gap:10px;padding:10px 12px;border-radius:14px;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.14)}
hr{border-color:rgba(255,255,255,.15)}
thead tr th{background:rgba(255,255,255,.06)!important}
.pillR{display:inline-flex;align-items:center;gap:.4rem;padding:.2rem .5rem;border-radius:999px;border:1px solid rgba(255,255,255,.2)}
.ovr{font-weight:900;color:#D4AF37}

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
.center-dot{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);width:6px;height:6px;background:#ffffff;border-radius:999px}
.box-left{position:absolute;left:0;top:20%;bottom:20%;width:16.5%;border:2px solid #ffffff;border-left:none}
.six-left{position:absolute;left:0;top:39%;bottom:39%;width:7.6%;border:2px solid #ffffff;border-left:none}
.pen-dot-left{position:absolute;left:11%;top:50%;transform:translate(-50%,-50%);width:6px;height:6px;background:#ffffff;border-radius:999px}
.goal-left{position:absolute;left:-1.0%;top:47%;bottom:47%;width:1%;border:2px solid #ffffff;border-right:none}
.box-right{position:absolute;right:0;top:20%;bottom:20%;width:16.5%;border:2px solid #ffffff;border-right:none}
.six-right{position:absolute;right:0;top:39%;bottom:39%;width:7.6%;border:2px solid #ffffff;border-right:none}
.pen-dot-right{position:absolute;right:11%;top:50%;transform:translate(50%,-50%);width:6px;height:6px;background:#ffffff;border-radius:999px}
.goal-right{position:absolute;right:-1.0%;top:47%;bottom:47%;width:1%;border:2px solid #ffffff;border-left:none}

.slot{position:absolute;transform:translate(-50%,-50%);display:flex;flex-direction:column;align-items:center;gap:.28rem;text-align:center}
.bubble{
  width:clamp(60px,7.0vw,80px);height:clamp(60px,7.0vw,80px);
  border-radius:999px;display:flex;align-items:center;justify-content:center;position:relative;
  background:linear-gradient(180deg,#0e1620,#0b131b);border:2px solid #2f4860;box-shadow:0 5px 14px rgba(0,0,0,.34)}
.bubble.motm{border-color:#D4AF37;box-shadow:0 0 0 2px rgba(212,175,55,.22),0 10px 22px rgba(212,175,55,.16)}
.bubble.gk{background:linear-gradient(180deg,#0c1e2b,#0a1924);border-color:#4db6ff}
.chip-gk{position:absolute;right:-6px;top:-6px;padding:.14rem .34rem;font-size:.66rem;font-weight:900;border-radius:8px;
  background:rgba(77,182,255,.18);color:#bfe6ff;border:1px solid rgba(77,182,255,.45)}
.init{font-weight:900;letter-spacing:.3px;color:#e8f4ff;font-size:clamp(1.0rem,1.15vw,1.12rem)}
.name{font-size:clamp(.9rem,1.05vw,1.02rem);font-weight:800;color:#F1F6FA;text-shadow:0 1px 0 rgba(0,0,0,.45);max-width:140px}
.pill{display:inline-flex;align-items:center;gap:.45rem;padding:.18rem .5rem;border-radius:999px;background:rgba(0,0,0,.25);border:1px solid rgba(255,255,255,.18);font-size:.95rem}
.tag{display:inline-flex;align-items:center;justify-content:center;width:20px;height:20px;border-radius:5px;font-size:.74rem;font-weight:900;border:1px solid rgba(255,255,255,.3)}
.tag-g{color:#D4AF37;background:rgba(212,175,55,.15);border-color:rgba(212,175,55,.55)}
.tag-a{color:#86c7ff;background:rgba(134,199,255,.15);border-color:rgba(134,199,255,.55)}

@media (max-width:450px){
  .pitchX{padding-top:62%}
  .bubble{width:64px;height:64px}
}
</style>
""", unsafe_allow_html=True)

# Compact mode for iPhone
if st.session_state.get("compact"):
    st.markdown("""
    <style>
    .pitchX{padding-top:56%}
    .bubble{width:58px;height:58px;box-shadow:none;border-width:1px}
    .name{font-size:.9rem}
    .pill{font-size:.85rem}
    </style>
    """, unsafe_allow_html=True)

# ------------------------------
# Cache utilities
# ------------------------------
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
        cols = ["id","season","gw","side_count","team_a","team_b","score_a","score_b","date","motm_name","is_draw","formation_a","formation_b","notes"]
        return pd.DataFrame(data) if data else pd.DataFrame(columns=cols)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_lineups() -> pd.DataFrame:
    try:
        data = sb.table("lineups").select("*").execute().data
        cols = ["id","season","gw","match_id","team","player_id","player_name","name","is_gk","goals","assists","line","slot","position"]
        return pd.DataFrame(data) if data else pd.DataFrame(columns=cols)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=120)
def fetch_awards() -> pd.DataFrame:
    try:
        data = sb.table("awards").select("*").execute().data
        cols = ["id","season","month","type","gw","player_id","player_name","notes"]
        return pd.DataFrame(data) if data else pd.DataFrame(columns=cols)
    except Exception:
        return pd.DataFrame()

# ------------------------------
# Helpers
# ------------------------------
def initials(name: str) -> str:
    parts = [p for p in (name or "").split() if p]
    return "".join([p[0] for p in parts[:2]]).upper() or "?"

def formation_to_lines(formation: Optional[str]) -> List[int]:
    try:
        return [int(x) for x in str(formation or "").strip().split("-") if str(x).strip().isdigit()]
    except Exception:
        return []

def validate_formation(formation: Optional[str], side_count: int) -> str:
    """5s: 4 outfielders; 7s: 6 outfielders."""
    try:
        parts = [int(x) for x in str(formation or "").split("-") if str(x).strip().isdigit()]
    except Exception:
        parts = []
    outfield_needed = 4 if int(side_count or 5) == 5 else 6
    if sum(parts) != outfield_needed or not parts or any(p <= 0 for p in parts):
        return "1-2-1" if outfield_needed == 4 else "2-1-2-1"
    return "-".join(str(p) for p in parts)

def normalize_lineup_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "name" in out.columns:
        out["name"] = out["name"].fillna(out.get("player_name")).fillna("")
    else:
        out["name"] = out.get("player_name", "")
    out["name"] = out["name"].fillna("").astype(str)
    return out

def _ensure_positions(df: pd.DataFrame, formation: str) -> pd.DataFrame:
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

# ------------------------------
# Pitch rendering (combined)
# ------------------------------
def stat_pill(goals: int, assists: int) -> str:
    parts = []
    if (goals or 0) > 0: parts.append(f"<span class='tag tag-g'>G</span><b>{int(goals)}</b>")
    if (assists or 0) > 0: parts.append(f"<span class='tag tag-a'>A</span><b>{int(assists)}</b>")
    if not parts: return ""
    return f"<span class='pill'>{'&nbsp;&nbsp;'.join(parts)}</span>"

def slot_html(x_pct: float, y_pct: float, name: str, *, motm: bool=False, pill: str="", is_gk: bool=False) -> str:
    cls = "bubble"
    if motm:
        cls += " motm"
    if is_gk:
        cls += " gk"

    # Build the GK badge outside the f-string to avoid backslashes in an expression
    gk_badge = "<span class='chip-gk'>GK</span>" if is_gk else ""

    return (
        f"<div class='slot' style='left:{x_pct}%;top:{y_pct}%;'>"
        f"  <div class='{cls}'>"
        f"    <span class='init'>{initials(name)}</span>{gk_badge}"
        f"  </div>"
        f"  <div class='name'>{name}</div>"
        f"  {pill}"
        f"</div>"
    )


def render_match_pitch_combined(a_rows: pd.DataFrame, b_rows: pd.DataFrame,
                                formation_a: str, formation_b: str,
                                motm_name: Optional[str],
                                team_a: str, team_b: str):
    def lerp(a: float, b: float, t: float) -> float: return a + (b - a) * t

    a_rows = _ensure_positions(normalize_lineup_names(a_rows), formation_a)
    b_rows = _ensure_positions(normalize_lineup_names(b_rows), formation_b)
    parts_a = formation_to_lines(formation_a) or [1,2,1]
    parts_b = formation_to_lines(formation_b) or [1,2,1]

    y_top_margin, y_bot_margin = 6, 6
    inner_h = 100 - y_top_margin - y_bot_margin

    left_min, left_max  = 6, 48
    right_min, right_max = 52, 94

    def _place_side(rows: pd.DataFrame, parts: List[int], *, left_half: bool):
        out = []
        n_lines = max(1, len(parts))

        # GK near goal
        gk = rows[rows.get("is_gk") == True]
        if not gk.empty:
            r = gk.iloc[0]; nm = str(r.get("name") or r.get("player_name") or "")
            x = (left_min - 2.0) if left_half else (right_max + 2.0)
            y = 50
            pill = stat_pill(int(r.get("goals") or 0), int(r.get("assists") or 0))
            out.append(slot_html(x, y, nm, motm=(motm_name==nm), pill=pill, is_gk=True))

        # Outfield lines from goal -> center
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
                pill = stat_pill(int(rr.get("goals") or 0), int(rr.get("assists") or 0))
                out.append(slot_html(x, y, nm, motm=(motm_name==nm), pill=pill, is_gk=False))
        return out

    html = ["<div class='pitchX'><div class='inner'>",
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

# --- Tap editor helpers ---
def slot_centers_pct(parts: List[int], *, left_half: bool) -> List[Tuple[int,int,float,float]]:
    """Return list of (line, slot, x%, y%) centers for the given formation lines."""
    y_top_margin, y_bot_margin = 6, 6
    inner_h = 100 - y_top_margin - y_bot_margin
    left_min, left_max  = 6, 48
    right_min, right_max = 52, 94
    def lerp(a: float, b: float, t: float) -> float: return a + (b - a) * t
    coords = []
    n_lines = max(1, len(parts))
    for line_idx in range(n_lines):
        t = (line_idx + 1) / (n_lines + 1)
        x = lerp(left_min, left_max, t) if left_half else lerp(right_max, right_min, t)
        count = parts[line_idx]
        for j in range(count):
            y_t = (j + 1) / (count + 1)
            y = y_top_margin + y_t * inner_h
            coords.append((line_idx, j, x, y))
    return coords

def pitch_bg_image(width: int, height: int) -> Image.Image:
    """Small visual pitch for the tap canvas background."""
    img = Image.new("RGB", (width, height), (26, 60, 40))
    d = ImageDraw.Draw(img)
    W,H = width, height
    margin = int(0.05*H)
    # outer
    d.rectangle([int(0.035*W), margin, int(0.965*W), H-margin], outline=(255,255,255), width=2)
    # halfway
    d.line([(W//2, margin), (W//2, H-margin)], fill=(255,255,255), width=2)
    # center circle
    r = int(0.065*W); cx,cy = W//2, H//2
    d.ellipse([cx-r, cy-r, cx+r, cy+r], outline=(255,255,255), width=2)
    # boxes
    def boxes(side):
        if side=="L":
            x1 = int(0.035*W); x2 = int(x1 + 0.165*W)
            x3 = int(x1);      x4 = int(x1 + 0.076*W)
            pdx = int(0.11*W)
        else:
            x2 = int(0.965*W); x1 = int(x2 - 0.165*W)
            x4 = int(0.965*W); x3 = int(x4 - 0.076*W)
            pdx = int(W - 0.11*W)
        y1 = int(0.20*H); y2 = int(0.80*H)
        y3 = int(0.39*H); y4 = int(0.61*H)
        d.rectangle([x1, y1, x2, y2], outline=(255,255,255), width=2)
        d.rectangle([x3, y3, x4, y4], outline=(255,255,255), width=2)
        d.ellipse([pdx-3, H//2-3, pdx+3, H//2+3], fill=(255,255,255))
    boxes("L"); boxes("R")
    return img

# ------------------------------
# Fact table for stats (robust)
# ------------------------------
@st.cache_data(ttl=90)
def build_fact(players: pd.DataFrame, matches: pd.DataFrame, lineups: pd.DataFrame) -> pd.DataFrame:
    if lineups.empty or matches.empty:
        return pd.DataFrame(columns=["match_id","season","gw","date","team","name","is_gk","goals","assists","for","against","result","contrib"])
    l = normalize_lineup_names(lineups.copy())
    m = matches.set_index("id")
    for col in ["season","gw","date","score_a","score_b","team_a","team_b"]:
        if col in m.columns:
            l[col] = l["match_id"].map(m[col])

    def fa(row):
        if row["team"] == "Non-bibs":
            return int(row.get("score_a") or 0), int(row.get("score_b") or 0)
        return int(row.get("score_b") or 0), int(row.get("score_a") or 0)

    fa_cols = l.apply(lambda r: pd.Series(fa(r), index=["for","against"]), axis=1)
    l[["for","against"]] = fa_cols
    l["result"] = np.where(l["for"] > l["against"], "W", np.where(l["for"] == l["against"], "D", "L"))
    l["goals"] = pd.to_numeric(l["goals"], errors="coerce").fillna(0).astype(int)
    l["assists"] = pd.to_numeric(l["assists"], errors="coerce").fillna(0).astype(int)

    team_goals = (l.groupby(["match_id","team"])["goals"].sum().rename("team_goals")).reset_index()
    l = l.merge(team_goals, on=["match_id","team"], how="left")
    l["contrib"] = ((l["goals"] + l["assists"]) / l["team_goals"].replace(0, np.nan) * 100).round(1).fillna(0)

    return l[["match_id","season","gw","date","team","name","is_gk","goals","assists","for","against","result","contrib"]]

# ------------------------------
# Storage: avatar upload
# ------------------------------
def upload_avatar(file) -> Optional[str]:
    if file is None: return None
    suffix = file.name.split(".")[-1].lower()
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

# ------------------------------
# Header / Sidebar Admin
# ------------------------------
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

def sidebar_admin():
    st.sidebar.markdown("### Admin")
    if not st.session_state.get("is_admin"):
        pwd = st.sidebar.text_input("Password", type="password", key="sb_pwd")
        if st.sidebar.button("Login", key="sb_login"):
            if ADMIN_PASSWORD and pwd == ADMIN_PASSWORD:
                st.session_state["is_admin"] = True; st.rerun()
            else:
                st.sidebar.error("Wrong password")
    else:
        st.sidebar.success("Admin: ON")
        if st.sidebar.button("Clear cache", key="sb_clear"):
            clear_caches(); st.rerun()
        if st.sidebar.button("Log out", key="sb_logout"):
            st.session_state["is_admin"] = False; st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Display")
    compact = st.sidebar.toggle("Compact mode (iPhone)", value=st.session_state.get("compact", True), key="ui_compact")
    initials_only = st.sidebar.toggle("Initials on pitch (faster)", value=st.session_state.get("initials_only", True), key="ui_initials_only")
    st.session_state["compact"] = compact
    st.session_state["initials_only"] = initials_only

# ------------------------------
# Slot-based lineup editor (fallback)
# ------------------------------
def lineup_slots_editor(team_name: str, mid: str, side_count: int, formation: str,
                        lineup_df: pd.DataFrame, all_players: pd.DataFrame, keypref: str):
    """
    Editor pattern:
      - Select GK from dropdown
      - For each line/slot in formation: select player (+ goals/assists)
      - Save button: delete-then-insert per team
    """
    st.markdown(f"#### {team_name}")
    formation = validate_formation(formation, side_count)
    parts = formation_to_lines(formation)
    outfield_needed = 4 if side_count == 5 else 6
    if sum(parts) != outfield_needed:
        parts = [1,2,1] if outfield_needed == 4 else [2,1,2,1]
        formation = "-".join(map(str, parts))

    def _to_int_or(v, default):
        try:
            if v is None or (hasattr(pd, "isna") and pd.isna(v)):
                return default
            return int(v)
        except Exception:
            return default

    # Current lineup -> names + (line, slot) + GA
    ld = normalize_lineup_names(lineup_df.copy())
    current_gk = ld[ld["is_gk"] == True]["name"].dropna().astype(str).tolist()

    current_assign = {}
    current_ga = {}
    for _, r in ld[ld["is_gk"] != True].iterrows():
        ln = _to_int_or(r.get("line"), -1)
        sl = _to_int_or(r.get("slot"), -1)
        if ln >= 0 and sl >= 0:
            nm = str(r.get("name") or r.get("player_name") or "").strip()
            if nm:
                current_assign[(ln, sl)] = nm
                g = _to_int_or(r.get("goals"), 0)
                a = _to_int_or(r.get("assists"), 0)
                current_ga[(ln, sl)] = (g, a)

    pool = all_players["name"].dropna().astype(str).tolist()

    # GK picker
    st.caption("Goalkeeper")
    gk_default = current_gk[0] if current_gk else "—"
    gk_options = ["—"] + pool
    gk_idx = gk_options.index(gk_default) if gk_default in gk_options else 0
    gk_pick = st.selectbox("GK", gk_options, index=gk_idx, key=f"{keypref}_gk")

    st.caption(f"Formation: **{formation}** (outfield)")
    # Build grid lines
    slot_values, goal_vals, assist_vals = {}, {}, {}
    used = {gk_pick} if gk_pick != "—" else set()

    for line_idx, count in enumerate(parts):
        st.write(f"Line {line_idx} — {count} slots")
        cols = st.columns(count)
        for j in range(count):
            key_base = f"{keypref}_L{line_idx}_S{j}"
            assigned_default = current_assign.get((line_idx, j), "—")
            avail = ["—"] + [n for n in pool if (n not in used or n == assigned_default)]
            sel_idx = avail.index(assigned_default) if assigned_default in avail else 0
            sel = cols[j].selectbox("Player", avail, index=sel_idx, key=f"{key_base}_sel")
            slot_values[(line_idx, j)] = sel
            if sel != "—": used.add(sel)

            g0, a0 = current_ga.get((line_idx, j), (0, 0))
            g = cols[j].number_input("G", 0, 50, int(g0), key=f"{key_base}_g")
            a = cols[j].number_input("A", 0, 50, int(a0), key=f"{key_base}_a")
            goal_vals[(line_idx, j)] = int(g)
            assist_vals[(line_idx, j)] = int(a)

    if st.button(f"💾 Save lineup for {team_name}", key=f"{keypref}_save"):
        s = service()
        if not s:
            st.error("Admin required.")
        else:
            s.table("lineups").delete().eq("match_id", mid).eq("team", team_name).execute()
            rows = []
            if gk_pick != "—":
                rows.append({
                    "id": str(uuid.uuid4()),
                    "match_id": mid,
                    "team": team_name,
                    "player_id": None,
                    "player_name": gk_pick,
                    "name": gk_pick,
                    "is_gk": True,
                    "goals": 0, "assists": 0,
                    "line": None, "slot": None, "position": None
                })
            for (ln, sl), nm in slot_values.items():
                if nm == "—": continue
                rows.append({
                    "id": str(uuid.uuid4()),
                    "match_id": mid,
                    "team": team_name,
                    "player_id": None,
                    "player_name": nm,
                    "name": nm,
                    "is_gk": False,
                    "goals": int(goal_vals[(ln, sl)]),
                    "assists": int(assist_vals[(ln, sl)]),
                    "line": int(ln),
                    "slot": int(sl),
                    "position": None
                })
            if rows:
                for i in range(0, len(rows), 500):
                    s.table("lineups").insert(rows[i:i+500]).execute()
            clear_caches(); st.success("Lineup saved."); st.rerun()

# ------------------------------
# Tap-to-place lineup editor (iPhone-friendly)
# ------------------------------
def click_lineup_editor(team_name: str, mid: str, side_count: int, formation: str,
                        lineup_df: pd.DataFrame, all_players: pd.DataFrame, keypref: str,
                        *, left_half: bool, lfact_for_auto: Optional[pd.DataFrame] = None):
    if not CANVAS_OK:
        st.info("Tap editor requires 'streamlit-drawable-canvas'. Using slot editor.")
        lineup_slots_editor(team_name, mid, side_count, formation, lineup_df, all_players, keypref=keypref)
        return

    formation = validate_formation(formation, side_count)
    parts = formation_to_lines(formation)
    outfield_needed = 4 if side_count == 5 else 6
    if sum(parts) != outfield_needed:
        parts = [1,2,1] if outfield_needed == 4 else [2,1,2,1]
        formation = "-".join(map(str, parts))

    sk_assign = f"{keypref}_assign"
    sk_gk = f"{keypref}_gk"
    sk_ga = f"{keypref}_ga"
    sk_sel = f"{keypref}_sel"
    sk_clicks = f"{keypref}_clicks"

    def _from_db():
        ld = normalize_lineup_names(lineup_df.copy())
        assign = {}
        ga = {}
        gk = None
        for _, r in ld.iterrows():
            nm = str(r.get("name") or r.get("player_name") or "").strip()
            if not nm: continue
            if bool(r.get("is_gk")):
                gk = nm
            else:
                ln = r.get("line"); sl = r.get("slot")
                if pd.notna(ln) and pd.notna(sl):
                    ln = int(ln); sl = int(sl)
                    assign[(ln, sl)] = nm
                    ga[(ln, sl)] = (int(r.get("goals") or 0), int(r.get("assists") or 0))
        st.session_state[sk_assign] = assign
        st.session_state[sk_ga] = ga
        st.session_state[sk_gk] = gk
        st.session_state[sk_sel] = None
        st.session_state[sk_clicks] = 0

    if sk_assign not in st.session_state:
        _from_db()

    pool = all_players["name"].dropna().astype(str).tolist()
    assign: Dict[Tuple[int,int], str] = dict(st.session_state.get(sk_assign, {}))
    ga: Dict[Tuple[int,int], Tuple[int,int]] = dict(st.session_state.get(sk_ga, {}))
    gk_name: Optional[str] = st.session_state.get(sk_gk)
    selected: Optional[str] = st.session_state.get(sk_sel)

    assigned_names = set(assign.values())
    bench = [n for n in pool if n not in assigned_names and n != gk_name]

    cTop = st.container()
    cPitch, cSide = st.columns([3, 2])

    with cSide:
        st.markdown(f"**{team_name} — Bench / Select**")
        q = st.text_input("Search", key=f"{keypref}_q")
        show_bench = [n for n in bench if (not q or q.lower() in n.lower())]
        st.caption("Tap a name to select, then tap the pitch")
        cols = st.columns(3)
        i = 0
        for nm in (show_bench + sorted(list(assigned_names))):
            if q and nm not in show_bench and q.lower() not in nm.lower():
                continue
            with cols[i%3]:
                if st.button(("✓ " if selected==nm else "") + nm, key=f"{keypref}_pick_{nm}"):
                    st.session_state[sk_sel] = nm; st.rerun()
            i += 1
        st.markdown("---")
        # GK controls
        st.caption("Goalkeeper")
        gk_opts = ["—"] + pool
        gk_idx = gk_opts.index(gk_name) if gk_name in gk_opts else 0
        new_gk = st.selectbox("GK", gk_opts, index=gk_idx, key=f"{keypref}_gksel")
        if new_gk != gk_name:
            st.session_state[sk_gk] = None if new_gk == "—" else new_gk
            st.rerun()

        # Auto-pick & Reset & Save
        colB1, colB2, colB3 = st.columns(3)
        if colB1.button("↺ Reset", key=f"{keypref}_reset"):
            _from_db(); st.rerun()

        def _auto_pick():
            names_rank = pool
            if lfact_for_auto is not None and not lfact_for_auto.empty:
                agg = lfact_for_auto.groupby("name").agg(G=("goals","sum"), A=("assists","sum")).reset_index()
                agg["GA"] = agg["G"] + agg["A"]
                ranked = agg.sort_values(["GA","G","A"], ascending=[False,False,False])["name"].tolist()
                names_rank = ranked + [n for n in pool if n not in ranked]
            gk_candidates = []
            if lfact_for_auto is not None and not lfact_for_auto.empty:
                gk_candidates = lfact_for_auto[lfact_for_auto["is_gk"]==True]["name"].value_counts().index.tolist()
            new_gk = None
            for nm in gk_candidates + pool:
                if nm in pool:
                    new_gk = nm; break
            needed = sum(parts)
            picks = []
            for nm in names_rank:
                if nm == new_gk: continue
                if nm in picks: continue
                picks.append(nm)
                if len(picks) >= needed: break
            new_assign = {}
            idx = 0
            for ln, count in enumerate(parts):
                for sl in range(count):
                    if idx < len(picks):
                        new_assign[(ln, sl)] = picks[idx]; idx += 1
            st.session_state[sk_assign] = new_assign
            st.session_state[sk_ga] = {}
            st.session_state[sk_gk] = new_gk
            st.session_state[sk_sel] = None

        if colB2.button("✨ Auto-Pick", key=f"{keypref}_autopick"):
            _auto_pick(); st.rerun()

        s = service()
        if colB3.button("💾 Save lineup", key=f"{keypref}_save"):
            if not s: st.error("Admin required.")
            else:
                s.table("lineups").delete().eq("match_id", mid).eq("team", team_name).execute()
                rows = []
                if st.session_state.get(sk_gk):
                    rows.append({
                        "id": str(uuid.uuid4()), "match_id": mid, "team": team_name,
                        "player_id": None, "player_name": st.session_state[sk_gk], "name": st.session_state[sk_gk],
                        "is_gk": True, "goals": 0, "assists": 0, "line": None, "slot": None, "position": None
                    })
                for (ln, sl), nm in st.session_state[sk_assign].items():
                    g0,a0 = st.session_state[sk_ga].get((ln,sl), (0,0))
                    rows.append({
                        "id": str(uuid.uuid4()), "match_id": mid, "team": team_name,
                        "player_id": None, "player_name": nm, "name": nm,
                        "is_gk": False, "goals": int(g0), "assists": int(a0),
                        "line": int(ln), "slot": int(sl), "position": None
                    })
                if rows:
                    for i in range(0, len(rows), 500):
                        s.table("lineups").insert(rows[i:i+500]).execute()
                clear_caches(); st.success("Saved."); st.rerun()

    # Pitch canvas (tap target)
    with cPitch:
        canvas_w = 360 if st.session_state.get("compact") else 720
        canvas_h = int(canvas_w * 0.58)
        bg = pitch_bg_image(canvas_w, canvas_h)
        canvas_res = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=0,
            background_color="#1a3c28",
            background_image=bg,
            height=canvas_h, width=canvas_w,
            drawing_mode="point",
            key=f"canvas_{keypref}"
        )

        # Handle clicks
        if canvas_res.json_data is not None and selected:
            objs = canvas_res.json_data.get("objects", [])
            prev = int(st.session_state.get(sk_clicks, 0))
            if len(objs) > prev:
                pt = objs[-1]
                px, py = float(pt.get("left", 0)), float(pt.get("top", 0))
                x_pct = (px / canvas_w) * 100.0
                y_pct = (py / canvas_h) * 100.0
                coords = slot_centers_pct(parts, left_half=left_half)
                best = None; best_d = 1e9
                for (ln, sl, cx, cy) in coords:
                    d = (cx - x_pct)**2 + (cy - y_pct)**2
                    if d < best_d:
                        best_d = d; best = (ln, sl)
                if best is not None:
                    old = None
                    for k, v in list(assign.items()):
                        if v == selected: old = k
                    if old: del assign[old]
                    if best in assign and assign[best] != selected and old:
                        assign[old] = assign[best]
                    assign[best] = selected
                    st.session_state[sk_assign] = assign
                    st.session_state[sk_clicks] = len(objs)
                    st.rerun()

    # Inline G/A per assigned slot
    with cTop:
        if assign:
            st.caption(f"{team_name} — goals & assists per slot")
            for ln, count in enumerate(parts):
                cols = st.columns(count)
                for sl in range(count):
                    nm = assign.get((ln,sl))
                    with cols[sl]:
                        if nm:
                            st.markdown(f"**{nm}**")
                            g0,a0 = ga.get((ln,sl), (0,0))
                            g = st.number_input("G", 0, 50, int(g0), key=f"{keypref}_g_{ln}_{sl}")
                            a = st.number_input("A", 0, 50, int(a0), key=f"{keypref}_a_{ln}_{sl}")
                            ga[(ln,sl)] = (int(g), int(a))
                        else:
                            st.write("—")
            st.session_state[sk_ga] = ga

# ------------------------------
# Add Match
# ------------------------------
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

# ------------------------------
# Matches
# ------------------------------
def page_matches():
    matches = fetch_matches()
    lineups = fetch_lineups()
    players = fetch_players()

    if matches.empty:
        st.info("No matches yet. Use 'Add Match' to create one.")
        return

    # Select Season & GW
    seasons = sorted(matches["season"].dropna().astype(int).unique().tolist())
    colA, colB, colC = st.columns([1,2,2])
    sel_season = colA.selectbox("Season", seasons, index=len(seasons)-1, key="pm_season")

    msub = matches[matches["season"] == sel_season].copy().sort_values("gw")
    labels = msub.apply(lambda r: f"GW {int(r['gw'])} — {r['team_a']} {int(r.get('score_a') or 0)}–{int(r.get('score_b') or 0)} {r['team_b']}", axis=1)
    id_map = {labels.iloc[i]: msub.iloc[i]["id"] for i in range(len(msub))}
    pick = colB.selectbox("Match", list(id_map.keys()), index=len(id_map)-1, key="pm_pick")
    mid = id_map[pick]
    editor_mode = colC.radio("Editor", ["Tap", "Slot"], horizontal=True, key=f"pm_edit_{mid}")

    m = msub[msub["id"] == mid].iloc[0]

    # Lineups filtered
    a_rows = lineups[(lineups["match_id"] == mid) & (lineups["team"] == "Non-bibs")].copy()
    b_rows = lineups[(lineups["match_id"] == mid) & (lineups["team"] == "Bibs")].copy()

    # Banner
    lcol, ccol, rcol = st.columns([3, 2, 3])
    with lcol:  st.markdown(f"### **{m['team_a']}**")
    with ccol:
        st.markdown(f"### **{int(m.get('score_a') or 0)} – {int(m.get('score_b') or 0)}**")
        motm = str(m.get("motm_name") or "")
        if motm: st.caption(f"⭐ MOTM: **{motm}**")
    with rcol:  st.markdown(f"### **{m['team_b']}**")

    # Admin: quick match info + formations
    if st.session_state.get("is_admin"):
        with st.expander("Edit match & formations (admin)", expanded=False):
            c1,c2,c3,c4,c5 = st.columns([1.2,1.2,1.1,1.2,2.4])
            sc_a = c1.number_input("Score (Non-bibs)", 0, 999, int(m.get("score_a") or 0), key=f"sc_a_{mid}")
            sc_b = c2.number_input("Score (Bibs)", 0, 999, int(m.get("score_b") or 0), key=f"sc_b_{mid}")
            side_count = int(m.get("side_count") or 5)
            side_new = c3.selectbox("Side count", [5,7], index=(0 if side_count==5 else 1), key=f"side_{mid}")
            motm_in = c4.text_input("MOTM name", value=str(m.get("motm_name") or ""), key=f"motm_{mid}")
            d = c5.date_input("Date", value=pd.to_datetime(m.get("date") or date.today()).date(), key=f"dt_{mid}")

            presets5 = ["1-2-1","1-3","2-2","3-1"]
            presets7 = ["2-1-2-1","3-2-1","2-3-1"]
            options = presets7 if side_new == 7 else presets5

            colf1, colf2, colf3 = st.columns([2,2,1])
            fa_pick = colf1.selectbox("Formation (Non-bibs)", options,
                        index=(options.index(m.get("formation_a")) if m.get("formation_a") in options else 0),
                        key=f"fa_{mid}")
            fb_pick = colf2.selectbox("Formation (Bibs)", options,
                        index=(options.index(m.get("formation_b")) if m.get("formation_b") in options else 0),
                        key=f"fb_{mid}")

            if st.button("Save match & formations", key=f"save_m_{mid}"):
                s = service()
                if not s: st.error("Admin required.")
                else:
                    s.table("matches").update({
                        "score_a": int(sc_a),
                        "score_b": int(sc_b),
                        "motm_name": motm_in,
                        "date": str(d),
                        "side_count": int(side_new),
                        "formation_a": validate_formation(fa_pick, side_new),
                        "formation_b": validate_formation(fb_pick, side_new),
                    }).eq("id", mid).execute()
                    clear_caches(); st.success("Saved."); st.rerun()

    # Combined pitch (always validated to side_count)
    side_count = int(m.get("side_count") or 5)
    fa_render = validate_formation(m.get("formation_a"), side_count)
    fb_render = validate_formation(m.get("formation_b"), side_count)
    st.caption(f"{m['team_a']} (left)  vs  {m['team_b']} (right)")
    render_match_pitch_combined(a_rows, b_rows, fa_render, fb_render, m.get("motm_name"), m["team_a"], m["team_b"])

    # Admin: lineup editor (choose mode)
    if st.session_state.get("is_admin"):
        with st.expander("Arrange lineup (admin)", expanded=False):
            lfact = build_fact(players, matches, lineups)
            colA, colB = st.columns(2)
            if editor_mode == "Tap":
                with colA:
                    click_lineup_editor("Non-bibs", mid, side_count, fa_render, a_rows, players,
                                        keypref=f"A_{mid}", left_half=True, lfact_for_auto=lfact)
                with colB:
                    click_lineup_editor("Bibs", mid, side_count, fb_render, b_rows, players,
                                        keypref=f"B_{mid}", left_half=False, lfact_for_auto=lfact)
            else:
                with colA:
                    lineup_slots_editor("Non-bibs", mid, side_count, fa_render, a_rows, players, keypref=f"A_{mid}")
                with colB:
                    lineup_slots_editor("Bibs", mid, side_count, fb_render, b_rows, players, keypref=f"B_{mid}")

# ------------------------------
# Players: cards + teammate/nemesis + ratings
# ------------------------------
def form_string(results: List[str], n: int = 5) -> str:
    r = results[-n:][::-1]
    out = []
    for x in r:
        if x == "W": out.append("🟩")
        elif x == "D": out.append("🟨")
        else: out.append("🟥")
    return "".join(out) if out else "—"

def _percentile(series: pd.Series, v: float) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty: return 50.0
    return float((series <= v).mean() * 100)

def _ratings_from_dataset(lfact: pd.DataFrame, mine: pd.DataFrame) -> Dict[str,int]:
    if lfact.empty or mine.empty:
        return {"OVR": 50, "Shooting": 50, "Passing": 50, "Impact": 50}
    gp = mine["match_id"].nunique()
    goals = mine["goals"].sum(); assists = mine["assists"].sum()
    gpg = goals / gp if gp else 0
    apg = assists / gp if gp else 0
    winp_p = (mine["result"].eq("W").mean() * 100.0) if gp else 0
    agg = lfact.groupby("name").agg(
        GP=("match_id","nunique"),
        Goals=("goals","sum"),
        Assists=("assists","sum"),
        Wins=("result", lambda s: (s=="W").sum())
    ).reset_index()
    agg["GPG"] = agg["Goals"]/agg["GP"].replace(0,np.nan)
    agg["APG"] = agg["Assists"]/agg["GP"].replace(0,np.nan)
    agg["Win%"] = (agg["Wins"]/agg["GP"].replace(0,np.nan))*100
    p_shoot = _percentile(agg["GPG"], gpg)
    p_pass  = _percentile(agg["APG"], apg)
    p_imp   = _percentile(agg["Win%"], winp_p)
    def map_rating(p): return int(round(40 + (p/100.0)*52))  # 40-92
    shooting = map_rating(p_shoot)
    passing  = map_rating(p_pass)
    impact   = map_rating(p_imp)
    ovr      = int(round(0.4*shooting + 0.35*passing + 0.25*impact))
    return {"OVR": ovr, "Shooting": shooting, "Passing": passing, "Impact": impact}

def best_teammate_table(lfact: pd.DataFrame, player: str, min_gp_together: int = 1) -> pd.DataFrame:
    mine = lfact[lfact["name"] == player]
    if mine.empty: return pd.DataFrame(columns=["Mate","GP","W","Win%"])
    same = mine.merge(lfact, on=["match_id","team"])
    same = same[same["name_x"] != same["name_y"]]
    grp = same.groupby(["name_x","name_y"])
    gp = grp["match_id"].nunique().rename("GP")
    w  = same[same["result_x"]=="W"].groupby(["name_x","name_y"])["match_id"].nunique().rename("W")
    out = pd.concat([gp,w], axis=1).fillna(0).reset_index()
    out = out[out["name_x"] == player]
    out["Win%"] = ((out["W"]/out["GP"]).replace(0,np.nan)*100).fillna(0).round(1)
    out = out[out["GP"] >= int(min_gp_together)]
    out = out.rename(columns={"name_y":"Mate"})
    return out[["Mate","GP","W","Win%"]].sort_values(["Win%","GP"], ascending=[False,False])

def nemesis_table_for_player(lfact: pd.DataFrame, player: str, min_meetings: int = 1) -> pd.DataFrame:
    mine = lfact[lfact["name"] == player]
    if mine.empty: return pd.DataFrame(columns=["Nemesis","GP","W","D","L","Win%"])
    opp = mine.merge(lfact, on="match_id")
    opp = opp[opp["team_x"] != opp["team_y"]]
    grp = opp.groupby(["name_x","name_y"])
    gp = grp["match_id"].nunique().rename("GP")
    w  = opp[(opp["result_x"]=="W")].groupby(["name_x","name_y"])["match_id"].nunique().rename("W")
    d  = opp[(opp["result_x"]=="D")].groupby(["name_x","name_y"])["match_id"].nunique().rename("D")
    l  = opp[(opp["result_x"]=="L")].groupby(["name_x","name_y"])["match_id"].nunique().rename("L")
    out = pd.concat([gp,w,d,l], axis=1).fillna(0).reset_index()
    out = out[out["name_x"] == player].rename(columns={"name_y":"Nemesis"})
    out["Win%"] = ((out["W"]/out["GP"]).replace(0,np.nan)*100).fillna(0).round(1)
    out = out[out["GP"] >= int(min_meetings)]
    return out[["Nemesis","GP","W","D","L","Win%"]].sort_values(["Win%","GP"], ascending=[True,False])

def page_players():
    players = fetch_players()
    matches = fetch_matches()
    lineups = fetch_lineups()
    lfact = build_fact(players, matches, lineups)

    if players.empty:
        st.info("No players yet. Add via Player Manager.")
        return

    names = players["name"].dropna().astype(str).tolist()
    sel = st.selectbox("Player", names, key="pp_pick")

    mine = lfact[lfact["name"] == sel].copy().sort_values(["season","gw"])
    if mine.empty:
        st.info("No games recorded for this player yet.")
        return

    gp = mine["match_id"].nunique()
    w = (mine["result"]=="W").sum()
    d = (mine["result"]=="D").sum()
    l = (mine["result"]=="L").sum()
    goals = mine["goals"].sum()
    assists = mine["assists"].sum()
    ga = goals + assists
    gapg = (ga / gp) if gp else 0
    contrib = mine["contrib"].mean() if "contrib" in mine.columns else 0

    n_last = st.number_input("Last N games", 1, max(1, gp), min(5, gp), key="pp_last")
    frm = form_string(mine["result"].tolist(), n=int(n_last))

    ratings = _ratings_from_dataset(lfact, mine)

    pr = players[players["name"] == sel].iloc[0]
    avatar = pr.get("photo_url") or None
    av_html = (
        f"<img src='{avatar}' style='width:96px;height:96px;border-radius:14px;object-fit:cover;border:1px solid rgba(255,255,255,.25)'>"
        if avatar else
        f"<div style='width:96px;height:96px;border-radius:14px;background:#1a2430;color:#e9eef3;display:flex;align-items:center;justify-content:center;font-weight:900;font-size:1.35rem'>{initials(sel)}</div>"
    )

    st.markdown(f"""
    <div class='badge'>
      {av_html}
      <div style='display:flex;flex-direction:column;gap:.2rem'>
        <div style='font-weight:900;font-size:1.15rem'>{sel}</div>
        <div class='small'>Form: {frm}</div>
        <div style='display:flex;gap:.6rem;margin-top:.2rem'>
          <span class='pillR'><span class='ovr'>OVR</span> {ratings["OVR"]}</span>
          <span class='pillR'>Shooting {ratings["Shooting"]}</span>
          <span class='pillR'>Passing {ratings["Passing"]}</span>
          <span class='pillR'>Impact {ratings["Impact"]}</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("#### Overview")
    st.markdown("<div class='card-grid'>", unsafe_allow_html=True)
    for k, v in [
        ("Games (GP)", gp),
        ("W-D-L", f"{w}-{d}-{l}"),
        ("Win%", f"{(w/gp*100 if gp else 0):.1f}"),
        ("Goals", goals),
        ("Assists", assists),
        ("G+A", ga),
        ("G+A / GP", f"{gapg:.2f}"),
        ("Team Contrib%", f"{contrib:.1f}")
    ]:
        st.markdown(f"<div class='metric'><div class='k'>{k}</div><div class='v'>{v}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Recent games")
    recent = mine.sort_values(["season","gw"], ascending=[False, False]).head(int(n_last))
    show = recent[["season","gw","team","for","against","result","goals","assists"]].rename(columns={
        "for":"For","against":"Ag","result":"Res","goals":"G","assists":"A"
    })
    st.dataframe(show, use_container_width=True, hide_index=True)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Best teammate")
        min_meet = st.number_input("Min games together", 1, 50, 1, key="pp_bt_min")
        bt = best_teammate_table(lfact, sel, int(min_meet))
        if not bt.empty:
            st.dataframe(bt.head(10), use_container_width=True, hide_index=True)
        else:
            st.caption("—")
    with c2:
        st.markdown("#### Nemesis")
        min_meet_n = st.number_input("Min meetings vs opponent", 1, 50, 1, key="pp_nem_min")
        nem = nemesis_table_for_player(lfact, sel, int(min_meet_n))
        if not nem.empty:
            st.dataframe(nem.head(10), use_container_width=True, hide_index=True)
        else:
            st.caption("—")

# ------------------------------
# Stats page (dropdown + filters)
# ------------------------------
def filter_fact(lfact: pd.DataFrame, season: Optional[int], last_gw: Optional[int]) -> pd.DataFrame:
    df = lfact.copy()
    if season and season != -1:
        df = df[df["season"] == int(season)]
    if last_gw and int(last_gw) > 0 and not df.empty:
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
    same = same[same["name_x"] < same["name_y"]]
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

def page_stats():
    players = fetch_players()
    matches = fetch_matches()
    lineups = fetch_lineups()
    lfact = build_fact(players, matches, lineups)
    if lfact.empty:
        st.info("No data yet.")
        return

    st.markdown("### Stats")

    seasons_unique = sorted(lfact["season"].dropna().astype(int).unique().tolist())
    seasons = [-1] + seasons_unique
    default_index = max(0, len(seasons) - 1)

    c1,c2,c3,c4 = st.columns(4)
    sel_season = c1.selectbox(
        "Season (or All)",
        seasons,
        index=default_index,
        format_func=lambda x: "All" if x == -1 else str(x),
        key="st_season",
    )
    min_games = c2.number_input("Min games", 0, 100, 1, key="st_min")
    last_gw = c3.number_input("Last N GWs (0 = all)", 0, 200, 0, key="st_last")
    top_n = c4.number_input("Rows", 5, 200, 25, key="st_rows")

    metric = st.selectbox(
        "Metric",
        ["Top Scorers","Top Assisters","Top G+A","Team Contribution%","MOTM Count","Best Duos","Nemesis"],
        key="st_metric"
    )
    season_filter = None if sel_season == -1 else int(sel_season)

    if metric in ["Top Scorers","Top Assisters","Top G+A","Team Contribution%","MOTM Count"]:
        agg = player_agg(lfact, season_filter, int(min_games), int(last_gw))
        if agg.empty:
            st.caption("No rows.")
            return
        if metric == "Top Scorers":
            out = agg.sort_values(["Goals","G+A","Win%"], ascending=[False,False,False]).head(int(top_n))
        elif metric == "Top Assisters":
            out = agg.sort_values(["Assists","G+A","Win%"], ascending=[False,False,False]).head(int(top_n))
        elif metric == "Top G+A":
            out = agg.sort_values(["G+A","Goals","Assists"], ascending=[False,False,False]).head(int(top_n))
        elif metric == "Team Contribution%":
            out = agg.sort_values(["Team Contrib%","G+A","GP"], ascending=[False,False,False]).head(int(top_n))
        else:  # MOTM Count
            m = fetch_matches().copy()
            cnt = m["motm_name"].dropna().value_counts().rename_axis("name").reset_index(name="MOTM")
            out = agg.merge(cnt, on="name", how="left").fillna({"MOTM":0}).sort_values(["MOTM","G+A"], ascending=[False,False]).head(int(top_n))
        st.dataframe(out, use_container_width=True, hide_index=True)
    elif metric == "Best Duos":
        out = duos_table(lfact, season_filter, int(min_games), int(last_gw)).head(int(top_n))
        st.dataframe(out, use_container_width=True, hide_index=True)
    else:
        out = nemesis_table(lfact, season_filter, int(min_games), int(last_gw)).head(int(top_n))
        st.dataframe(out, use_container_width=True, hide_index=True)

# ------------------------------
# Awards (MOTM from matches + POTM add)
# ------------------------------
def page_awards():
    matches = fetch_matches()
    awards = fetch_awards()

    st.markdown("### Awards")

    st.markdown("**MOTM (from matches):**")
    if matches.empty or "motm_name" not in matches.columns:
        st.caption("No matches.")
    else:
        motm = matches["motm_name"].dropna()
        if motm.empty:
            st.caption("No MOTMs recorded yet.")
        else:
            motm_tbl = motm.value_counts().rename_axis("name").reset_index(name="MOTM")
            st.dataframe(motm_tbl, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("**POTM (manual):**")
    potm = awards[awards["type"]=="POTM"] if not awards.empty else pd.DataFrame(columns=["season","month","player_name","notes"])
    if not potm.empty:
        potm_view = potm.sort_values(["season","month"]).rename(columns={"player_name":"POTM"})
        st.dataframe(potm_view[["season","month","POTM","notes"]], use_container_width=True, hide_index=True)
    else:
        st.caption("No POTMs yet.")

    if st.session_state.get("is_admin"):
        with st.expander("Add POTM", expanded=False):
            c1,c2,c3,c4 = st.columns(4)
            season = c1.number_input("Season", 2023, 2100, datetime.now().year, key="aw_s")
            month = c2.number_input("Month", 1, 12, datetime.now().month, key="aw_m")
            pl = fetch_players()
            pname = c3.selectbox("Player", pl["name"].tolist() if not pl.empty else [], key="aw_p")
            notes = c4.text_input("Notes", key="aw_n")
            if st.button("Save POTM", key="aw_save"):
                s = service()
                if s:
                    s.table("awards").insert({
                        "id": str(uuid.uuid4()), "season": int(season), "month": int(month),
                        "type": "POTM", "gw": None,
                        "player_id": None, "player_name": pname, "notes": notes
                    }).execute()
                    clear_caches(); st.success("POTM saved."); st.rerun()

# ------------------------------
# Player Manager
# ------------------------------
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

# ------------------------------
# Import / Export
# ------------------------------
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
                s = service()
                if not s: st.error("Admin required.")
                else:
                    for (mid, team), grp in df.groupby(["match_id","team"]):
                        s.table("lineups").delete().eq("match_id", mid).eq("team", team).execute()
                        recs = grp.to_dict("records")
                        for i in range(0, len(recs), 500):
                            s.table("lineups").insert(recs[i:i+500]).execute()
                    clear_caches(); st.success("Lineups imported.")

    st.divider()
    st.markdown("#### Export")
    pl = fetch_players(); mt = fetch_matches(); ln = fetch_lineups()
    col1,col2,col3 = st.columns(3)
    col1.download_button("players.csv", pl.to_csv(index=False).encode("utf-8"), "players.csv", "text/csv")
    col2.download_button("matches.csv", mt.to_csv(index=False).encode("utf-8"), "matches.csv", "text/csv")
    col3.download_button("lineups.csv", ln.to_csv(index=False).encode("utf-8"), "lineups.csv", "text/csv")

# ------------------------------
# Router
# ------------------------------
def run_app():
    header()
    sidebar_admin()
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
