# app.py — Powerleague Stats (Matches phase, combined pitch + 5s/7s guard)

import streamlit as st
import pandas as pd
from datetime import date
from typing import Optional, List
from supabase import create_client

# -------------------------
# Streamlit config
# -------------------------
st.set_page_config(page_title="Powerleague Stats", layout="wide", initial_sidebar_state="collapsed")

# -------------------------
# Secrets / Supabase
# -------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
SUPABASE_SERVICE_KEY = st.secrets.get("SUPABASE_SERVICE_KEY", SUPABASE_ANON_KEY)
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "")
AVATAR_BUCKET = st.secrets.get("AVATAR_BUCKET", "avatars")

sb = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
sb_service = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# -------------------------
# Caching helpers
# -------------------------
def clear_caches():
    st.cache_data.clear()

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

def service():
    return sb_service if st.session_state.get("is_admin") else None

# -------------------------
# Core helpers
# -------------------------
def formation_to_lines(formation: Optional[str]) -> List[int]:
    try:
        parts = [int(x) for x in str(formation or "").strip().split("-") if str(x).strip().isdigit()]
        return parts if parts else []
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
    Non-GKs get centered defaults on missing values; clamp to formation bounds.
    """
    rows = df.copy()
    parts = formation_to_lines(formation)
    if not parts:
        parts = [1, 2, 1]
    n_lines = max(1, len(parts))
    max_slots = max(parts + [1])

    # ensure columns exist
    for c in ["is_gk", "line", "slot", "goals", "assists", "name", "player_name"]:
        if c not in rows.columns:
            rows[c] = None

    rows["is_gk"] = rows["is_gk"].fillna(False).astype(bool)
    rows["goals"] = pd.to_numeric(rows["goals"], errors="coerce").fillna(0).astype(int)
    rows["assists"] = pd.to_numeric(rows["assists"], errors="coerce").fillna(0).astype(int)

    # name resolution
    if "name" not in rows.columns:
        rows["name"] = rows["player_name"]
    rows["name"] = rows["name"].fillna(rows["player_name"]).fillna("").astype(str)

    rows["line"] = pd.to_numeric(rows["line"], errors="coerce").astype("Int64")
    rows["slot"] = pd.to_numeric(rows["slot"], errors="coerce").astype("Int64")

    center_line = (n_lines - 1) // 2

    for i in rows.index:
        if bool(rows.at[i, "is_gk"]):
            rows.at[i, "line"] = pd.NA
            rows.at[i, "slot"] = pd.NA
            continue

        # LINE default: center row
        ln = rows.at[i, "line"]
        ln = center_line if pd.isna(ln) else int(ln)
        ln = max(0, min(ln, n_lines - 1))
        rows.at[i, "line"] = ln

        # SLOT default: centered within that line
        slots = int(parts[ln])
        offset = (max_slots - slots) // 2
        sl = rows.at[i, "slot"]
        sl = (offset + (slots - 1) // 2) if pd.isna(sl) else int(sl)
        sl = max(offset, min(sl, offset + slots - 1))
        rows.at[i, "slot"] = sl

    return rows

def normalize_lineup_names(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a 'name' column exists and is string."""
    out = df.copy()
    if "name" not in out.columns:
        out["name"] = out["player_name"] if "player_name" in out.columns else ""
    else:
        if "player_name" in out.columns:
            out["name"] = out["name"].fillna(out["player_name"])
        out["name"] = out["name"].fillna("")
    out["name"] = out["name"].astype(str)
    return out

# -------------------------
# Pitch UI helpers (compact & readable)
# -------------------------
def stat_pill(goals: int, assists: int) -> str:
    """Premium GA chip: clear monogram tags that read well on mobile."""
    parts = []
    if (goals or 0) > 0:
        parts.append(f"<span class='tag tag-g'>G</span><b>{int(goals)}</b>")
    if (assists or 0) > 0:
        parts.append(f"<span class='tag tag-a'>A</span><b>{int(assists)}</b>")
    if not parts:
        return ""
    return f"<span class='pill'>{'&nbsp;&nbsp;&nbsp;'.join(parts)}</span>"

def slot_html(x_pct: float, y_pct: float, name: str, *, motm: bool=False, pill: str = "", is_gk: bool=False) -> str:
    """Player bubble + name + pill; GK has distinct style + GK chip."""
    cls = "bubble"
    if motm: cls += " motm"
    if is_gk: cls += " gk"
    init = "".join([t[0] for t in name.split()[:2]]).upper() or "?"
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
    One compact, premium pitch with both teams.
    SIDE-ORIENTED: lines run goal→center along X for each half (mirrored for Team B).
    Bigger bubbles + tighter margins = less empty space, still no overlaps.
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
    /* Penalty boxes & goals left/right (tightened) */
    .box-left{position:absolute;left:0;top:20%;bottom:20%;width:16.5%;border:2px solid #ffffff;border-left:none}
    .six-left{position:absolute;left:0;top:39%;bottom:39%;width:7.6%;border:2px solid #ffffff;border-left:none}
    .pen-dot-left{position:absolute;left:11%;top:50%;transform:translate(-50%,-50%);width:6px;height:6px;background:#ffffff;border-radius:999px}
    .goal-left{position:absolute;left:-1.0%;top:47%;bottom:47%;width:1%;border:2px solid #ffffff;border-right:none}

    .box-right{position:absolute;right:0;top:20%;bottom:20%;width:16.5%;border:2px solid #ffffff;border-right:none}
    .six-right{position:absolute;right:0;top:39%;bottom:39%;width:7.6%;border:2px solid #ffffff;border-right:none}
    .pen-dot-right{position:absolute;right:11%;top:50%;transform:translate(50%,-50%);width:6px;height:6px;background:#ffffff;border-radius:999px}
    .goal-right{position:absolute;right:-1.0%;top:47%;bottom:47%;width:1%;border:2px solid #ffffff;border-left:none}

    /* Players: larger bubbles, still responsive */
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

    /* G/A pill */
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

    def lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    # Sanitize per formation
    a_rows = _ensure_positions(a_rows, formation_a)
    b_rows = _ensure_positions(b_rows, formation_b)
    parts_a = formation_to_lines(formation_a) or [1,2,1]
    parts_b = formation_to_lines(formation_b) or [1,2,1]

    # tighter vertical margins; use more of height
    y_top_margin, y_bot_margin = 6, 6
    inner_h = 100 - y_top_margin - y_bot_margin

    # make halves wider to reduce empty space
    left_min, left_max  = 5, 48    # Team A x from 5% to 48%
    right_min, right_max = 52, 95  # Team B x from 95% to 52%

    def _place_side(rows: pd.DataFrame, parts: List[int], *, left_half: bool):
        out = []
        n_lines = max(1, len(parts))

        # GK very close to goal edge but inside field
        gk = rows[rows.get("is_gk") == True]
        if not gk.empty:
            r = gk.iloc[0]
            nm = str(r.get("name") or r.get("player_name") or "")
            x = lerp(left_min, left_max, -0.03) if left_half else lerp(right_max, right_min, -0.03)
            y = 50
            pill = stat_pill(int(r.get("goals") or 0), int(r.get("assists") or 0)) if show_stats else ""
            out.append(slot_html(x, y, nm, motm=(motm_name==nm), pill=pill, is_gk=True))

        # Lines: goal → center along X, spread players along Y
        for line_idx in range(n_lines):
            t = (line_idx + 1) / (n_lines + 1)
            x = lerp(left_min, left_max, t) if left_half else lerp(right_max, right_min, t)

            line_df = rows[(rows.get("is_gk") != True) & (rows.get("line") == line_idx)].copy()
            if line_df.empty:
                continue
            line_df["slot"] = pd.to_numeric(line_df["slot"], errors="coerce")
            line_df = line_df.sort_values("slot", na_position="last").reset_index(drop=True)

            count = len(line_df)
            for j in range(count):
                rr = line_df.iloc[j]
                nm = str(rr.get("name") or rr.get("player_name") or "")
                y_t = (j + 1) / (count + 1)
                y = y_top_margin + y_t * inner_h
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

# -------------------------
# Header / auth
# -------------------------
def header():
    left, mid, right = st.columns([3,1,3])
    with left:
        st.markdown("## Powerleague Stats — Matches")
    with mid:
        if st.button("Clear cache", use_container_width=True):
            clear_caches()
            st.success("Cache cleared.")
            st.experimental_rerun()
    with right:
        if not st.session_state.get("is_admin"):
            with st.popover("Admin login", use_container_width=True):
                pwd = st.text_input("Password", type="password", key="admin_pwd")
                if st.button("Login", use_container_width=True):
                    if ADMIN_PASSWORD and pwd == ADMIN_PASSWORD:
                        st.session_state["is_admin"] = True
                        st.success("Admin enabled")
                        st.experimental_rerun()
                    else:
                        st.error("Wrong password")
        else:
            st.markdown("**Admin:** ✅")
            if st.button("Log out", use_container_width=True):
                st.session_state["is_admin"] = False
                st.experimental_rerun()

# -------------------------
# Matches page
# -------------------------
def page_matches():
    matches = fetch_matches()
    lineups = fetch_lineups()

    if matches.empty:
        st.info("No matches yet.")
        return

    k = "pm"
    # picker
    seasons = sorted(matches["season"].dropna().astype(int).unique().tolist())
    sel_season = st.selectbox("Season", seasons, index=len(seasons)-1, key=f"{k}_season")

    msub = matches[matches["season"] == sel_season].copy().sort_values("gw")
    labels = msub.apply(lambda r: f"GW {int(r['gw'])} — {r['team_a']} {int(r.get('score_a') or 0)}–{int(r.get('score_b') or 0)} {r['team_b']}", axis=1)
    id_map = {labels.iloc[i]: msub.iloc[i]["id"] for i in range(len(msub))}
    pick = st.selectbox("Match", list(id_map.keys()), index=len(id_map)-1, key=f"{k}_pick")
    mid = id_map[pick]
    m = msub[msub["id"] == mid].iloc[0]

    # lineups for each side
    a_rows = lineups[(lineups["match_id"] == mid) & (lineups["team"] == "Non-bibs")].copy()
    b_rows = lineups[(lineups["match_id"] == mid) & (lineups["team"] == "Bibs")].copy()

    # Guarantee we can show names (fix for 'g' NameError case)
    both = normalize_lineup_names(pd.concat([a_rows, b_rows], ignore_index=True))
    if both.empty:
        st.info("No lineup data yet for this match.")
        return

    # header banner
    lcol, ccol, rcol = st.columns([3, 2, 3])
    with lcol:
        st.markdown(f"### **{m['team_a']}**")
    with ccol:
        st.markdown(f"### **{int(m.get('score_a') or 0)} – {int(m.get('score_b') or 0)}**")
        motm = str(m.get("motm_name") or "")
        if motm:
            st.caption(f"⭐ MOTM: **{motm}**")
    with rcol:
        st.markdown(f"### **{m['team_b']}**")

    # admin: set formations (validated for 5s/7s)
    if st.session_state.get("is_admin"):
        presets5 = ["1-2-1", "1-3", "2-2", "3-1"]
        presets7 = ["2-1-2-1", "3-2-1", "2-3-1"]
        side_count = int(m.get("side_count") or 5)
        options = presets7 if side_count == 7 else presets5

        colf1, colf2, colf3 = st.columns([2, 2, 1])
        fa = colf1.selectbox(
            "Formation (Non-bibs)",
            options,
            index=(options.index(m.get("formation_a")) if m.get("formation_a") in options else 0),
            key=f"{k}_fa_{mid}",
        )
        fb = colf2.selectbox(
            "Formation (Bibs)",
            options,
            index=(options.index(m.get("formation_b")) if m.get("formation_b") in options else 0),
            key=f"{k}_fb_{mid}",
        )
        if colf3.button("Save formations", key=f"{k}_save_forms_{mid}"):
            s = service()
            if not s:
                st.error("Admin required.")
            else:
                fa_s = validate_formation(fa, side_count)
                fb_s = validate_formation(fb, side_count)
                s.table("matches").update({"formation_a": fa_s, "formation_b": fb_s}).eq("id", mid).execute()
                clear_caches()
                st.success("Formations updated.")
                st.experimental_rerun()

    # render combined pitch with validated formations
    side_count = int(m.get("side_count") or 5)
    fa_render = validate_formation(m.get("formation_a"), side_count)
    fb_render = validate_formation(m.get("formation_b"), side_count)

    st.caption(f"{m['team_a']} (left)  vs  {m['team_b']} (right)")
    render_match_pitch_combined(
        a_rows, b_rows, fa_render, fb_render, m.get("motm_name"), m["team_a"], m["team_b"], show_stats=True
    )

# -------------------------
# App entry
# -------------------------
def run_app():
    header()
    st.divider()
    page_matches()

if __name__ == "__main__":
    run_app()
