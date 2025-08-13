import io
from datetime import date
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import pillow_heif
from supabase import create_client, Client

# ---------------------------
# App config & global styles
# ---------------------------
st.set_page_config(
    page_title="Powerleague Stats",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
:root { --pitch-green:#1e7f3a; --chip-bg:rgba(255,255,255,.06); --chip-br:rgba(255,255,255,.18); }
footer,#MainMenu{visibility:hidden}
.pitch{background:radial-gradient(circle at 50% 20%, #2aa14d 0%, #1f8440 60%, #166332 100%);border-radius:18px;padding:10px;position:relative;box-shadow:0 8px 24px rgba(0,0,0,.2);border:1px solid rgba(255,255,255,.2);min-height:420px}
.pitch .grid{display:grid;gap:18px}
.line{display:grid;gap:10px}
.slot{display:flex;align-items:center;justify-content:center}
.chip{backdrop-filter:blur(6px);background:var(--chip-bg);border:1px solid var(--chip-br);color:#fff;padding:6px 10px;border-radius:14px;font-size:13px;display:inline-flex;align-items:center;gap:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.chip img{width:20px;height:20px;border-radius:9999px;object-fit:cover;border:1px solid rgba(255,255,255,.25)}
.banner{background:linear-gradient(90deg,#1e293b 0%,#0f172a 100%);color:#fff;padding:10px 14px;border-radius:14px;display:flex;align-items:center;justify-content:space-between;gap:12px;border:1px solid rgba(255,255,255,.15)}
.badge{background:#22c55e;color:#052e14;padding:4px 10px;border-radius:9999px;font-weight:600;font-size:12px}
.small{font-size:12px;opacity:.85}
.hr{height:1px;background:rgba(0,0,0,.08);margin:10px 0}
.tablecard{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.12);border-radius:12px;padding:10px}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------
# Secrets & Supabase clients
# ---------------------------
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

def ensure_bucket():
    try:
        s = service()
        if not s:
            return
        buckets = s.storage.list_buckets()
        names = {getattr(b, "name", getattr(b, "id", None)) for b in buckets}
        if AVATAR_BUCKET not in names:
            s.storage.create_bucket(AVATAR_BUCKET, public=True)
    except Exception:
        pass

try:
    pillow_heif.register_heif_opener()
except Exception:
    pass

ensure_bucket()

# ---------------------------
# Cached DB reads
# ---------------------------
@st.cache_data(ttl=30)
def fetch_players() -> pd.DataFrame:
    data = sb_public.table("players").select("*").execute().data or []
    return pd.DataFrame(data)

@st.cache_data(ttl=30)
def fetch_matches() -> pd.DataFrame:
    data = (
        sb_public.table("matches")
        .select("*")
        .order("season")
        .order("gw")
        .execute()
        .data
        or []
    )
    return pd.DataFrame(data)

@st.cache_data(ttl=30)
def fetch_lineups() -> pd.DataFrame:
    data = sb_public.table("lineups").select("*").execute().data or []
    return pd.DataFrame(data)

@st.cache_data(ttl=30)
def fetch_awards() -> pd.DataFrame:
    data = sb_public.table("awards").select("*").execute().data or []
    return pd.DataFrame(data)

def clear_caches():
    fetch_players.clear()
    fetch_matches.clear()
    fetch_lineups.clear()
    fetch_awards.clear()

# ---------------------------
# Helpers
# ---------------------------
def formation_to_lines(form: str):
    """Return counts per outfield line. We assume GK line first (1)."""
    if not form:
        return [1, 2, 1]  # default 5s
    parts = [int(x) for x in str(form).split("-") if x.strip().isdigit()]
    if not parts:
        return [1, 2, 1]
    return [1] + parts  # prepend GK line

def chip(name, goals, assists, photo):
    img = f'<img src="{photo}" alt=""/>' if (photo and str(photo).strip()) else ""
    g = int(goals or 0)
    a = int(assists or 0)
    stats = " · ".join(
        s for s in [f"⚽ {g}" if g else "", f"🅰️ {a}" if a else ""] if s
    )
    if stats:
        return f'<span class="chip">{img}<span>{name}</span> <span class="small">{stats}</span></span>'
    else:
        return f'<span class="chip">{img}<span>{name}</span></span>'


def public_image_url(path: str) -> str:
    try:
        url = sb_public.storage.from_(AVATAR_BUCKET).get_public_url(path)
        if isinstance(url, str):
            return url
        if hasattr(url, "get"):
            return url.get("publicUrl") or url.get("public_url") or ""
    except Exception:
        pass
    return f"{SUPABASE_URL}/storage/v1/object/public/{AVATAR_BUCKET}/{path}"

def upload_avatar(file, name):
    s = service()
    if not s:
        st.error("Admin required.")
        return None
    img = Image.open(file).convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    key = f"{name.lower().replace(' ', '_')}.png"
    try:
        s.storage.from_(AVATAR_BUCKET).upload(
            key, buf.getvalue(), {"content-type": "image/png", "upsert": "true"}
        )
    except Exception:
        try:
            s.storage.from_(AVATAR_BUCKET).remove([key])
        except Exception:
            pass
        s.storage.from_(AVATAR_BUCKET).upload(
            key, buf.getvalue(), {"content-type": "image/png"}
        )
    return public_image_url(key)

# ---------------------------
# Data shaping
# ---------------------------
def build_fact(players, matches, lineups):
    """
    Returns (lfact, matches):
    - lfact: lineups joined to matches + players with derived fields
    Ensures numeric types and a reliable 'name' column.
    """
    players = players.copy()
    matches = matches.copy()
    lineups = lineups.copy()

    for df in (players, matches, lineups):
        if "id" in df.columns:
            df["id"] = df["id"].astype(str)

    use_cols = [
        "id",
        "team_a",
        "team_b",
        "score_a",
        "score_b",
        "season",
        "gw",
        "date",
        "is_draw",
        "motm_name",
        "formation_a",
        "formation_b",
    ]
    mi = matches[[c for c in use_cols if c in matches.columns]].rename(
        columns={"id": "match_id"}
    )

    l = lineups.merge(mi, on="match_id", how="left")

    # Coerce numeric
    l["goals"] = pd.to_numeric(l.get("goals"), errors="coerce").fillna(0).astype(int)
    l["assists"] = pd.to_numeric(l.get("assists"), errors="coerce").fillna(0).astype(int)
    l["season"] = pd.to_numeric(l.get("season"), errors="coerce").astype("Int64")
    l["gw"] = pd.to_numeric(l.get("gw"), errors="coerce").astype("Int64")

    # Side and team/opp goals
    l["side"] = np.where(
        l["team"] == l["team_a"], "A", np.where(l["team"] == l["team_b"], "B", None)
    )
    l["team_goals"] = np.where(
        l["side"] == "A",
        l["score_a"],
        np.where(l["side"] == "B", l["score_b"], np.nan),
    )
    l["opp_goals"] = np.where(
        l["side"] == "A",
        l["score_b"],
        np.where(l["side"] == "B", l["score_a"], np.nan),
    )
    l["team_goals"] = pd.to_numeric(l["team_goals"], errors="coerce")
    l["opp_goals"] = pd.to_numeric(l["opp_goals"], errors="coerce")

    # Result and GA
    l["result"] = np.where(
        l["is_draw"] == True,
        "D",
        np.where(l["team_goals"] > l["opp_goals"], "W", "L"),
    )
    l["ga"] = (l["goals"] + l["assists"]).astype(int)

    # Attach player info
    p = players.rename(columns={"id": "player_id"})
    cols = [c for c in ["player_id", "name", "photo_url"] if c in p.columns]
    if cols:
        l = l.merge(p[cols], on="player_id", how="left")
    l["name"] = l["player_name"].where(
        l["player_name"].notna() & (l["player_name"].astype(str).str.strip() != ""),
        l.get("name"),
    )
    l["name"] = l["name"].fillna("Unknown")
    l["photo"] = l.get("photo_url")

    return l, matches

def player_agg(l, season=None, min_games=0, last_gw=None):
    df = l.copy()

    if season is not None:
        df = df[df["season"] == season]

    if last_gw:
        max_gw = pd.to_numeric(df["gw"], errors="coerce").max()
        if pd.notna(max_gw):
            df = df[pd.to_numeric(df["gw"], errors="coerce") >= (int(max_gw) - int(last_gw) + 1)]

    if df.empty:
        return pd.DataFrame(
            columns=[
                "name",
                "gp",
                "w",
                "d",
                "l",
                "win_pct",
                "goals",
                "assists",
                "ga",
                "ga_pg",
                "team_contrib_pct",
                "photo",
            ]
        )

    df["goals"] = pd.to_numeric(df["goals"], errors="coerce").fillna(0).astype(int)
    df["assists"] = pd.to_numeric(df["assists"], errors="coerce").fillna(0).astype(int)
    df["team_goals"] = pd.to_numeric(df["team_goals"], errors="coerce")

    gp = df.groupby("name").size().rename("gp")
    w = df[df["result"] == "W"].groupby("name").size().reindex(gp.index, fill_value=0)
    d = df[df["result"] == "D"].groupby("name").size().reindex(gp.index, fill_value=0)
    lcnt = df[df["result"] == "L"].groupby("name").size().reindex(gp.index, fill_value=0)

    goals = df.groupby("name")["goals"].sum().reindex(gp.index, fill_value=0)
    assists = df.groupby("name")["assists"].sum().reindex(gp.index, fill_value=0)
    ga = (goals + assists).astype(float)

    ga_pg = (ga / gp.replace(0, np.nan)).round(2).fillna(0)
    team_goals_sum = df.groupby("name")["team_goals"].sum(min_count=1).reindex(gp.index)
    denom = team_goals_sum.replace(0, np.nan)
    team_contrib = ((ga / denom) * 100).round(1).fillna(0)
    win_pct = ((w.values / np.maximum(gp.values, 1)) * 100).round(1)
    photo = df.groupby("name")["photo"].last().reindex(gp.index)

    out = pd.DataFrame(
        {
            "name": gp.index,
            "gp": gp.values,
            "w": w.values,
            "d": d.values,
            "l": lcnt.values,
            "win_pct": win_pct,
            "goals": goals.values.astype(int),
            "assists": assists.values.astype(int),
            "ga": (goals.values + assists.values).astype(int),
            "ga_pg": ga_pg.values,
            "team_contrib_pct": team_contrib.values,
            "photo": photo.values,
        }
    )

    if min_games > 0:
        out = out[out["gp"] >= min_games]

    return out.sort_values(["ga", "goals", "assists"], ascending=False).reset_index(drop=True)

# ---------------------------
# Pitch rendering (with fallback placement)
# ---------------------------
def auto_assign_positions(rows: pd.DataFrame, formation: str) -> pd.DataFrame:
    """
    If 'line'/'slot' are missing for most players, spread them sensibly:
    - GK (is_gk=True) goes to line 0 (slot center)
    - Others are distributed across remaining lines left-to-right
    Does not persist to DB; only for display.
    """
    lines = formation_to_lines(formation)
    max_slots = max(lines)
    rows = rows.copy()

    have_positions = rows["line"].notna().sum() > 0 and rows["slot"].notna().sum() > 0
    if have_positions:
        return rows

    rows["line"] = np.nan
    rows["slot"] = np.nan

    # GK first
    gk = rows[rows.get("is_gk", False) == True]
    if not gk.empty:
        idx = gk.index[0]
        rows.loc[idx, "line"] = 0
        rows.loc[idx, "slot"] = max_slots // 2

    # Others
    others = rows[rows.index != (gk.index[0] if not gk.empty else -1)]
    cur_line = 1  # start after GK line
    slots_filled = [0] * len(lines)
    for idx, _r in others.iterrows():
        if cur_line >= len(lines):
            cur_line = 1
        slots_here = lines[cur_line]
        # center alignment range
        offset = (max_slots - slots_here) // 2
        pos_in_line = slots_filled[cur_line] % slots_here
        rows.loc[idx, "line"] = cur_line
        rows.loc[idx, "slot"] = offset + pos_in_line
        slots_filled[cur_line] += 1
        cur_line += 1

    return rows

def draw_pitch(formation, rows, show_photos=True, key_prefix=""):
    lines = formation_to_lines(formation)
    max_slots = max(lines)
    rows = auto_assign_positions(rows, formation)

    html_lines = []
    for i, slots in enumerate(lines):
        grid_tpl = " ".join(["1fr"] * max_slots)
        placed = {
            int(r["slot"]): r
            for _, r in rows[rows["line"] == i]
            .dropna(subset=["slot"])
            .sort_values("slot")
            .iterrows()
        }
        items = []
        offset = (max_slots - slots) // 2
        for s in range(slots):
            gs = s + offset
            r = placed.get(gs)
            if r is not None:
                items.append(
                    f'<div class="slot">{chip(r.get("name",""), int(r.get("goals") or 0), int(r.get("assists") or 0), r.get("photo") if show_photos else None)}</div>'
                )
            else:
                items.append('<div class="slot"></div>')
        html_lines.append(
            f'<div class="line" style="grid-template-columns:{grid_tpl}">{"".join(items)}</div>'
        )
    st.markdown(f'<div class="pitch"><div class="grid">{"".join(html_lines)}</div></div>', unsafe_allow_html=True)

# ---------------------------
# Admin update helpers
# ---------------------------
def update_match_formations(match_id: str, formation_a: str, formation_b: str):
    s = service()
    if not s:
        st.error("Admin required.")
        return
    s.table("matches").update(
        {"formation_a": formation_a.strip() or None, "formation_b": formation_b.strip() or None}
    ).eq("id", match_id).execute()
    clear_caches()
    st.success("Formations updated.")

def update_lineup_positions(rows: pd.DataFrame):
    """Update line/slot/is_gk for given lineup IDs."""
    s = service()
    if not s:
        st.error("Admin required.")
        return
    payload = []
    for _, r in rows.iterrows():
        rid = str(r["id"])
        ln = r.get("line")
        sl = r.get("slot")
        igk = r.get("is_gk")
        payload.append(
            {
                "id": rid,
                "line": int(ln) if pd.notna(ln) and str(ln).strip() != "" else None,
                "slot": int(sl) if pd.notna(sl) and str(sl).strip() != "" else None,
                "is_gk": bool(igk) if not pd.isna(igk) else False,
            }
        )
    if payload:
        s.table("lineups").upsert(payload, on_conflict="id").execute()
        clear_caches()
        st.success("Positions saved.")

# ---------------------------
# Pages
# ---------------------------
def header():
    c1, c2 = st.columns([1, 1])
    with c1:
        st.title("⚽ Powerleague Stats")
    with c2:
        if "is_admin" not in st.session_state:
            st.session_state["is_admin"] = False
        if st.session_state["is_admin"]:
            st.success("Admin mode", icon="🔐")
            if st.button("Logout", key="logout_admin"):
                st.session_state["is_admin"] = False
                st.rerun()
        else:
            with st.popover("Admin login"):
                pw = st.text_input("Password", type="password", key="admin_pw")
                if st.button("Login", key="admin_login_btn"):
                    if pw == ADMIN_PASSWORD:
                        st.session_state["is_admin"] = True
                        st.rerun()
                    else:
                        st.error("Invalid password")

def page_matches():
    players = fetch_players()
    matches = fetch_matches()
    lineups = fetch_lineups()
    lfact, _ = build_fact(players, matches, lineups)

    st.subheader("Matches")
    if matches.empty:
        st.info("No matches yet.")
        return

    for _, m in matches.sort_values(["season", "gw"]).iterrows():
        show_photos = st.toggle("Show photos", True, key=f"show_photos_{m['id']}")
        st.markdown(
            f'<div class="banner"><div><div><strong>Season {m["season"]} · GW {m["gw"]}</strong></div><div class="small">{m.get("date") or ""}</div></div><div><strong>{m["team_a"]} {m["score_a"]} – {m["score_b"]} {m["team_b"]}</strong></div></div>',
            unsafe_allow_html=True,
        )
        if m.get("motm_name"):
            st.markdown(
                f'<div class="banner"><span>Man of the Match</span><span class="badge">🏅 {m["motm_name"]}</span></div>',
                unsafe_allow_html=True,
            )

        g = lfact[lfact["match_id"] == str(m["id"])]
        a_rows = g[g["team"] == m["team_a"]].copy()
        b_rows = g[g["team"] == m["team_b"]].copy()

        c1, c2 = st.columns(2)
        with c1:
            st.caption(m["team_a"])
            draw_pitch(m.get("formation_a") or "1-2-1", a_rows, show_photos, key_prefix=f"a_{m['id']}")
        with c2:
            st.caption(m["team_b"])
            draw_pitch(m.get("formation_b") or "1-2-1", b_rows, show_photos, key_prefix=f"b_{m['id']}")

        if st.session_state.get("is_admin"):
            with st.expander("✏️ Edit match (formation & positions)"):
                # Formation presets
                five_presets = ["1-2-1", "1-3", "2-2", "3-1"]
                seven_presets = ["2-1-2-1", "3-2-1", "2-3-1"]
                sc = int(m.get("side_count") or 5)
                presets = five_presets if sc == 5 else seven_presets

                colf1, colf2, colf3 = st.columns([2, 2, 1])
                with colf1:
                    fa = st.selectbox("Formation (Non-bibs)", presets, index=presets.index(m.get("formation_a") or presets[0]) if (m.get("formation_a") in presets) else 0, key=f"fa_{m['id']}")
                with colf2:
                    fb = st.selectbox("Formation (Bibs)", presets, index=presets.index(m.get("formation_b") or presets[0]) if (m.get("formation_b") in presets) else 0, key=f"fb_{m['id']}")
                with colf3:
                    if st.button("Save formations", key=f"save_form_{m['id']}"):
                        update_match_formations(str(m["id"]), fa, fb)
                        st.rerun()

                st.markdown("**Positions (line / slot / GK)**")
                # Editable tables per team
                def editor(df, label, key):
                    if df.empty:
                        st.caption(f"No {label} players.")
                        return df
                    cols = ["id", "name", "goals", "assists", "is_gk", "line", "slot"]
                    dfv = df.copy()[[c for c in cols if c in df.columns]]
                    dfv["line"] = pd.to_numeric(dfv.get("line"), errors="coerce")
                    dfv["slot"] = pd.to_numeric(dfv.get("slot"), errors="coerce")
                    edit = st.data_editor(
                        dfv,
                        hide_index=True,
                        disabled=["id", "name", "goals", "assists"],
                        column_config={
                            "is_gk": st.column_config.CheckboxColumn("GK"),
                            "line": st.column_config.NumberColumn("Line", step=1, min_value=0),
                            "slot": st.column_config.NumberColumn("Slot", step=1, min_value=0),
                        },
                        key=key,
                    )
                    return edit

                ea = editor(a_rows, m["team_a"], key=f"ea_{m['id']}")
                eb = editor(b_rows, m["team_b"], key=f"eb_{m['id']}")
                colb1, colb2 = st.columns(2)
                with colb1:
                    if st.button("💾 Save Non-bibs positions", key=f"save_a_{m['id']}"):
                        update_lineup_positions(ea)
                        st.rerun()
                with colb2:
                    if st.button("💾 Save Bibs positions", key=f"save_b_{m['id']}"):
                        update_lineup_positions(eb)
                        st.rerun()

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

def page_players():
    players = fetch_players()
    matches = fetch_matches()
    lineups = fetch_lineups()
    lfact, _ = build_fact(players, matches, lineups)
    st.subheader("Players")

    names = sorted(players["name"].dropna().astype(str).unique().tolist())
    sel = st.selectbox("Select player", [None] + names, index=0)
    if not sel:
        st.info("Choose a player to view profile.")
        return

    # Player row
    prow = players[players["name"] == sel]
    if prow.empty:
        st.warning("Player not found in players table (but might exist in lineups text).")
        p = {"id": None, "name": sel, "photo_url": None, "notes": None}
    else:
        p = prow.iloc[0].to_dict()

    mine = lfact[lfact["name"] == sel].copy()
    agg_all = player_agg(lfact)
    me = agg_all[agg_all["name"] == sel]
    me_row = me.iloc[0] if not me.empty else None

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(p.get("photo_url") or "https://placehold.co/200x200?text=No+Photo", width=140)
        st.markdown(f"### {sel}")
        if p.get("notes"):
            st.caption(p["notes"])
        if st.session_state.get("is_admin") and p.get("id"):
            up = st.file_uploader("Update photo (HEIC/JPG/PNG)", type=["heic", "HEIC", "jpg", "jpeg", "png"], key="upl_photo")
            if up and st.button("Upload", key="upload_photo_btn"):
                url = upload_avatar(up, sel)
                if url:
                    s = service()
                    s.table("players").update({"photo_url": url}).eq("id", p["id"]).execute()
                    clear_caches()
                    st.success("Photo updated.")
                    st.rerun()
    with col2:
        if me_row is not None:
            st.markdown(
                f"**Career** — GP: {int(me_row['gp'])} · W-D-L: {int(me_row['w'])}-{int(me_row['d'])}-{int(me_row['l'])} · Win%: {me_row['win_pct']}%"
            )
            st.markdown(
                f"**Goals**: {int(me_row['goals'])} · **Assists**: {int(me_row['assists'])} · **G+A**: {int(me_row['ga'])} · **G+A/PG**: {me_row['ga_pg']} · **Team Contribution**: {me_row['team_contrib_pct']}%"
            )

        st.markdown("#### Recent Games")
        if mine.empty:
            st.caption("No games.")
        else:
            # Ensure season/gw exist & numeric to avoid KeyError
            mine["season"] = pd.to_numeric(mine.get("season"), errors="coerce")
            mine["gw"] = pd.to_numeric(mine.get("gw"), errors="coerce")
            for _, r in mine.sort_values(["season", "gw"], ascending=[False, False]).head(10).iterrows():
                st.write(f"S{int(r['season'])} GW{int(r['gw'])} · {r['team_a']} {r['score_a']}–{r['score_b']} {r['team_b']} — {r['team']}")
                st.caption(f"⚽ {int(r.get('goals') or 0)} · 🅰️ {int(r.get('assists') or 0)} · Result: {r['result']}")

        st.markdown("#### Awards")
        aw = fetch_awards()
        mine_aw = aw[(aw["player_name"] == sel) | ((p.get("id") is not None) & (aw["player_id"] == p["id"]))]
        if mine_aw.empty:
            st.caption("No awards yet.")
        else:
            for _, a in mine_aw.sort_values(["season", "month", "gw"]).iterrows():
                mo = f" · Month {int(a['month'])}" if pd.notna(a["month"]) else ""
                gw = f" · GW {int(a['gw'])}" if pd.notna(a["gw"]) else ""
                st.write(f"🏅 {a['type']} — Season {a['season']}{mo}{gw}")

def page_stats():
    players = fetch_players()
    matches = fetch_matches()
    lineups = fetch_lineups()
    lfact, _ = build_fact(players, matches, lineups)
    st.subheader("Stats & Leaderboards")

    c1, c2, c3, c4 = st.columns(4)
    seasons = sorted(matches["season"].dropna().unique().tolist()) if not matches.empty else []
    season = c1.selectbox("Season", [None] + seasons, index=0)
    min_games = int(c2.number_input("Min games", min_value=0, value=0, step=1))
    last_x = c3.selectbox("Last X GWs", [None, 3, 5, 10], index=0)
    top_n = int(c4.number_input("Rows (Top N)", min_value=5, value=10, step=1))

    metric = st.selectbox(
        "Metric",
        ["G+A", "Goals", "Assists", "G+A per Game", "Win %", "Team Contribution %", "MOTM"],
        index=0,
    )
    show_photos = st.toggle("Show photos", True)

    agg = player_agg(lfact, season=season, min_games=min_games, last_gw=last_x)

    # MOTM count (from awards)
    motm = fetch_awards()
    if metric == "MOTM":
        motm_f = motm[motm["type"] == "MOTM"].copy()
        if season is not None:
            motm_f = motm_f[motm_f["season"] == season]
        motm_count = (
            motm_f.groupby("player_name")
            .size()
            .reset_index(name="MOTM")
            .sort_values("MOTM", ascending=False)
        )
        df = motm_count.rename(columns={"player_name": "name"})
        # attach photos if possible
        photos = players[["name", "photo_url"]]
        df = df.merge(photos, on="name", how="left")
    else:
        df = agg.copy()
        if metric == "Goals":
            df = df.sort_values(["goals", "ga", "assists"], ascending=False)
        elif metric == "Assists":
            df = df.sort_values(["assists", "ga", "goals"], ascending=False)
        elif metric == "G+A":
            df = df.sort_values(["ga", "goals", "assists"], ascending=False)
        elif metric == "G+A per Game":
            df = df.sort_values(["ga_pg", "ga", "goals"], ascending=False)
        elif metric == "Win %":
            df = df.sort_values(["win_pct", "ga", "goals"], ascending=False)
        elif metric == "Team Contribution %":
            df = df.sort_values(["team_contrib_pct", "ga", "goals"], ascending=False)

    def board(df, title_col, value_col, title):
        st.markdown(f"### {title}")
        if df.empty:
            st.caption("No data.")
            return
        for r in df.head(top_n).to_dict(orient="records"):
            a, b = st.columns([4, 1])
            photo = (r.get("photo") or r.get("photo_url")) if show_photos else None
            with a:
                st.markdown(chip(r.get(title_col, "Unknown"), r.get("goals", 0), r.get("assists", 0), photo), unsafe_allow_html=True)
            with b:
                val = r.get(value_col)
                if isinstance(val, float):
                    val = round(val, 2)
                st.metric(value_col.replace("_", " ").title(), val)

    if metric == "MOTM":
        board(df, "name", "MOTM", "Man of the Match (count)")
    elif metric == "Goals":
        board(df, "name", "goals", "Top Scorers")
    elif metric == "Assists":
        board(df, "name", "assists", "Top Assisters")
    elif metric == "G+A per Game":
        board(df, "name", "ga_pg", "G+A per Game")
    elif metric == "Win %":
        board(df, "name", "win_pct", "Win %")
    elif metric == "Team Contribution %":
        board(df, "name", "team_contrib_pct", "Team Contribution %")
    else:
        board(df, "name", "ga", "G + A")

def page_awards():
    aw = fetch_awards()
    st.subheader("Awards")
    if st.session_state.get("is_admin"):
        with st.form("add_award"):
            season = st.number_input("Season", min_value=1, value=1, step=1)
            month = st.number_input("Month (1-12, for POTM)", min_value=1, max_value=12, value=1, step=1)
            atype = st.selectbox("Type", ["MOTM", "POTM"])
            gw = st.number_input("GW (for MOTM)", min_value=1, value=1, step=1)
            player_name = st.text_input("Player name")
            notes = st.text_input("Notes")
            if st.form_submit_button("Add"):
                s = service()
                s.table("awards").insert(
                    {
                        "season": int(season),
                        "month": int(month) if atype == "POTM" else None,
                        "type": atype,
                        "gw": int(gw) if atype == "MOTM" else None,
                        "player_name": player_name,
                        "notes": notes,
                    }
                ).execute()
                clear_caches()
                st.success("Saved.")
                st.rerun()
    st.markdown("#### Player of the Month")
    potm = aw[aw["type"] == "POTM"]
    if potm.empty:
        st.caption("No POTM yet.")
    else:
        for _, r in potm.sort_values(["season", "month"]).iterrows():
            st.write(f"🏆 S{r['season']} · M{int(r['month'])}: {r['player_name']}")
    st.markdown("#### Man of the Match (History)")
    motm = aw[aw["type"] == "MOTM"]
    if motm.empty:
        st.caption("No MOTM yet.")
    else:
        for _, r in motm.sort_values(["season", "gw"]).iterrows():
            st.write(f"🎖️ S{r['season']} GW{r['gw']}: {r['player_name']}")

def page_admin():
    st.subheader("Admin Tools")
    st.write("Historical import is **done directly in Supabase** (see `import.sql`).")
    st.download_button(
        "Export players.csv",
        fetch_players().to_csv(index=False).encode("utf-8"),
        file_name="players.csv",
        mime="text/csv",
    )
    st.download_button(
        "Export matches.csv",
        fetch_matches().to_csv(index=False).encode("utf-8"),
        file_name="matches.csv",
        mime="text/csv",
    )
    st.download_button(
        "Export lineups.csv",
        fetch_lineups().to_csv(index=False).encode("utf-8"),
        file_name="lineups.csv",
        mime="text/csv",
    )
    st.button("Force refresh caches", on_click=clear_caches)

# ---------------------------
# Router
# ---------------------------
def run_app():
    header()
    Page = getattr(st, "Page", None)
    nav = getattr(st, "navigation", None)
    if Page and nav:
        sections = {
            "Main": [
                Page(page_matches, title="Matches", icon="📋"),
                Page(page_players, title="Players", icon="👤"),
                Page(page_stats, title="Stats", icon="📊"),
                Page(page_awards, title="Awards", icon="🏆"),
            ]
        }
        if st.session_state.get("is_admin"):
            sections["Admin"] = [Page(page_admin, title="Export & Cache", icon="🛠️")]
        n = nav(sections)
        n.run()
    else:
        sel = st.sidebar.radio(
            "Go to",
            ["Matches", "Players", "Stats", "Awards"]
            + (["Admin"] if st.session_state.get("is_admin") else []),
            index=0,
        )
        {"Matches": page_matches, "Players": page_players, "Stats": page_stats, "Awards": page_awards, "Admin": page_admin}.get(sel, page_matches)()

if __name__ == "__main__":
    run_app()
