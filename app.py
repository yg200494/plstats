import io
from datetime import date
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import pillow_heif
from supabase import create_client, Client

st.set_page_config(page_title="Powerleague Stats", page_icon="⚽", layout="wide", initial_sidebar_state="collapsed")

CSS = '''
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
</style>
'''
st.markdown(CSS, unsafe_allow_html=True)

# ========== Secrets / Clients ==========
REQ = ["SUPABASE_URL","SUPABASE_ANON_KEY","ADMIN_PASSWORD","AVATAR_BUCKET"]
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
        if not s: return
        buckets = s.storage.list_buckets()
        names = {getattr(b, "name", getattr(b, "id", None)) for b in buckets}
        if AVATAR_BUCKET not in names:
            s.storage.create_bucket(AVATAR_BUCKET, public=True)
    except Exception: pass
ensure_bucket()

try:
    pillow_heif.register_heif_opener()
except Exception:
    pass

# ========== Cached fetches ==========
@st.cache_data(ttl=30)
def fetch_players() -> pd.DataFrame:
    return pd.DataFrame(sb_public.table("players").select("*").execute().data or [])

@st.cache_data(ttl=30)
def fetch_matches() -> pd.DataFrame:
    return pd.DataFrame(sb_public.table("matches").select("*").order("season").order("gw").execute().data or [])

@st.cache_data(ttl=30)
def fetch_lineups() -> pd.DataFrame:
    return pd.DataFrame(sb_public.table("lineups").select("*").execute().data or [])

@st.cache_data(ttl=30)
def fetch_awards() -> pd.DataFrame:
    return pd.DataFrame(sb_public.table("awards").select("*").execute().data or [])

def clear_caches():
    fetch_players.clear(); fetch_matches.clear(); fetch_lineups.clear(); fetch_awards.clear()

# ========== Helpers ==========
def formation_to_lines(form: str):
    if not form: return [1,2,1]
    parts = [int(x) for x in str(form).split("-") if x.strip().isdigit()]
    return [1] + (parts if parts else [2,1])

def chip(name, goals, assists, photo):
    img = f'<img src="{photo}" alt=""/>' if photo else ""
    stats = " · ".join([s for s in [f"⚽ {int(goals)}" if goals else "", f"🅰️ {int(assists)}" if assists else ""] if s])
    return f'<span class="chip">{img}<span>{name}</span>{"<span class=\"small\">"+stats+"</span>" if stats else ""}</span>'

def public_image_url(path: str) -> str:
    try:
        url = sb_public.storage.from_(AVATAR_BUCKET).get_public_url(path)
        if isinstance(url, str): return url
        if hasattr(url, "get"):
            return url.get("publicUrl") or url.get("public_url") or ""
    except Exception: pass
    return f"{SUPABASE_URL}/storage/v1/object/public/{AVATAR_BUCKET}/{path}"

def upload_avatar(file, name):
    s = service()
    if not s: st.error("Admin required."); return None
    img = Image.open(file)
    buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
    key = f"{name.lower().replace(' ','_')}.png"
    try:
        s.storage.from_(AVATAR_BUCKET).upload(key, buf.getvalue(), {"content-type":"image/png","upsert":"true"})
    except Exception:
        try: s.storage.from_(AVATAR_BUCKET).remove([key])
        except Exception: pass
        s.storage.from_(AVATAR_BUCKET).upload(key, buf.getvalue(), {"content-type":"image/png"})
    return public_image_url(key)

def build_fact(players, matches, lineups):
    if matches.empty: return lineups.copy(), matches.copy()
    mi = matches[["id","team_a","team_b","score_a","score_b","season","gw","date","is_draw","motm_name"]].rename(columns={"id":"match_id"})
    l = lineups.merge(mi, on="match_id", how="left")
    l["side"] = np.where(l["team"]==l["team_a"], "A", np.where(l["team"]==l["team_b"], "B", None))
    l["team_goals"] = np.where(l["side"]=="A", l["score_a"], np.where(l["side"]=="B", l["score_b"], None))
    l["opp_goals"] = np.where(l["side"]=="A", l["score_b"], np.where(l["side"]=="B", l["score_a"], None))
    l["result"] = np.where(l["is_draw"], "D", np.where(l["team_goals"]>l["opp_goals"], "W", "L"))
    l["ga"] = (l["goals"].fillna(0) + l["assists"].fillna(0)).astype(int)
    players = players.rename(columns={"id":"player_id"})
    l = l.merge(players[["player_id","name","photo_url"]], on="player_id", how="left")
    l["name"] = l["player_name"].combine_first(l["name"])
    l["photo"] = l["photo_url"]
    return l, matches

def player_agg(l, season=None, min_games=0, last_gw=None):
    df = l.copy()
    if season: df = df[df["season"]==season]
    if last_gw: df = df[df["gw"] >= (df["gw"].max() - last_gw + 1)]
    if df.empty:
        return pd.DataFrame(columns=["name","gp","w","d","l","win_pct","goals","assists","ga","ga_pg","team_contrib_pct","photo"])
    gp = df.groupby("name").size().rename("gp")
    w = df[df["result"]=="W"].groupby("name").size().reindex(gp.index, fill_value=0)
    d = df[df["result"]=="D"].groupby("name").size().reindex(gp.index, fill_value=0)
    lcnt = df[df["result"]=="L"].groupby("name").size().reindex(gp.index, fill_value=0)
    goals = df.groupby("name")["goals"].sum().reindex(gp.index, fill_value=0)
    assists = df.groupby("name")["assists"].sum().reindex(gp.index, fill_value=0)
    ga = goals + assists
    ga_pg = (ga / gp).round(2)
    team_goals = df.groupby("name")["team_goals"].sum().replace(0, np.nan).reindex(gp.index)
    contrib = ((ga / team_goals) * 100).round(1).fillna(0)
    photo = df.groupby("name")["photo"].last().reindex(gp.index)
    out = pd.DataFrame({
        "name": gp.index, "gp": gp.values, "w": w.values, "d": d.values, "l": lcnt.values,
        "win_pct": ((w.values/np.maximum(gp.values,1))*100).round(1),
        "goals": goals.values, "assists": assists.values, "ga": ga.values, "ga_pg": ga_pg.values,
        "team_contrib_pct": contrib.values, "photo": photo.values
    }).sort_values(["ga","goals","assists"], ascending=False)
    if min_games>0: out = out[out["gp"]>=min_games]
    return out

# ========== UI ==========
def header():
    c1, c2 = st.columns([1,1])
    with c1: st.title("⚽ Powerleague Stats")
    with c2:
        if "is_admin" not in st.session_state: st.session_state["is_admin"] = False
        if st.session_state["is_admin"]:
            st.success("Admin mode", icon="🔐")
            if st.button("Logout"): st.session_state["is_admin"] = False; st.rerun()
        else:
            with st.popover("Admin login"):
                pw = st.text_input("Password", type="password")
                if st.button("Login"):
                    if pw == ADMIN_PASSWORD: st.session_state["is_admin"] = True; st.rerun()
                    else: st.error("Invalid password")

def draw_pitch(formation, rows, show_photos=True):
    lines = formation_to_lines(formation)
    max_slots = max(lines)
    html_lines = []
    for i, slots in enumerate(lines):
        grid_tpl = " ".join(["1fr"]*max_slots)
        placed = {int(r["slot"]) : r for _,r in rows[rows["line"]==i].dropna(subset=["slot"]).sort_values("slot").iterrows()}
        items = []
        offset = (max_slots - slots)//2
        for s in range(slots):
            gs = s + offset
            r = placed.get(gs)
            if r is not None:
                items.append(f'<div class="slot">{chip(r.get("name",""), int(r.get("goals") or 0), int(r.get("assists") or 0), r.get("photo") if show_photos else None)}</div>')
            else:
                items.append('<div class="slot"></div>')
        html_lines.append(f'<div class="line" style="grid-template-columns:{grid_tpl}">{"".join(items)}</div>')
    st.markdown(f'<div class="pitch"><div class="grid">{"".join(html_lines)}</div></div>', unsafe_allow_html=True)

def page_matches():
    players = fetch_players(); matches = fetch_matches(); lineups = fetch_lineups()
    lfact, _ = build_fact(players, matches, lineups)
    st.subheader("Matches")
    if matches.empty: st.info("No matches yet."); return
    for _, m in matches.sort_values(["season","gw"]).iterrows():
        left, right = st.columns([3,2])
        with left:
            st.markdown(f'<div class="banner"><div><div><strong>Season {m["season"]} · GW {m["gw"]}</strong></div><div class="small">{m.get("date") or ""}</div></div><div><strong>{m["team_a"]} {m["score_a"]} – {m["score_b"]} {m["team_b"]}</strong></div></div>', unsafe_allow_html=True)
        with right:
            if m.get("motm_name"):
                st.markdown(f'<div class="banner"><span>Man of the Match</span><span class="badge">🏅 {m["motm_name"]}</span></div>', unsafe_allow_html=True)
        g = lfact[lfact["match_id"]==m["id"]]
        a_rows = g[g["team"]==m["team_a"]]; b_rows = g[g["team"]==m["team_b"]]
        show = st.toggle("Show photos", True, key=f"photos_{m['id']}")
        c1, c2 = st.columns(2)
        with c1: st.caption(m["team_a"]); draw_pitch(m.get("formation_a") or "1-2-1", a_rows, show)
        with c2: st.caption(m["team_b"]); draw_pitch(m.get("formation_b") or "1-2-1", b_rows, show)
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

def page_players():
    players = fetch_players(); matches = fetch_matches(); lineups = fetch_lineups()
    lfact, _ = build_fact(players, matches, lineups)
    st.subheader("Players")
    names = players["name"].sort_values().tolist()
    sel = st.selectbox("Select player", [None]+names, index=0)
    if not sel: st.info("Choose a player to view profile."); return
    p = players[players["name"]==sel].iloc[0].to_dict()
    mine = lfact[lfact["name"]==sel]
    agg = player_agg(lfact); me = agg[agg["name"]==sel].iloc[0] if not agg[agg["name"]==sel].empty else None
    col1, col2 = st.columns([1,2])
    with col1:
        st.image(p.get("photo_url") or "https://placehold.co/200x200?text=No+Photo", width=140)
        st.markdown(f"### {p['name']}")
        if p.get("notes"): st.caption(p["notes"])
        if st.session_state.get("is_admin"):
            up = st.file_uploader("Update photo (HEIC/JPG/PNG)", type=["heic","HEIC","jpg","jpeg","png"])
            if up and st.button("Upload"):
                url = upload_avatar(up, p["name"])
                if url:
                    s = service(); s.table("players").update({"photo_url": url}).eq("id", p["id"]).execute()
                    clear_caches(); st.success("Photo updated."); st.rerun()
    with col2:
        if me is not None:
            st.markdown(f"**Career** — GP: {int(me['gp'])} · W-D-L: {int(me['w'])}-{int(me['d'])}-{int(me['l'])} · Win%: {me['win_pct']}%")
            st.markdown(f"**Goals**: {int(me['goals'])} · **Assists**: {int(me['assists'])} · **G+A**: {int(me['ga'])} · **G+A/PG**: {me['ga_pg']} · **Team Contribution**: {me['team_contrib_pct']}%")
        st.markdown("#### Recent Games")
        if mine.empty: st.caption("No games."); 
        else:
            for _, r in mine.sort_values(["season","gw"], ascending=[False, False]).head(10).iterrows():
                st.write(f"S{r['season']} GW{r['gw']} · {r['team_a']} {r['score_a']}–{r['score_b']} {r['team_b']} — {r['team']}")
                st.caption(f"⚽ {int(r.get('goals') or 0)} · 🅰️ {int(r.get('assists') or 0)} · Result: {r['result']}")
        st.markdown("#### Awards")
        aw = fetch_awards(); mine_aw = aw[(aw["player_name"]==sel) | (aw["player_id"]==p["id"])]
        if mine_aw.empty: st.caption("No awards yet.")
        else:
            for _, a in mine_aw.sort_values(["season","month","gw"]).iterrows():
                st.write(f"🏅 {a['type']} — Season {a['season']} · {('Month '+str(int(a['month'])) if pd.notna(a['month']) else '')} · GW {a.get('gw') or ''}")

def page_stats():
    players = fetch_players(); matches = fetch_matches(); lineups = fetch_lineups()
    lfact, _ = build_fact(players, matches, lineups)
    st.subheader("Stats & Leaderboards")
    c1, c2, c3 = st.columns(3)
    season = c1.selectbox("Season", [None]+sorted(matches["season"].dropna().unique().tolist()), index=0)
    min_games = c2.number_input("Min games", min_value=0, value=0, step=1)
    last_x = c3.selectbox("Last X GWs", [None,3,5,10], index=0)
    show_photos = st.toggle("Show photos", True)
    agg = player_agg(lfact, season=season, min_games=min_games, last_gw=last_x)
    def board(df, col, title, n=10):
        st.markdown(f"### {title}")
        if df.empty: st.caption("No data."); return
        for r in df.sort_values([col,"ga","goals"], ascending=False).head(n).to_dict(orient="records"):
            a,b = st.columns([4,1])
            with a: st.markdown(chip(r['name'], r.get("goals",0), r.get("assists",0), r.get("photo") if show_photos else None), unsafe_allow_html=True)
            with b: st.metric(col.replace("_"," ").title(), r[col])
    board(agg, "goals", "Top Scorers")
    board(agg, "assists", "Top Assisters")
    board(agg, "ga", "G + A")
    board(agg.sort_values("team_contrib_pct", ascending=False), "team_contrib_pct", "Team Contribution %")

def page_awards():
    aw = fetch_awards()
    st.subheader("Awards")
    if st.session_state.get("is_admin"):
        with st.form("add_award"):
            season = st.number_input("Season", min_value=1, value=1, step=1)
            month = st.number_input("Month (1-12, for POTM)", min_value=1, max_value=12, value=1, step=1)
            atype = st.selectbox("Type", ["MOTM","POTM"])
            gw = st.number_input("GW (for MOTM)", min_value=1, value=1, step=1)
            player_name = st.text_input("Player name")
            notes = st.text_input("Notes")
            if st.form_submit_button("Add"):
                s = service()
                s.table("awards").insert({
                    "season": int(season),
                    "month": int(month) if atype=="POTM" else None,
                    "type": atype,
                    "gw": int(gw) if atype=="MOTM" else None,
                    "player_name": player_name,
                    "notes": notes
                }).execute()
                clear_caches(); st.success("Saved."); st.rerun()
    st.markdown("#### Player of the Month")
    potm = aw[aw["type"]=="POTM"]; 
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

def page_admin():
    st.subheader("Admin Tools")
    st.write("Historical import is **done directly in Supabase** now (see README & `import.sql`).")
    st.download_button("Export players.csv", fetch_players().to_csv(index=False).encode("utf-8"), file_name="players.csv", mime="text/csv")
    st.download_button("Export matches.csv", fetch_matches().to_csv(index=False).encode("utf-8"), file_name="matches.csv", mime="text/csv")
    st.download_button("Export lineups.csv", fetch_lineups().to_csv(index=False).encode("utf-8"), file_name="lineups.csv", mime="text/csv")
    st.button("Force refresh caches", on_click=clear_caches)

# Header + Navigation
header()
Page = getattr(st, "Page", None); nav = getattr(st, "navigation", None)
if Page and nav:
    sections = {"Main": [Page(page_matches, title="Matches", icon="📋"),
                         Page(page_players, title="Players", icon="👤"),
                         Page(page_stats, title="Stats", icon="📊"),
                         Page(page_awards, title="Awards", icon="🏆")]}
    if st.session_state.get("is_admin"):
        sections["Admin"] = [Page(page_admin, title="Export & Cache", icon="🛠️")]
    n = nav(sections); n.run()
else:
    sel = st.sidebar.radio("Go to", ["Matches","Players","Stats","Awards"] + (["Admin"] if st.session_state.get("is_admin") else []), index=0)
    {"Matches":page_matches,"Players":page_players,"Stats":page_stats,"Awards":page_awards,"Admin":page_admin}.get(sel, page_matches)()
