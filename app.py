import io
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import pillow_heif
from supabase import create_client, Client

# =========================
# App config & global style
# =========================
st.set_page_config(
    page_title="Powerleague Stats",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
:root{
  --bg1:#0b1324;
  --bg2:#0a1a2f;
  --card:rgba(15,23,42,.66);
  --card-br:rgba(255,255,255,.10);
  --chip-bg:rgba(255,255,255,.10);
  --chip-br:rgba(255,255,255,.30);
  --green1:#2fb25c; --green2:#1c7a3a; --green3:#0f5a2b;
}
html,body,[data-testid="stAppViewContainer"]{
  background: linear-gradient(180deg,var(--bg1) 0%, var(--bg2) 100%);
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
.banner .sub{opacity:.8;font-size:13px}
.badge{background:#22c55e;color:#062616;font-weight:700;border-radius:999px;padding:6px 12px;font-size:12px;border:1px solid rgba(0,0,0,.25)}
.hr{height:1px; background:rgba(255,255,255,.10); margin:10px 0 16px}

.pitch{
  background:radial-gradient(ellipse at 50% 8%, var(--green1) 0%, var(--green2) 60%, var(--green3) 100%);
  border-radius:22px; padding:18px; border:1px solid rgba(255,255,255,.16);
  box-shadow:0 10px 30px rgba(0,0,0,.35); min-height:520px;
}
.pitch .grid{display:grid;gap:22px}
.line{display:grid;gap:12px}
.slot{display:flex;align-items:center;justify-content:center}

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
.edit-slot-btn{
  width:100%; height:42px; border-radius:10px; border:1px dashed rgba(255,255,255,.25);
  background:rgba(255,255,255,.04);
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =========================
# Secrets & Supabase clients
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
# Helpers
# =========================
def formation_to_lines(form: str):
    if not form: return [1,2,1]   # default 5s
    parts = [int(x) for x in str(form).split("-") if x.strip().isdigit()]
    return [1] + (parts if parts else [2,1])

def chip(name, goals, assists, photo):
    img = f'<img src="{photo}" alt=""/>' if (photo and str(photo).strip()) else ""
    g = int(goals or 0); a = int(assists or 0)
    stats = " · ".join(s for s in [f"⚽ {g}" if g else "", f"🅰️ {a}" if a else ""] if s)
    if stats:
        return f'<span class="chip">{img}<span>{name}</span><span class="meta">{stats}</span></span>'
    return f'<span class="chip">{img}<span>{name}</span></span>'

def public_image_url(path: str) -> str:
    try:
        url = sb_public.storage.from_(AVATAR_BUCKET).get_public_url(path)
        if isinstance(url, str): return url
        if hasattr(url, "get"): return url.get("publicUrl") or url.get("public_url") or ""
    except Exception: pass
    return f"{SUPABASE_URL}/storage/v1/object/public/{AVATAR_BUCKET}/{path}"

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
    return public_image_url(key)

# =========================
# Data shaping
# =========================
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

    # numerics
    l["goals"]   = pd.to_numeric(l.get("goals"), errors="coerce").fillna(0).astype(int)
    l["assists"] = pd.to_numeric(l.get("assists"), errors="coerce").fillna(0).astype(int)

    # side & team goals
    l["side"] = np.where(l["team"]==l["team_a"], "A", np.where(l["team"]==l["team_b"], "B", None))
    l["team_goals"] = np.where(l["side"]=="A", l["score_a"], np.where(l["side"]=="B", l["score_b"], np.nan))
    l["opp_goals"]  = np.where(l["side"]=="A", l["score_b"], np.where(l["side"]=="B", l["score_a"], np.nan))
    l["team_goals"] = pd.to_numeric(l["team_goals"], errors="coerce")
    l["opp_goals"]  = pd.to_numeric(l["opp_goals"],  errors="coerce")

    # result & GA
    l["result"] = np.where(l["is_draw"]==True, "D", np.where(l["team_goals"]>l["opp_goals"], "W", "L"))
    l["ga"] = (l["goals"] + l["assists"]).astype(int)

    # attach player info (fallback to player_name)
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
        return pd.DataFrame(columns=["name","gp","w","d","l","win_pct","goals","assists","ga","ga_pg","team_contrib_pct","photo"])
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
    team_goals_sum = df.groupby("name")["team_goals"].sum(min_count=1).reindex(gp.index)
    denom = team_goals_sum.replace(0,np.nan)
    team_contrib = ((ga/denom)*100).round(1).fillna(0)
    win_pct = ((w.values/np.maximum(gp.values,1))*100).round(1)
    photo = df.groupby("name")["photo"].last().reindex(gp.index)

    out = pd.DataFrame({
        "name": gp.index, "gp": gp.values, "w": w.values, "d": d.values, "l": l_.values,
        "win_pct": win_pct,
        "goals": goals.values.astype(int), "assists": assists.values.astype(int),
        "ga": (goals.values+assists.values).astype(int), "ga_pg": ga_pg.values,
        "team_contrib_pct": team_contrib.values, "photo": photo.values
    }).sort_values(["ga","goals","assists"], ascending=False).reset_index(drop=True)
    if min_games>0: out = out[out["gp"]>=min_games]
    return out

# ---------- Pair tables (Duos / Nemesis) ----------
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
    a = a[a["name_a"] < a["name_b"]]  # unique pair
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
    w_b = a[a["result_b"]=="W"].groupby(["name_a","name_b"]).size().reindex(gp.index, fill_value=0)
    # win rate of the first named vs the second
    win_pct_a = ((w_a/gp)*100).round(1)
    out = pd.DataFrame({"gp":gp, "win_pct_vs":win_pct_a}).reset_index()
    out["pair"] = out["name_a"]+" vs "+out["name_b"]
    out = out[out["gp"]>=min_gp].sort_values(["win_pct_vs","gp"], ascending=[True,False])[["pair","gp","win_pct_vs"]]
    return out

# =========================
# Pitch display (read-only)
# =========================
def auto_assign_positions(rows: pd.DataFrame, formation: str) -> pd.DataFrame:
    lines = formation_to_lines(formation); max_slots = max(lines)
    rows = rows.copy()
    have = rows["line"].notna().sum()>0 and rows["slot"].notna().sum()>0
    if have: return rows
    rows["line"] = np.nan; rows["slot"] = np.nan
    gk = rows[rows.get("is_gk", False)==True]
    if not gk.empty:
        idx=gk.index[0]; rows.loc[idx,"line"]=0; rows.loc[idx,"slot"]=max_slots//2
    others = rows.drop(index=(gk.index[0] if not gk.empty else []), errors="ignore")
    cur=1; filled=[0]*len(lines)
    for idx,_ in others.iterrows():
        if cur>=len(lines): cur=1
        slots_here = lines[cur]; offset=(max_slots-slots_here)//2
        pos = filled[cur] % slots_here
        rows.loc[idx,"line"]=cur; rows.loc[idx,"slot"]=offset+pos
        filled[cur]+=1; cur+=1
    return rows

def draw_pitch(formation, rows, show_photos=True):
    lines = formation_to_lines(formation); max_slots = max(lines)
    rows = auto_assign_positions(rows, formation)
    html=[]
    for i, slots in enumerate(lines):
        grid_tpl = " ".join(["1fr"]*max_slots)
        placed = {int(r["slot"]):r for _,r in rows[rows["line"]==i].dropna(subset=["slot"]).sort_values("slot").iterrows()}
        items=[]; offset=(max_slots-slots)//2
        for s in range(slots):
            gs = s+offset; r=placed.get(gs)
            if r is not None:
                items.append(f'<div class="slot">{chip(r.get("name",""), r.get("goals",0), r.get("assists",0), r.get("photo") if show_photos else None)}</div>')
            else:
                items.append('<div class="slot"></div>')
        html.append(f'<div class="line" style="grid-template-columns:{grid_tpl}">{"".join(items)}</div>')
    st.markdown(f'<div class="pitch"><div class="grid">{"".join(html)}</div></div>', unsafe_allow_html=True)

# =========================
# Tap-to-place lineup editor
# =========================
def _formation_slots(formation: str):
    lines = formation_to_lines(formation)
    max_slots = max(lines)
    offsets = [(max_slots - s)//2 for s in lines]
    return lines, max_slots, offsets

def place_player(mid: str, team_rows: pd.DataFrame, formation: str, player_id: str, line_i: int, grid_slot: int):
    """Persist: clear any occupant of (line_i,grid_slot), set selected player's line/slot."""
    s = service()
    if not s: st.error("Admin required."); return
    # find occupant at this cell
    occ = team_rows[(team_rows["line"]==line_i) & (team_rows["slot"]==grid_slot)]
    payload=[]
    for _,r in occ.iterrows():
        payload.append({"id": str(r["id"]), "line": None, "slot": None})
    # set selected
    payload.append({"id": str(player_id), "line": int(line_i), "slot": int(grid_slot)})
    s.table("lineups").upsert(payload, on_conflict="id").execute()
    clear_caches()
    st.success("Position saved.")

def set_gk(mid: str, team_rows: pd.DataFrame, formation: str, player_id: str):
    """Set GK & center the keeper on back line."""
    s = service()
    if not s: st.error("Admin required."); return
    lines, max_slots, _ = _formation_slots(formation)
    center = max_slots//2
    # unset GK for others
    payload=[]
    for _,r in team_rows.iterrows():
        payload.append({"id": str(r["id"]), "is_gk": bool(r["id"]==player_id)})
    # also place on back line center
    payload.append({"id": str(player_id), "line": 0, "slot": center})
    s.table("lineups").upsert(payload, on_conflict="id").execute()
    clear_caches()
    st.success("GK updated.")

def pitch_editor(mid: str, team_name: str, formation: str, team_rows: pd.DataFrame, key_prefix: str):
    """UI: select player -> tap slot to place (saves to DB)."""
    team_rows = team_rows.copy()
    st.caption(f"{team_name} — tap a slot to place the selected player")
    # bench selector
    team_rows["display"] = team_rows.apply(lambda r: f"{r['name']} ({'GK' if r.get('is_gk') else 'Outfield'})", axis=1)
    bench_opts = team_rows["display"].tolist()
    bench_map = {team_rows.iloc[i]["display"]: str(team_rows.iloc[i]["id"]) for i in range(len(team_rows))}
    pick = st.selectbox("Player", bench_opts, key=f"{key_prefix}_pick")
    pid = bench_map.get(pick)

    c1,c2 = st.columns([1,1])
    with c1:
        if st.button("Set as GK", key=f"{key_prefix}_setgk"):
            set_gk(mid, team_rows, formation, pid); st.rerun()
    with c2:
        if st.button("Clear position", key=f"{key_prefix}_clear"):
            s = service()
            if not s: st.error("Admin required.")
            else:
                s.table("lineups").upsert([{"id": pid, "line": None, "slot": None}], on_conflict="id").execute()
                clear_caches(); st.success("Cleared."); st.rerun()

    # grid buttons
    lines, max_slots, offsets = _formation_slots(formation)
    for line_i, slots in enumerate(lines):
        cols = st.columns(max_slots)
        for grid_slot in range(max_slots):
            # is this grid cell active for this line?
            start = offsets[line_i]; end = start + slots
            if grid_slot < start or grid_slot >= end:
                with cols[grid_slot]:
                    st.write("")  # spacer
                continue
            with cols[grid_slot]:
                # show occupant (if any)
                occ = team_rows[(team_rows["line"]==line_i) & (team_rows["slot"]==grid_slot)]
                label = " "
                if not occ.empty:
                    nm = occ.iloc[0]["name"]
                    label = f"{nm}"
                if st.button(label, key=f"{key_prefix}_{line_i}_{grid_slot}", use_container_width=True):
                    place_player(mid, team_rows, formation, pid, line_i, grid_slot); st.rerun()

# =========================
# Admin updates (non-editor)
# =========================
def update_match_formations(match_id: str, formation_a: str, formation_b: str):
    s = service()
    if not s: st.error("Admin required."); return
    s.table("matches").update({"formation_a": formation_a or None, "formation_b": formation_b or None}).eq("id", match_id).execute()
    clear_caches(); st.success("Formations updated.")

def update_lineup_positions(rows: pd.DataFrame):
    s = service()
    if not s: st.error("Admin required."); return
    payload=[]
    for _,r in rows.iterrows():
        payload.append({
            "id": str(r["id"]),
            "line": int(r["line"]) if pd.notna(r.get("line")) else None,
            "slot": int(r["slot"]) if pd.notna(r.get("slot")) else None,
            "is_gk": bool(r.get("is_gk")) if not pd.isna(r.get("is_gk")) else False
        })
    if payload:
        s.table("lineups").upsert(payload, on_conflict="id").execute()
        clear_caches(); st.success("Positions saved.")

# =========================
# Pages
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

    # FotMob-like summary banner
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
        draw_pitch(m.get("formation_a") or ("2-1-2-1" if int(m.get("side_count") or 5)==7 else "1-2-1"), a_rows, show_photos)
    with c2:
        st.caption(m["team_b"])
        draw_pitch(m.get("formation_b") or ("2-1-2-1" if int(m.get("side_count") or 5)==7 else "1-2-1"), b_rows, show_photos)

    if st.session_state.get("is_admin"):
        with st.expander("✏️ Edit match (formations & positions)"):
            sc = int(m.get("side_count") or 5)
            presets5 = ["1-2-1","1-3","2-2","3-1"]; presets7=["2-1-2-1","3-2-1","2-3-1"]
            presets = presets7 if sc==7 else presets5
            colf1, colf2, colf3 = st.columns([2,2,1])
            fa = colf1.selectbox("Formation (Non-bibs)", presets, index=presets.index(m.get("formation_a") or presets[0]) if (m.get("formation_a") in presets) else 0)
            fb = colf2.selectbox("Formation (Bibs)", presets, index=presets.index(m.get("formation_b") or presets[0]) if (m.get("formation_b") in presets) else 0)
            if colf3.button("Save formations"): update_match_formations(mid, fa, fb); st.rerun()

            st.markdown("**Tap-to-place editor**")
            ea, eb = st.columns(2)
            with ea:
                pitch_editor(mid, m["team_a"], fa or m.get("formation_a") or "1-2-1", a_rows, key_prefix=f"editA_{mid}")
            with eb:
                pitch_editor(mid, m["team_b"], fb or m.get("formation_b") or "1-2-1", b_rows, key_prefix=f"editB_{mid}")

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
                f'<div><span class="k">G+A/PG</span> <b>{me_row["ga_pg"]}</b></div>'
                f'<div><span class="k">Team Contrib%</span> <b>{me_row["team_contrib_pct"]}%</b></div>'
                f'</div></div>', unsafe_allow_html=True
            )

        # Duos & Nemesis (per player)
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

        st.markdown("#### Awards")
        aw = fetch_awards(); mine_aw = aw[(aw["player_name"]==sel) | ((p.get("id") is not None) & (aw["player_id"]==p["id"]))]
        if mine_aw.empty: st.caption("No awards yet.")
        else:
            for _, a in mine_aw.sort_values(["season","month","gw"]).iterrows():
                mo = f" · Month {int(a['month'])}" if pd.notna(a["month"]) else ""
                gw = f" · GW {int(a['gw'])}" if pd.notna(a["gw"]) else ""
                st.write(f"🏅 {a['type']} — Season {a['season']}{mo}{gw}")

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
    metric = st.selectbox("Metric", ["G+A","Goals","Assists","G+A per Game","Win %","Team Contribution %","MOTM"], index=0)
    show_photos = st.toggle("Show photos", True)

    agg = player_agg(lfact, season=season, min_games=min_games, last_gw=last_x)

    # MOTM special
    if metric=="MOTM":
        aw = fetch_awards(); motm = aw[aw["type"]=="MOTM"].copy()
        if season is not None: motm = motm[motm["season"]==season]
        df = motm.groupby("player_name").size().reset_index(name="MOTM").sort_values("MOTM", ascending=False)
        df = df.merge(players[["name","photo_url"]], left_on="player_name", right_on="name", how="left")
        df["photo"] = df["photo_url"]; df["name"] = df["player_name"]
    else:
        df = agg.copy()
        order_map = {
            "Goals": ["goals","ga","assists"],
            "Assists": ["assists","ga","goals"],
            "G+A": ["ga","goals","assists"],
            "G+A per Game": ["ga_pg","ga","goals"],
            "Win %": ["win_pct","ga","goals"],
            "Team Contribution %": ["team_contrib_pct","ga","goals"]
        }
        by = order_map.get(metric, ["ga","goals","assists"])
        df = df.sort_values(by, ascending=False)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    for r in df.head(top_n).to_dict(orient="records"):
        name = r.get("name") or r.get("player_name") or "Unknown"
        photo = (r.get("photo") or r.get("photo_url")) if show_photos else None
        if metric=="Goals": val=r.get("goals",0)
        elif metric=="Assists": val=r.get("assists",0)
        elif metric=="G+A": val=r.get("ga",0)
        elif metric=="G+A per Game": val=r.get("ga_pg",0)
        elif metric=="Win %": val=r.get("win_pct",0)
        elif metric=="Team Contribution %": val=r.get("team_contrib_pct",0)
        elif metric=="MOTM": val=r.get("MOTM",0)
        else: val=r.get("ga",0)
        cA,cB = st.columns([6,1])
        with cA: st.markdown(chip(name, r.get("goals",0), r.get("assists",0), photo), unsafe_allow_html=True)
        with cB: st.metric(metric, val)

    st.markdown("### Duos (Global) · min 3 GP")
    dg = duo_global(lfact, min_gp=3)
    if dg.empty: st.caption("—")
    else:
        for _,r in dg.head(10).iterrows():
            st.write(f"👥 {r['pair']} — {r['win_pct']}% · GP {int(r['gp'])}")

    st.markdown("### Nemesis (Global) · min 3 GP")
    ng = nemesis_global(lfact, min_gp=3)
    if ng.empty: st.caption("—")
    else:
        for _,r in ng.head(10).iterrows():
            st.write(f"⚔️ {r['pair']} — {r['win_pct_vs']}% · GP {int(r['gp'])}")

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

def page_admin():
    st.subheader("Admin Tools")
    st.write("Historical import is **done in Supabase** via staging tables and `import.sql`.")
    st.download_button("Export players.csv", fetch_players().to_csv(index=False).encode("utf-8"), file_name="players.csv", mime="text/csv")
    st.download_button("Export matches.csv", fetch_matches().to_csv(index=False).encode("utf-8"), file_name="matches.csv", mime="text/csv")
    st.download_button("Export lineups.csv", fetch_lineups().to_csv(index=False).encode("utf-8"), file_name="lineups.csv", mime="text/csv")
    st.button("Force refresh caches", on_click=clear_caches)

# =========================
# Router
# =========================
def run_app():
    header()
    Page = getattr(st, "Page", None); nav = getattr(st, "navigation", None)
    if Page and nav:
        sections = {"Main":[
            Page(page_matches, title="Matches", icon="📋"),
            Page(page_players, title="Players", icon="👤"),
            Page(page_stats, title="Stats", icon="📊"),
            Page(page_awards, title="Awards", icon="🏆"),
        ]}
        if st.session_state.get("is_admin"):
            sections["Admin"] = [Page(page_admin, title="Export & Cache", icon="🛠️")]
        n = nav(sections); n.run()
    else:
        sel = st.sidebar.radio("Go to", ["Matches","Players","Stats","Awards"] + (["Admin"] if st.session_state.get("is_admin") else []), index=0)
        {"Matches":page_matches,"Players":page_players,"Stats":page_stats,"Awards":page_awards,"Admin":page_admin}.get(sel, page_matches)()

if __name__ == "__main__":
    run_app()
