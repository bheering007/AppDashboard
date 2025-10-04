import io
import shutil
import sqlite3
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st
import yaml

from app_review import (
    ROLE_NAMES,
    fetch_audit_log,
    get_interview_definition,
    import_csv_to_db,
    update_review,
)

st.set_page_config(page_title="NMC Applications", layout="wide")

BADGE_PREFIX = "badge__"
LEGACY_OUTCOME_MAP = {
    "true": "yes",
    "hold": "maybe",
    "false": "no",
}



def snapshot_review_data(cfg: Dict, event: str = "update") -> None:
    persistence_cfg = cfg.get("persistence") or {}
    if not persistence_cfg.get("enabled", True):
        return
    snapshot_dir = Path(persistence_cfg.get("snapshot_dir", "data_snapshots")).resolve()
    try:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        st.warning(f"Snapshot folder error: {exc}")
        return
    keep_history = persistence_cfg.get("keep_history", True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    resources: List[tuple[Path, Path]] = []
    db_path = Path(cfg.get("database", {}).get("path", ""))
    if db_path.exists():
        resources.append((db_path, snapshot_dir / db_path.name))
    csv_cfg = cfg.get("csv", {})
    csv_candidate = csv_cfg.get("write_back_path") or csv_cfg.get("path")
    csv_path = Path(csv_candidate) if csv_candidate else None
    if csv_path and csv_path.exists():
        resources.append((csv_path, snapshot_dir / csv_path.name))
    history_paths: List[Path] = []
    for src, dest in resources:
        try:
            shutil.copy2(src, dest)
        except Exception as exc:
            st.warning(f"Snapshot copy failed for {src.name}: {exc}")
            continue
        if keep_history:
            history_dir = snapshot_dir / "history"
            try:
                history_dir.mkdir(parents=True, exist_ok=True)
                hist_suffix = f"{timestamp}-{event}-{src.name}" if event else f"{timestamp}-{src.name}"
                hist_dest = history_dir / hist_suffix
                shutil.copy2(src, hist_dest)
                history_paths.append(hist_dest)
            except Exception as exc:
                st.warning(f"History snapshot failed for {src.name}: {exc}")
    if not resources:
        return
    if not persistence_cfg.get("auto_git_commit"):
        return
    try:
        repo_root = Path(persistence_cfg.get("repo_root", ".")).resolve()
        tracked_paths = {dest.resolve() for _, dest in resources if dest.exists()}
        tracked_paths.update(path.resolve() for path in history_paths if path.exists())
        git_paths = []
        for path in tracked_paths:
            if path.is_relative_to(repo_root):
                git_paths.append(str(path.relative_to(repo_root)))
            else:
                git_paths.append(str(path))
        if not git_paths:
            return
        subprocess.run(["git", "add", *git_paths], cwd=repo_root, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        diff_proc = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo_root)
        if diff_proc.returncode == 0:
            subprocess.run(["git", "reset", "--", *git_paths], cwd=repo_root, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        if diff_proc.returncode not in {0, 1}:
            raise RuntimeError("git diff --cached failed")
        message = persistence_cfg.get("commit_message") or f"chore: snapshot review data ({timestamp})"
        commit_proc = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if commit_proc.returncode != 0:
            raise RuntimeError(commit_proc.stderr.strip() or commit_proc.stdout.strip() or "git commit failed")
    except Exception as exc:
        st.warning(f"Git snapshot failed: {exc}")


def build_zip_from_files(paths: List[Path]) -> bytes | None:
    valid = [path for path in paths if path and path.exists()]
    if not valid:
        return None
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in valid:
            archive.write(path, arcname=path.name)
    buffer.seek(0)
    return buffer.getvalue()


def setup_autorefresh(seconds: float | int) -> None:
    if not seconds or seconds <= 0:
        return
    refresh_fn = getattr(st, "autorefresh", None)
    interval_ms = int(seconds * 1000)
    if callable(refresh_fn):
        refresh_fn(interval=interval_ms, limit=None, key="auto_refresh_timer")
    else:
        st.markdown(
            f"<script>setTimeout(function(){{window.location.reload();}},{interval_ms});</script>",
            unsafe_allow_html=True,
        )


def trigger_rerun() -> None:
    rerun = getattr(st, "rerun", None)
    if callable(rerun):
        rerun()
        return
    experimental = getattr(st, "experimental_rerun", None)
    if callable(experimental):
        experimental()
    else:
        raise RuntimeError("Streamlit rerun API not available")



def login(auth_cfg: Dict) -> tuple[str, Dict]:
    users = auth_cfg.get("users", {}) or {}
    if not users:
        return "guest", {}

    session = st.session_state
    saved_user = session.get("_auth_user")
    if saved_user and saved_user in users:
        return saved_user, users[saved_user]

    st.title("New Maroon Camp – Application Review")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username").strip()
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Log in")
    if submit:
        profile = users.get(username)
        if profile and password == profile.get("password"):
            session["_auth_user"] = username
            session["_auth_display"] = profile.get("display_name", username)
            session.pop("_auth_error", None)
            trigger_rerun()
        else:
            session["_auth_error"] = "Invalid username or password."
            trigger_rerun()
    if session.get("_auth_error"):
        st.error(session["_auth_error"])
    st.stop()


def get_cfg() -> Dict:
    config_path = Path("config.yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    # Attach the file timestamp so callers can detect changes if needed
    data.setdefault("_meta", {})["config_mtime"] = config_path.stat().st_mtime
    return data


@st.cache_data(show_spinner=False)
def load_dataframe(
    db_path: str,
    table: str,
    order_by: str | None,
    session_version: int,
    storage_token: float,
) -> pd.DataFrame:
    db_file = Path(db_path)
    if not db_file.exists():
        return pd.DataFrame()
    with sqlite3.connect(db_path) as conn:
        try:
            df = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
        except Exception:
            return pd.DataFrame()
    if order_by and order_by in df.columns:
        try:
            df[order_by] = pd.to_datetime(df[order_by])
            df.sort_values(order_by, ascending=False, inplace=True, ignore_index=True)
        except Exception:
            pass
    return df


def prepare_dataframe(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    df = df.copy()
    unique_key = cfg["database"]["unique_key"]
    if unique_key in df.columns:
        df[unique_key] = df[unique_key].astype(str)
    score_col = cfg["review"]["rubric_score_field"]
    if score_col in df.columns:
        df["_fit_numeric"] = pd.to_numeric(df[score_col], errors="coerce")
    ai_score_col = "ai_score"
    if ai_score_col in df.columns:
        df["_ai_numeric"] = pd.to_numeric(df[ai_score_col], errors="coerce")
    rating_base = cfg["review"].get("rating_field", "review_score")
    rating_cols = []
    if rating_base:
        rating_cols = [col for col in df.columns if col.startswith(f"{rating_base}__")]
        if rating_cols:
            avg_col = f"{rating_base}_avg"

            def _avg_rating(row):
                values = []
                for col in rating_cols:
                    val = row.get(col, "")
                    if pd.isna(val) or val == "":
                        continue
                    try:
                        numeric = float(val)
                    except (TypeError, ValueError):
                        continue
                    values.append(max(0.0, min(5.0, numeric)))
                return round(sum(values) / len(values), 1) if values else None

            df[avg_col] = df[rating_cols].apply(_avg_rating, axis=1)
            df["_rating_avg_numeric"] = pd.to_numeric(df[avg_col], errors="coerce")
    status_col = cfg["review"]["status_field"]
    if status_col in df.columns:
        df[status_col] = df[status_col].fillna("")
    notes_col = cfg["review"]["notes_field"]
    note_cols = [col for col in df.columns if col.startswith(f"{notes_col}__")]
    for col in [notes_col] + note_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")
    recommend_base = cfg["review"].get("recommendation_field", "review_recommendation")
    if recommend_base:
        recommend_cols = [col for col in df.columns if col.startswith(f"{recommend_base}__")]
        for col in recommend_cols:
            df[col] = df[col].fillna("")
    interview_cols = [col for col in df.columns if col.startswith("interview_")]
    for col in interview_cols:
        df[col] = df[col].fillna("")
    return df


def badge_columns(df: pd.DataFrame) -> Dict[str, str]:
    cols = {}
    for col in df.columns:
        if col.startswith(BADGE_PREFIX):
            label = col[len(BADGE_PREFIX) :].replace("_", " ").title()
            cols[col] = label
    return cols


def filtered_dataframe(
    df: pd.DataFrame,
    cfg: Dict,
    search_text: str,
    status_filter: List[str],
    ai_filter: str,
    gpa_only: bool,
    badge_filter: List[str],
    role_focus: str,
    role_pref_rule: str,
) -> pd.DataFrame:
    unique_key = cfg["database"]["unique_key"]
    status_col = cfg["review"]["status_field"]
    notes_col = cfg["review"]["notes_field"]
    ai_col = cfg["review"]["ai_flag_field"]
    note_prefix = cfg["review"]["notes_field"]
    note_search_cols = [col for col in df.columns if col.startswith(f"{note_prefix}__")]
    search_cols = set(
        cfg.get("fields", {}).get("name_fields", [])
        + cfg.get("fields", {}).get("contact_fields", [])
        + [notes_col, "summary"]
        + note_search_cols
    )
    filtered = df.copy()
    if search_text:
        needle = search_text.lower()
        filtered = filtered[
            filtered.apply(
                lambda row: any(
                    needle in str(row.get(col, "")).lower() for col in search_cols if col in row
                ),
                axis=1,
            )
        ]
    if status_filter:
        filtered = filtered[filtered[status_col].isin(status_filter)]
    if ai_filter == "Flagged" and ai_col in filtered.columns:
        filtered = filtered[filtered[ai_col] == "YES"]
    elif ai_filter == "Not flagged" and ai_col in filtered.columns:
        filtered = filtered[filtered[ai_col] == "NO"]
    if gpa_only:
        if "gpa_flag" in filtered.columns:
            filtered = filtered[filtered["gpa_flag"].fillna("") == "LOW"]
        else:
            filtered = filtered.iloc[0:0]
    for badge_col in badge_filter:
        if badge_col in filtered.columns:
            filtered = filtered[filtered[badge_col].str.lower() == "yes"]
    if role_focus != "All":
        pref_col = f"{role_focus}__pref"
        if pref_col in filtered.columns:
            normalized = filtered[pref_col].astype(str).str.strip().str.lower()
            filtered = filtered[~normalized.isin({"", "nan", "none", "null"})]
            if role_pref_rule == "#1 only":
                filtered = filtered[normalized == "1"]
            elif role_pref_rule == "#1 or #2":
                filtered = filtered[normalized.isin({"1", "2"})]
        else:
            filtered = filtered.iloc[0:0]
    sort_cols = []
    sort_orders = []
    rating_base = cfg["review"].get("rating_field", "review_score")
    rating_avg_col = f"{rating_base}_avg" if rating_base else None
    if rating_avg_col and rating_avg_col in filtered.columns:
        sort_cols.append(rating_avg_col)
        sort_orders.append(False)
    if ai_col in filtered.columns:
        sort_cols.append(ai_col)
        sort_orders.append(True)
    if sort_cols:
        filtered = filtered.sort_values(by=sort_cols, ascending=sort_orders, ignore_index=True)
    filtered[unique_key] = filtered[unique_key].astype(str)
    return filtered



def reviewer_pending_mask(df: pd.DataFrame, reviewer_cols: List[str]) -> pd.Series:
    if df.empty:
        return pd.Series([], dtype=bool, index=df.index)
    if not reviewer_cols:
        return pd.Series([False] * len(df), index=df.index, dtype=bool)
    mask = pd.Series(True, index=df.index, dtype=bool)
    for col in reviewer_cols:
        if col not in df.columns:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            empty_series = series.isna()
        else:
            empty_series = series.fillna("").astype(str).str.strip() == ""
        mask &= empty_series
    return mask


def format_option_label(row: pd.Series, cfg: Dict) -> str:
    status_col = cfg["review"]["status_field"]
    first_name = row.get("First Name", "")
    last_name = row.get("Last Name", "")
    status = row.get(status_col, "").upper() or "UNREVIEWED"
    return f"{row[cfg['database']['unique_key']]} – {first_name} {last_name} ({status})"


def render_badges(row: pd.Series, badge_map: Dict[str, str]) -> None:
    if not badge_map:
        return
    palette = {"yes": "#20895d", "maybe": "#e0a800", "no": "#9aa1ad"}
    chips = []
    for column, label in badge_map.items():
        value = str(row.get(column, "")).lower()
        if not value:
            continue
        color = palette.get(value, "#666")
        chips.append(
            f"<span style='background-color:{color};padding:2px 8px;border-radius:12px;color:white;margin-right:6px;font-size:0.75rem;'>"
            f"{label}: {value.title()}</span>"
        )
    if chips:
        st.markdown("".join(chips), unsafe_allow_html=True)


def render_role_table(df: pd.DataFrame, role: str, cfg: Dict) -> None:
    score_col_raw = cfg["review"].get("rubric_score_field")
    score_col = score_col_raw if score_col_raw and score_col_raw in df.columns else None
    status_col = cfg["review"]["status_field"]
    recommend_base = cfg["review"].get("recommendation_field", "review_recommendation")
    display_cols = [
        cfg["database"]["unique_key"],
        "First Name",
        "Last Name",
        f"{role}__fit",
        f"{role}__pref",
        status_col,
        cfg["review"]["notes_field"],
        cfg["review"]["ai_flag_field"],
        "gpa_flag",
    ]
    if score_col:
        display_cols.insert(3, score_col)
    display_cols = [col for col in display_cols if col in df.columns]
    if not display_cols:
        st.info("No data available for this role yet.")
        return
    data = df.copy()
    pref_col = f"{role}__pref"
    if pref_col in data.columns:
        normalized = data[pref_col].astype(str).str.strip().str.lower()
        data = data[~normalized.isin({"", "nan", "none", "null"})]
    if data.empty:
        st.info("No applicants ranked this role yet.")
        return
    if f"{role}__fit" in data.columns:
        data["_role_fit_numeric"] = pd.to_numeric(data[f"{role}__fit"], errors="coerce")
        data.sort_values("_role_fit_numeric", ascending=False, inplace=True, ignore_index=True)
    st.dataframe(data[display_cols], use_container_width=True, hide_index=True)
    if not data.empty:
        csv_export = data[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            f"Download {role} shortlist",
            data=csv_export,
            file_name=f"{role.lower().replace(' ', '_')}_shortlist.csv",
            mime="text/csv",
        )


cfg = get_cfg()
current_user, current_profile = login(cfg.get("auth", {}))
setup_autorefresh(cfg.get("ui", {}).get("auto_refresh_seconds", 0))
notes_col = cfg["review"]["notes_field"]
status_col = cfg["review"]["status_field"]
ai_col = cfg["review"]["ai_flag_field"]
score_col_raw = cfg["review"].get("rubric_score_field")
rating_base = cfg["review"].get("rating_field", "review_score")
recommend_base = cfg["review"].get("recommendation_field", "review_recommendation")
assignment_field = cfg["review"].get("assignment_field", "role_assignment")
family_field = cfg["review"].get("family_field", "family_group")
family_pair_field = cfg["review"].get("family_pair_field", "family_pod")
family_cfg = cfg.get("family", {}) or {}
raw_family_names = family_cfg.get("names") or []
family_names = [str(name).strip() for name in raw_family_names if str(name).strip()]
family_default_capacity = family_cfg.get("default_capacity", 12)
family_capacities_cfg = family_cfg.get("capacities", {}) or {}
staffing_cfg = cfg.get("staffing", {}) or {}
role_targets_cfg = staffing_cfg.get("role_targets", {}) or {}
interview_def = get_interview_definition(cfg)
interview_prompts = interview_def.get("prompts", [])
interview_prompt_cols = {item["slug"]: f"{item['column']}__{current_user}" for item in interview_prompts}
user_note_col = f"{notes_col}__{current_user}"
user_rating_col = f"{rating_base}__{current_user}" if rating_base else None
user_recommend_col = f"{recommend_base}__{current_user}" if recommend_base else None
user_interview_rating_col = f"{interview_def['rating_field']}__{current_user}"
user_interview_outcome_col = f"{interview_def['outcome_field']}__{current_user}"
user_interview_log_col = f"{interview_def['log_field']}__{current_user}"
user_interview_resources_col = f"{interview_def['resources_field']}__{current_user}"
user_interview_summary_col = f"{interview_def['summary_field']}__{current_user}"
interview_questions = interview_def.get("questions", [])
interview_question_cols = {item["slug"]: f"{item['column']}__{current_user}" for item in interview_questions}
interview_outcome_options = interview_def.get("outcome_options", [])
interview_summary_template = interview_def.get("summary_template", [])
reviewer_personal_cols = [col for col in [user_note_col, user_rating_col, user_recommend_col] if col]
interview_resource_links = cfg.get("interview", {}).get("resource_links", [])
all_reviewers = cfg.get("auth", {}).get("users", {}) or {}
csv_write_target = cfg["csv"].get("write_back_path", cfg["csv"]["path"])
if "data_version" not in st.session_state:
    st.session_state["data_version"] = 0
if "last_uploaded_token" not in st.session_state:
    st.session_state["last_uploaded_token"] = None
if "upload_message" not in st.session_state:
    st.session_state["upload_message"] = ""

if st.session_state.get("upload_message"):
    st.toast(st.session_state.pop("upload_message"), icon="✅")
if st.session_state.get("assignment_message"):
    st.toast(st.session_state.pop("assignment_message"), icon="📦")
if st.session_state.get("family_message"):
    st.toast(st.session_state.pop("family_message"), icon="👪")

order_by = cfg.get("database", {}).get("order_by", "DateSubmitted")
db_file = Path(cfg["database"]["path"])
db_token = db_file.stat().st_mtime if db_file.exists() else 0.0
raw_df = load_dataframe(
    cfg["database"]["path"],
    cfg["database"]["table"],
    order_by,
    st.session_state["data_version"],
    db_token,
)
df = prepare_dataframe(raw_df, cfg)
score_col = score_col_raw if score_col_raw and score_col_raw in df.columns else None
pending_mask_all = reviewer_pending_mask(df, reviewer_personal_cols)
badge_map = badge_columns(df)

reviewer_progress: List[Dict[str, object]] = []
if not df.empty:
    total_applicants = len(df)
    for reviewer_username, profile in all_reviewers.items():
        reviewer_cols = []
        note_col = f"{notes_col}__{reviewer_username}"
        if note_col in df.columns:
            reviewer_cols.append(note_col)
        if rating_base:
            rating_col = f"{rating_base}__{reviewer_username}"
            if rating_col in df.columns:
                reviewer_cols.append(rating_col)
        if recommend_base:
            recommend_col = f"{recommend_base}__{reviewer_username}"
            if recommend_col in df.columns:
                reviewer_cols.append(recommend_col)
        if not reviewer_cols:
            continue
        mask = reviewer_pending_mask(df, reviewer_cols)
        pending = int(mask.sum())
        reviewer_progress.append(
            {
                "username": reviewer_username,
                "display": profile.get("display_name", reviewer_username),
                "pending": pending,
                "completed": total_applicants - pending,
                "total": total_applicants,
            }
        )

with st.sidebar:
    display_name = st.session_state.get("_auth_display", current_profile.get("display_name", current_user))
    st.markdown(f"**Logged in as:** {display_name}")
    if st.button("Log out", key="logout_btn"):
        for key in ["_auth_user", "_auth_display", "data_version", "active_submission", "review_status_value", "review_notes_value", "review_rating_value", "review_rating_str", "review_snapshot", "upload_message", "last_uploaded_token", "review_recommend_value"]:
            st.session_state.pop(key, None)
        for dynamic_key in list(st.session_state.keys()):
            if dynamic_key.startswith("interview_"):
                st.session_state.pop(dynamic_key, None)
        trigger_rerun()
    if reviewer_progress:
        with st.expander("Reviewer progress", expanded=False):
            for stats in sorted(reviewer_progress, key=lambda item: item["pending"], reverse=True):
                label_prefix = "(you) " if stats["username"] == current_user else ""
                completed = stats["completed"]
                total = stats["total"]
                pending = stats["pending"]
                completion_pct = 0 if not total else int(round((completed / total) * 100))
                st.write(
                    f"**{label_prefix}{stats['display']}** — {completed}/{total} completed · {pending} pending ({completion_pct}%)"
                )
    st.header("Manage Data")
    upload = st.file_uploader("Add new CSV export", type=["csv"], help="Drop a fresh form export to append/update applicants.")

    def process_upload(file):
        if file is None:
            return
        token = f"{file.name}-{file.size}"
        if st.session_state.get("last_uploaded_token") == token:
            return
        incoming_dir = Path(cfg["csv"].get("incoming_dir", "incoming_uploads"))
        incoming_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        destination = incoming_dir / f"{timestamp}_{file.name}"
        destination.write_bytes(file.getvalue())
        enriched = import_csv_to_db(
            csv_path=str(destination),
            skiprows=cfg["csv"].get("skiprows", 0),
            db_path=cfg["database"]["path"],
            table=cfg["database"]["table"],
            unique_key=cfg["database"]["unique_key"],
            cfg=cfg,
            write_back_csv=cfg["csv"].get("write_back", True),
            write_back_target=csv_write_target,
        )
        snapshot_review_data(cfg, event="import")
        st.session_state["last_uploaded_token"] = token
        st.session_state["upload_message"] = f"Imported {len(enriched)} rows from {file.name}."
        st.session_state["data_version"] += 1
        load_dataframe.clear()
        trigger_rerun()

    process_upload(upload)

    st.divider()
    with st.expander("Filter applicants", expanded=False):
        search_text = st.text_input(
            "Search name/email/notes",
            help="Matches across names, contact info, shared notes, and your comments.",
        )
        pending_only = st.checkbox(
            "Only show my pending reviews",
            key="pending_only_filter",
            help="Show applications where you have not saved notes, rating, or recommendation yet.",
        )
        status_filter = st.multiselect(
            "Review status",
            options=cfg["review"].get("allowed_statuses", ["yes", "maybe", "no"]),
            help="Leave empty to include every decision.",
        )
        ai_filter = st.selectbox("AI flag", options=["All", "Flagged", "Not flagged"], index=0)
        gpa_only = st.checkbox("Only show GPA flag (<3.0)", value=False)
        role_focus = st.selectbox("Role focus", options=["All"] + ROLE_NAMES)
        role_pref_rule = st.selectbox("Preference filter", options=["Any", "#1 only", "#1 or #2"], index=0)
        badge_label = "Must include badges" if badge_map else "Badges"
        selected_badges = st.multiselect(
            badge_label,
            options=list(badge_map.keys()),
            format_func=lambda key: badge_map[key],
        )

    st.divider()
    with st.expander("Download backups", expanded=False):
        download_files: List[Path] = []
        if db_file.exists():
            db_label = datetime.fromtimestamp(db_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            st.download_button(
                "Database (.sqlite)",
                data=db_file.read_bytes(),
                file_name=db_file.name,
                mime="application/octet-stream",
                key="dl_sqlite_db",
            )
            st.caption(f"DB updated {db_label}")
            download_files.append(db_file)
        else:
            st.caption("Database file not found yet. Upload a CSV to initialize it.")

        csv_candidate = Path(csv_write_target)
        if csv_candidate.exists():
            csv_label = datetime.fromtimestamp(csv_candidate.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            st.download_button(
                "Master CSV",
                data=csv_candidate.read_bytes(),
                file_name=csv_candidate.name,
                mime="text/csv",
                key="dl_master_csv",
            )
            st.caption(f"CSV updated {csv_label}")
            download_files.append(csv_candidate)

        archive_bytes = build_zip_from_files(download_files)
        if archive_bytes:
            st.download_button(
                "ZIP backup (DB + CSV)",
                data=archive_bytes,
                file_name="nmc_review_backup.zip",
                mime="application/zip",
                key="dl_zip_bundle",
            )
        snapshot_dir = Path(cfg.get("persistence", {}).get("snapshot_dir", "data_snapshots"))
        history_dir = snapshot_dir / "history"
        if history_dir.exists():
            recent = sorted(history_dir.glob("*"), key=lambda item: item.stat().st_mtime, reverse=True)[:5]
            if recent:
                st.caption("Recent history snapshots:")
                for item in recent:
                    ts = datetime.fromtimestamp(item.stat().st_mtime).strftime("%m-%d %H:%M")
                    st.text(f"{item.name} ({ts})")

    if df.empty:
        st.info("No applications in the database yet. Upload your Formstack CSV using the sidebar to get started.")
        st.stop()

filtered_df = filtered_dataframe(
    df=df,
    cfg=cfg,
    search_text=search_text,
    status_filter=status_filter,
    ai_filter=ai_filter,
    gpa_only=gpa_only,
    badge_filter=selected_badges,
    role_focus=role_focus,
    role_pref_rule=role_pref_rule,
)
pending_only_active = st.session_state.get("pending_only_filter", False)
if pending_only_active and not filtered_df.empty:
    filtered_df = filtered_df[reviewer_pending_mask(filtered_df, reviewer_personal_cols)]

unique_key = cfg["database"]["unique_key"]

if not family_names:
    if family_field in df.columns:
        family_names = sorted(
            {
                value.strip()
                for value in df[family_field].astype(str).tolist()
                if value and value.strip()
            }
        )
    if not family_names:
        family_names = ["Family 1", "Family 2", "Family 3", "Family 4", "Family 5"]

family_name_lookup = {name.lower(): name for name in family_names}
family_capacity_map = {
    name: int(family_capacities_cfg.get(name, family_capacities_cfg.get(name.lower(), family_default_capacity)))
    for name in family_names
}

role_badge_symbols = ["🟦", "🟩", "🟧", "🟥", "🟪", "🟨", "🟫", "⬛"]
role_badge_map = {role: role_badge_symbols[idx % len(role_badge_symbols)] for idx, role in enumerate(ROLE_NAMES)}

PRESET_FAMILY_COLORS = {
    "rev": ("🟪", "#8b5cf6"),
    "collegian": ("🟨", "#facc15"),
    "tonk": ("🟥", "#f87171"),
    "cadets": ("🟩", "#34d399"),
    "ptolomy": ("🟦", "#60a5fa"),
    "ptolemy": ("🟦", "#60a5fa"),
    "maroons": ("🟧", "#fb923c"),
}
fallback_family_symbols = ["⬜", "⬛", "🟪", "🟧", "🟩", "🟦", "🟥", "🟨", "🟫"]
family_badge_map: dict[str, str] = {}
family_color_map: dict[str, str] = {}
for idx, name in enumerate(family_names):
    key = name.lower()
    if key in PRESET_FAMILY_COLORS:
        symbol, color = PRESET_FAMILY_COLORS[key]
    else:
        symbol = fallback_family_symbols[idx % len(fallback_family_symbols)]
        color = "#9ca3af"
    family_badge_map[name] = symbol
    family_badge_map[key] = symbol
    family_color_map[name] = color
    family_color_map[key] = color

def normalize_family(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    return family_name_lookup.get(value.lower(), value)

def format_role_badge(role_value: str) -> str:
    value = str(role_value or "").strip()
    if not value:
        return "—"
    symbol = role_badge_map.get(value, "▪️")
    return f"{symbol} {value}"

def format_family_badge(family_value: str) -> str:
    value = str(family_value or "").strip()
    if not value:
        return "—"
    symbol = family_badge_map.get(value, family_badge_map.get(value.lower(), "▪️"))
    return f"{symbol} {value}"

if assignment_field in filtered_df.columns:
    filtered_df["Role tag"] = filtered_df[assignment_field].apply(format_role_badge)
else:
    filtered_df["Role tag"] = "—"
if family_field in filtered_df.columns:
    filtered_df[family_field] = filtered_df[family_field].apply(normalize_family)
    filtered_df["Family tag"] = filtered_df[family_field].apply(format_family_badge)
else:
    filtered_df["Family tag"] = "—"

review_tab, staffing_tab = st.tabs(["Review Dashboard", "Staffing Boards"])

with review_tab:
    col_metrics = st.columns(5)
    with col_metrics[0]:
        st.metric("Applicants", len(df))
    with col_metrics[1]:
        reviewed = df[df[status_col].isin(cfg["review"].get("allowed_statuses", []))]
        st.metric("Reviewed", len(reviewed))
    with col_metrics[2]:
        st.metric("AI flagged", int((df[ai_col] == "YES").sum()))
    with col_metrics[3]:
        if "gpa_flag" in df.columns:
            gpa_series = df["gpa_flag"].fillna("")
        else:
            gpa_series = pd.Series(dtype=object)
        st.metric("GPA alerts", int((gpa_series == "LOW").sum()))
    with col_metrics[4]:
        st.metric("My pending reviews", int(pending_mask_all.sum()))

    st.caption("Autosave is enabled for review status and notes. Filters affect the table and selection list but not the underlying data.")

    rating_base = cfg["review"].get("rating_field", "review_score")
    rating_avg_col = f"{rating_base}_avg" if rating_base else None
    default_table_cols = [
        unique_key,
        "First Name",
        "Last Name",
        *(cfg.get("fields", {}).get("contact_fields", [])[:1]),
    ]
    if score_col:
        default_table_cols.append(score_col)
    if rating_avg_col:
        default_table_cols.append(rating_avg_col)
    default_table_cols.extend([
        ai_col,
        "ai_score",
        "gpa_flag",
        status_col,
        "Role tag",
        "Family tag",
        notes_col,
        "summary",
    ])
    available_cols = [col for col in default_table_cols if col in filtered_df.columns]
    with st.expander("Applicant table", expanded=True):
        selected_table_cols = st.multiselect(
            "Columns to show",
            options=list(filtered_df.columns),
            default=available_cols,
        )
        if not selected_table_cols:
            st.warning("Select at least one column to display.")
        else:
            st.dataframe(
                filtered_df[selected_table_cols],
                use_container_width=True,
                hide_index=True,
            )
            download_ready = filtered_df[selected_table_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download filtered CSV",
                data=download_ready,
                file_name="nmc_filtered.csv",
                mime="text/csv",
            )

    st.subheader("Review Panel")
    if filtered_df.empty:
        st.info("No applicants match the current filters.")
    else:
        option_labels = {
            row[unique_key]: format_option_label(row, cfg) for _, row in filtered_df.iterrows()
        }
        options = list(option_labels.keys())
        default_index = 0
        active_submission = st.session_state.get("active_submission")
        if active_submission in options:
            default_index = options.index(active_submission)
        elif options:
            st.session_state["active_submission"] = options[0]
        selected_id = st.selectbox(
            "Choose applicant",
            options=options,
            index=default_index,
            format_func=lambda sid: option_labels.get(sid, str(sid)),
        )

        if selected_id:
            selected_row = df[df[unique_key] == selected_id].iloc[0]
            note_value = selected_row.get(user_note_col, "")
            if pd.isna(note_value):
                note_value = ""
            rating_raw = selected_row.get(user_rating_col, "") if user_rating_col else ""
            if pd.isna(rating_raw):
                rating_raw = ""
            rating_value = 0.0
            rating_str = ""
            if user_rating_col:
                if isinstance(rating_raw, (int, float)) and not pd.isna(rating_raw):
                    rating_value = max(0.0, min(5.0, float(rating_raw)))
                    rating_str = f"{rating_value:.1f}"
                elif isinstance(rating_raw, str) and rating_raw.strip():
                    try:
                        rating_value = max(0.0, min(5.0, float(rating_raw)))
                        rating_str = f"{rating_value:.1f}"
                    except ValueError:
                        rating_str = rating_raw.strip()
            recommend_raw = selected_row.get(user_recommend_col, "") if user_recommend_col else ""
            if pd.isna(recommend_raw):
                recommend_raw = ""
            recommend_value = str(recommend_raw).strip().lower() if recommend_raw else ""
            interview_prompt_values = {}
            for prompt in interview_prompts:
                slug = prompt.get("slug")
                column_name = interview_prompt_cols.get(slug)
                raw = selected_row.get(column_name, "") if column_name else ""
                if pd.isna(raw):
                    raw = ""
                interview_prompt_values[slug] = str(raw)
            interview_summary_value = ""
            if user_interview_summary_col:
                summary_raw = selected_row.get(user_interview_summary_col, "")
                if pd.isna(summary_raw):
                    summary_raw = ""
                interview_summary_value = str(summary_raw)
            interview_rating_value = 0.0
            interview_rating_display = ""
            if user_interview_rating_col:
                interview_rating_raw = selected_row.get(user_interview_rating_col, "")
                if pd.isna(interview_rating_raw):
                    interview_rating_raw = ""
                if isinstance(interview_rating_raw, (int, float)) and not pd.isna(interview_rating_raw):
                    interview_rating_value = max(0.0, min(5.0, float(interview_rating_raw)))
                    interview_rating_display = f"{interview_rating_value:.1f}"
                elif isinstance(interview_rating_raw, str) and interview_rating_raw.strip():
                    try:
                        interview_rating_value = max(0.0, min(5.0, float(interview_rating_raw)))
                        interview_rating_display = f"{interview_rating_value:.1f}"
                    except ValueError:
                        interview_rating_display = interview_rating_raw.strip()
            interview_outcome_value = ""
            if user_interview_outcome_col:
                outcome_raw = selected_row.get(user_interview_outcome_col, "")
                if pd.isna(outcome_raw):
                    outcome_raw = ""
                normalized_outcome = str(outcome_raw).strip().lower()
                interview_outcome_value = LEGACY_OUTCOME_MAP.get(normalized_outcome, normalized_outcome)
            interview_resources_value = ""
            if user_interview_resources_col:
                resources_raw = selected_row.get(user_interview_resources_col, "")
                if pd.isna(resources_raw):
                    resources_raw = ""
                interview_resources_value = str(resources_raw)
            interview_log_value = ""
            if user_interview_log_col:
                log_raw = selected_row.get(user_interview_log_col, "")
                if pd.isna(log_raw):
                    log_raw = ""
                interview_log_value = str(log_raw)
            question_states = {}
            for item in interview_questions:
                slug = item.get("slug")
                column_name = interview_question_cols.get(slug)
                raw = selected_row.get(column_name, "") if column_name else ""
                if pd.isna(raw):
                    raw = ""
                question_states[slug] = str(raw).strip().lower()
            prompt_snapshot = tuple(interview_prompt_values.get(prompt.get("slug"), "") for prompt in interview_prompts)
            question_snapshot = tuple(question_states.get(item.get("slug"), "") for item in interview_questions)
            snapshot = (
                selected_row.get(status_col, ""),
                note_value,
                rating_str,
                recommend_value,
                prompt_snapshot,
                interview_rating_display,
                interview_outcome_value,
                interview_log_value,
                interview_resources_value,
                interview_summary_value,
                question_snapshot,
            )
            if (
                st.session_state.get("active_submission") != selected_id
                or st.session_state.get("review_snapshot") != snapshot
            ):
                st.session_state["active_submission"] = selected_id
                st.session_state["review_status_value"] = snapshot[0]
                st.session_state["review_notes_value"] = snapshot[1]
                st.session_state["review_rating_value"] = rating_value if rating_str else 0.0
                st.session_state["review_rating_str"] = rating_str
                st.session_state["review_recommend_value"] = recommend_value
                st.session_state["interview_rating_value"] = interview_rating_value if interview_rating_display else 0.0
                st.session_state["interview_rating_display"] = interview_rating_display
                st.session_state["interview_outcome_value"] = interview_outcome_value
                st.session_state["interview_log_value"] = interview_log_value
                st.session_state["interview_log_entry"] = ""
                st.session_state["interview_resources_value"] = interview_resources_value
                st.session_state["interview_summary_value"] = interview_summary_value
                for slug, value in interview_prompt_values.items():
                    if slug:
                        st.session_state[f"interview_prompt_{slug}_value"] = value
                for slug, value in question_states.items():
                    if slug:
                        st.session_state[f"interview_question_{slug}_value"] = value
                st.session_state["review_snapshot"] = snapshot

            allowed_statuses = [""] + cfg["review"].get("allowed_statuses", [])
            question_status_options = ["", "asked", "needs follow-up"]
            question_status_labels = {
                "": "(pending)",
                "asked": "Asked",
                "needs follow-up": "Needs follow-up",
            }

            def current_snapshot():
                rating_display = st.session_state.get("review_rating_str", "") if user_rating_col else ""
                recommend_display = st.session_state.get("review_recommend_value", "") if user_recommend_col else ""
                prompt_snapshot_state = tuple(
                    st.session_state.get(f"interview_prompt_{item['slug']}_value", "") for item in interview_prompts
                )
                interview_rating_state = (
                    st.session_state.get("interview_rating_display", "")
                    if user_interview_rating_col
                    else ""
                )
                interview_outcome_state = (
                    st.session_state.get("interview_outcome_value", "")
                    if user_interview_outcome_col
                    else ""
                )
                interview_log_state = st.session_state.get("interview_log_value", "")
                interview_resources_state = st.session_state.get("interview_resources_value", "")
                interview_summary_state = st.session_state.get("interview_summary_value", "")
                question_snapshot_state = tuple(
                    st.session_state.get(f"interview_question_{item['slug']}_value", "") for item in interview_questions
                )
                return (
                    st.session_state.get("review_status_value", ""),
                    st.session_state.get("review_notes_value", ""),
                    rating_display,
                    recommend_display,
                    prompt_snapshot_state,
                    interview_rating_state,
                    interview_outcome_state,
                    interview_log_state,
                    interview_resources_state,
                    interview_summary_state,
                    question_snapshot_state,
                )

            def persist_updates(updates, toast_message: str) -> None:
                if not updates:
                    return
                csv_write_enabled = cfg["csv"].get("write_back", True)
                update_review(
                    db_path=cfg["database"]["path"],
                    table=cfg["database"]["table"],
                    unique_key=unique_key,
                    keyval=selected_id,
                    updates=updates,
                    csv_path=csv_write_target if csv_write_enabled else None,
                    skiprows=cfg["csv"].get("skiprows", 0),
                    actor=current_user,
                )
                snapshot_review_data(cfg, event="review-update")
                st.session_state["review_snapshot"] = current_snapshot()
                st.session_state["data_version"] += 1
                load_dataframe.clear()
                if toast_message:
                    st.toast(toast_message, icon="💾")

            def save_rating() -> None:
                if not user_rating_col:
                    return
                value = float(st.session_state.get("review_rating_value", 0.0))
                value = max(0.0, min(5.0, value))
                formatted = f"{value:.1f}"
                st.session_state["review_rating_str"] = formatted
                persist_updates({user_rating_col: formatted}, "Rating saved")

            def save_recommendation() -> None:
                if not user_recommend_col:
                    return
                value = st.session_state.get("review_recommend_value", "")
                if isinstance(value, str):
                    value = value.lower().strip()
                persist_updates({user_recommend_col: value}, "Recommendation saved")

            def save_status() -> None:
                value = st.session_state.get("review_status_value", "")
                persist_updates({status_col: value}, "Status saved")

            def save_notes() -> None:
                value = st.session_state.get("review_notes_value", "")
                persist_updates({user_note_col: value}, "Notes saved")

            def save_interview_rating() -> None:
                if not user_interview_rating_col:
                    return
                value = float(st.session_state.get("interview_rating_value", 0.0))
                value = max(0.0, min(5.0, value))
                formatted = f"{value:.1f}"
                st.session_state["interview_rating_display"] = formatted
                persist_updates({user_interview_rating_col: formatted}, "Interview rating saved")

            def save_interview_outcome() -> None:
                if not user_interview_outcome_col:
                    return
                value = str(st.session_state.get("interview_outcome_value", "")).strip().lower()
                value = LEGACY_OUTCOME_MAP.get(value, value)
                st.session_state["interview_outcome_value"] = value
                persist_updates({user_interview_outcome_col: value}, "Interview outcome saved")

            def save_interview_resources() -> None:
                if not user_interview_resources_col:
                    return
                value = st.session_state.get("interview_resources_value", "")
                persist_updates({user_interview_resources_col: value}, "Resource notes saved")

            def save_interview_summary() -> None:
                if not user_interview_summary_col:
                    return
                value = st.session_state.get("interview_summary_value", "")
                persist_updates({user_interview_summary_col: value}, "Interview summary saved")

            def make_interview_prompt_saver(slug: str, column: str, label: str):
                state_key = f"interview_prompt_{slug}_value"

                def _save() -> None:
                    persist_updates({column: st.session_state.get(state_key, "")}, f"{label} saved")

                return _save

            def make_interview_question_saver(slug: str, column: str):
                state_key = f"interview_question_{slug}_value"

                def _save() -> None:
                    persist_updates({column: st.session_state.get(state_key, "")}, "Question status saved")

                return _save

            def append_interview_log() -> None:
                if not user_interview_log_col:
                    return
                entry = st.session_state.get("interview_log_entry", "").strip()
                if not entry:
                    return
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                existing = st.session_state.get("interview_log_value", "").strip()
                new_log = f"[{timestamp}] {entry}" if not existing else f"{existing}\n[{timestamp}] {entry}"
                st.session_state["interview_log_value"] = new_log
                st.session_state["interview_log_entry"] = ""
                persist_updates({user_interview_log_col: new_log}, "Interview log updated")

            if "interview_log_entry" not in st.session_state:
                st.session_state["interview_log_entry"] = ""

            display_row = selected_row.copy()
            display_row[user_note_col] = st.session_state.get("review_notes_value", display_row.get(user_note_col, ""))
            if user_rating_col:
                display_row[user_rating_col] = st.session_state.get("review_rating_str", display_row.get(user_rating_col, ""))
            if user_recommend_col:
                display_row[user_recommend_col] = st.session_state.get("review_recommend_value", display_row.get(user_recommend_col, ""))
            if user_interview_summary_col:
                display_row[user_interview_summary_col] = st.session_state.get("interview_summary_value", display_row.get(user_interview_summary_col, ""))
            if user_interview_rating_col:
                display_row[user_interview_rating_col] = st.session_state.get("interview_rating_display", display_row.get(user_interview_rating_col, ""))
            if user_interview_outcome_col:
                outcome_state = str(
                    st.session_state.get("interview_outcome_value", display_row.get(user_interview_outcome_col, ""))
                ).strip().lower()
                outcome_state = LEGACY_OUTCOME_MAP.get(outcome_state, outcome_state)
                display_row[user_interview_outcome_col] = outcome_state
            if user_interview_log_col:
                display_row[user_interview_log_col] = st.session_state.get("interview_log_value", display_row.get(user_interview_log_col, ""))
            if user_interview_resources_col:
                display_row[user_interview_resources_col] = st.session_state.get("interview_resources_value", display_row.get(user_interview_resources_col, ""))
            for slug, column_name in interview_prompt_cols.items():
                if not column_name:
                    continue
                state_key = f"interview_prompt_{slug}_value"
                if state_key in st.session_state:
                    display_row[column_name] = st.session_state[state_key]
            for slug, column_name in interview_question_cols.items():
                if not column_name:
                    continue
                state_key = f"interview_question_{slug}_value"
                if state_key in st.session_state:
                    display_row[column_name] = st.session_state[state_key]

            overview_tab, my_review_tab, interview_tab, feedback_tab = st.tabs([
                "Overview",
                "My Review",
                "Interview",
                "Feedback",
            ])

            with overview_tab:
                first_name = str(display_row.get("First Name", "") or "").strip()
                last_name = str(display_row.get("Last Name", "") or "").strip()
                name_heading = " ".join(filter(None, [first_name, last_name])) or option_labels.get(selected_id, str(selected_id))
                st.markdown(f"### {name_heading}")
                top_cols = st.columns([2, 1])
                with top_cols[0]:
                    st.write("**Submission ID**", selected_id)
                    for contact_field in cfg.get("fields", {}).get("contact_fields", []):
                        if contact_field in display_row:
                            st.write(f"**{contact_field}**", display_row.get(contact_field, ""))
                with top_cols[1]:
                    st.write("**Decision status**", st.session_state.get("review_status_value", "") or "(pending)")
                    if user_rating_col:
                        rating_display = st.session_state.get("review_rating_str", "")
                        st.write("**My rating**", rating_display or "—")
                    if user_recommend_col:
                        rec_display = st.session_state.get("review_recommend_value", "")
                        st.write("**My recommendation**", rec_display.title() if rec_display else "(none)")
                    ai_flag_value = display_row.get(ai_col, "")
                    ai_score_val = display_row.get("ai_score", "—")
                    st.write("**AI suspect**", f"{ai_flag_value} ({ai_score_val})".strip())
                    gpa_flag_value = str(display_row.get("gpa_flag", "") or "").strip()
                    if gpa_flag_value:
                        st.error("GPA flagged below threshold", icon="⚠️")
                render_badges(display_row, badge_map)
                role_summaries: list[str] = []
                for role in ROLE_NAMES:
                    pref_col = f"{role}__pref"
                    fit_col = f"{role}__fit"
                    pref_value = str(display_row.get(pref_col, "") or "").strip()
                    fit_value = str(display_row.get(fit_col, "") or "").strip()
                    if pref_value or fit_value:
                        pref_label = f"# {pref_value}" if pref_value else "—"
                        fit_label = fit_value if fit_value else "—"
                        role_summaries.append(f"{role}: pref {pref_label} · fit {fit_label}")
            if role_summaries:
                st.caption("Role interest snapshot")
                for line in role_summaries:
                    st.write(line)
            summary_text = str(display_row.get("summary", "") or "").strip()
            if summary_text:
                st.markdown("**AI summary**")
                st.write(summary_text)
            text_questions = cfg.get("fields", {}).get("text_fields", [])
            if text_questions:
                st.markdown("**Application responses**")
                for question in text_questions:
                    answer = str(display_row.get(question, "") or "").strip()
                    with st.expander(question, expanded=False):
                        if answer:
                            st.write(answer)
                        else:
                            st.caption("No response provided.")

            with my_review_tab:
                st.caption("Updates save automatically when you interact with the controls.")
                pending_template = st.session_state.pop("pending_notes_template", None)
                if pending_template is not None:
                    st.session_state["review_notes_value"] = pending_template
                review_cols = st.columns(2)
                with review_cols[0]:
                    if user_rating_col:
                        st.slider(
                            "Reviewer score",
                            min_value=0.0,
                            max_value=5.0,
                            step=0.5,
                            key="review_rating_value",
                            on_change=save_rating,
                            help="0 = not ready / 5 = outstanding. Autosaves on release.",
                        )
                    if user_recommend_col:
                        recommendation_options = ["", "yes", "maybe", "no"]
                        current_recommend = st.session_state.get("review_recommend_value", "")
                        if current_recommend not in recommendation_options:
                            current_recommend = ""
                            st.session_state["review_recommend_value"] = current_recommend
                        st.selectbox(
                            "My recommendation",
                            options=recommendation_options,
                            index=recommendation_options.index(current_recommend),
                            key="review_recommend_value",
                            on_change=save_recommendation,
                            format_func=lambda val: str(val).title() if str(val).strip() else "(none)",
                        )
                with review_cols[1]:
                    current_status = st.session_state.get("review_status_value", "")
                    if current_status not in allowed_statuses:
                        current_status = ""
                        st.session_state["review_status_value"] = current_status
                    st.selectbox(
                        "Decision",
                        options=allowed_statuses,
                        index=allowed_statuses.index(current_status),
                        key="review_status_value",
                        on_change=save_status,
                        format_func=lambda val: str(val).title() if str(val).strip() else "(pending)",
                        help="Pick yes/maybe/no. Changes auto-save.",
                    )
                st.text_area(
                    "My notes",
                    key="review_notes_value",
                    on_change=save_notes,
                    height=220,
                    help="Free-form notes. Saving happens when you click outside the field.",
                )

            with interview_tab:
                st.caption("Interview tools sync instantly so multiple reviewers can collaborate.")
                interview_cols = st.columns(2)
                with interview_cols[0]:
                    st.caption("Structured prompts")
                    for prompt in interview_prompts:
                        slug = prompt.get("slug")
                        column_name = interview_prompt_cols.get(slug)
                        if not column_name or not slug:
                            continue
                        label = prompt.get("label", slug.title())
                        state_key = f"interview_prompt_{slug}_value"
                        st.text_area(
                            label,
                            key=state_key,
                            on_change=make_interview_prompt_saver(slug, column_name, label),
                            height=120,
                            help="Autosaves when you click outside the field.",
                        )
                    if interview_summary_template:
                        if st.button("Insert summary template", key="insert_summary_template"):
                            template_text = "\n".join(interview_summary_template)
                            st.session_state["interview_summary_value"] = template_text
                            st.toast("Template added to notes", icon="📝")
                            st.session_state["pending_notes_template"] = template_text
                            trigger_rerun()
                    st.text_area(
                        "Interview summary",
                        key="interview_summary_value",
                        on_change=save_interview_summary,
                        height=160,
                        help="Quick roll-up saved separately from free-form notes.",
                    )
                with interview_cols[1]:
                    st.caption("Live interview tools")
                    if user_interview_rating_col:
                        st.slider(
                            "Interview rating",
                            min_value=0.0,
                            max_value=5.0,
                            step=0.5,
                            key="interview_rating_value",
                            on_change=save_interview_rating,
                            help="Score the interview itself. Autosaves on release.",
                        )
                    outcome_options = [""] + [str(opt).strip() for opt in interview_outcome_options if str(opt).strip()]
                    if user_interview_outcome_col and outcome_options:
                        current_outcome = str(st.session_state.get("interview_outcome_value", "")).strip().lower()
                        current_outcome = LEGACY_OUTCOME_MAP.get(current_outcome, current_outcome)
                        if current_outcome not in outcome_options:
                            current_outcome = ""
                            st.session_state["interview_outcome_value"] = current_outcome
                        st.selectbox(
                            "Interview outcome",
                            options=outcome_options,
                            index=outcome_options.index(current_outcome),
                            key="interview_outcome_value",
                            on_change=save_interview_outcome,
                            format_func=lambda val: str(val).title() if str(val).strip() else "(undecided)",
                        )
                    st.text_area(
                        "Resource notes / links",
                        key="interview_resources_value",
                        on_change=save_interview_resources,
                        height=100,
                        help="Paste Zoom links, agendas, or reminders. Autosaves on blur.",
                    )
                    if interview_resource_links:
                        st.caption("Reference links")
                        for item in interview_resource_links:
                            label = item.get("label") or item.get("title") or item.get("url")
                            url = item.get("url")
                            if url:
                                st.markdown(f"- [{label}]({url})")
                            elif label:
                                st.markdown(f"- {label}")
                    st.text_input(
                        "Add quick note",
                        key="interview_log_entry",
                        placeholder="Capture a moment or quote...",
                    )
                    if st.button("Add timestamped note", key="add_interview_log"):
                        append_interview_log()
                    st.text_area(
                        "Interview log",
                        value=st.session_state.get("interview_log_value", ""),
                        height=150,
                        disabled=True,
                    )
                if interview_questions:
                    with st.expander("Question checklist", expanded=False):
                        for item in interview_questions:
                            slug = item.get("slug")
                            column_name = interview_question_cols.get(slug)
                            if not column_name or not slug:
                                continue
                            state_key = f"interview_question_{slug}_value"
                            current_value = st.session_state.get(state_key, "")
                            if current_value not in question_status_options:
                                current_value = ""
                                st.session_state[state_key] = current_value
                    st.selectbox(
                        item.get("question"),
                        options=question_status_options,
                        index=question_status_options.index(current_value),
                        key=state_key,
                        on_change=make_interview_question_saver(slug, column_name),
                        format_func=lambda val: question_status_labels.get(
                            val,
                            str(val).title() if str(val).strip() else "(pending)",
                        ),
                    )

            with feedback_tab:
                shared_note = str(display_row.get(notes_col, "") or "").strip()
                if shared_note:
                    st.markdown("**Shared note**")
                    st.write(shared_note)
                else:
                    st.caption("No shared note yet.")

                def _clean(value) -> str:
                    if pd.isna(value):
                        return ""
                    return str(value or "").strip()

                if all_reviewers:
                    st.markdown("**Reviewer notes**")
                    for reviewer, profile in all_reviewers.items():
                        note_col = f"{notes_col}__{reviewer}"
                        rating_col_user = f"{rating_base}__{reviewer}" if rating_base else None
                        recommend_col_user = f"{recommend_base}__{reviewer}" if recommend_base else None
                        note_text = _clean(display_row.get(note_col, ""))
                        rating_text = _clean(display_row.get(rating_col_user, "")) if rating_col_user else ""
                        recommend_text = _clean(display_row.get(recommend_col_user, "")) if recommend_col_user else ""
                        interview_summary_col_user = f"{interview_def['summary_field']}__{reviewer}"
                        interview_rating_col_user = f"{interview_def['rating_field']}__{reviewer}"
                        interview_outcome_col_user = f"{interview_def['outcome_field']}__{reviewer}"
                        summary_text = _clean(display_row.get(interview_summary_col_user, ""))
                        interviewer_rating_text = _clean(display_row.get(interview_rating_col_user, ""))
                        interview_outcome_text = _clean(display_row.get(interview_outcome_col_user, ""))
                        prompt_highlights = []
                        for prompt in interview_prompts:
                            slug = prompt.get("slug")
                            column_name = prompt.get("column")
                            if not column_name or not slug:
                                continue
                            value = display_row.get(f"{column_name}__{reviewer}", "")
                            if pd.isna(value) or not str(value).strip():
                                continue
                            prompt_highlights.append((prompt.get("label", slug.title()), str(value)))
                        display_name = profile.get("display_name", reviewer)
                        expanded = reviewer == current_user
                        with st.expander(display_name, expanded=expanded):
                            if rating_col_user:
                                st.write(f"Rating: {rating_text or '—'}")
                            if recommend_col_user:
                                rec_display = recommend_text.title() if recommend_text else "(none)"
                                st.write(f"Recommendation: {rec_display}")
                            if interviewer_rating_text:
                                st.write(f"Interview rating: {interviewer_rating_text}")
                            if interview_outcome_text:
                                st.write(f"Interview outcome: {interview_outcome_text.title()}")
                            if summary_text:
                                st.markdown("**Interview summary**")
                                st.write(summary_text)
                            if prompt_highlights:
                                st.markdown("**Interview notes**")
                                for label, value in prompt_highlights:
                                    st.write(f"{label}: {value}")
                            if note_text:
                                st.markdown("**General notes**")
                                st.write(note_text)
                            else:
                                st.caption("No notes yet.")
                else:
                    st.caption("No reviewer profiles configured.")

                st.divider()
                st.markdown("**Application responses**")
                for question in cfg.get("fields", {}).get("text_fields", []):
                    answer = str(display_row.get(question, "") or "").strip()
                    with st.expander(question, expanded=False):
                        if answer:
                            st.write(answer)
                        else:
                            st.caption("No response provided.")

    st.subheader("Potential Conflicts")
    analytics_cfg = cfg.get("analytics", {})
    rating_threshold = analytics_cfg.get("conflict_rating_threshold", 2.0) or 0
    recommend_conflicts_enabled = analytics_cfg.get("conflict_recommendation", True)
    conflict_mask = pd.Series(False, index=df.index)
    max_gap = pd.Series(0.0, index=df.index)
    recommend_summary = pd.Series('', index=df.index)
    if rating_base and rating_threshold and rating_threshold > 0:
        rating_cols = [col for col in df.columns if col.startswith(f"{rating_base}__")]
        if rating_cols:
            numeric = df[rating_cols].apply(pd.to_numeric, errors='coerce')
            max_gap = (numeric.max(axis=1) - numeric.min(axis=1)).fillna(0.0)
            conflict_mask = conflict_mask | (max_gap > rating_threshold)
    if recommend_conflicts_enabled:
        recommend_base = cfg["review"].get("recommendation_field", "review_recommendation")
        if recommend_base:
            recommend_cols = [col for col in df.columns if col.startswith(f"{recommend_base}__")]
            if recommend_cols:
                def _rec_mix(row):
                    vals = {
                        str(row[col]).strip().lower()
                        for col in recommend_cols
                        if isinstance(row[col], str) and str(row[col]).strip()
                    }
                    return ', '.join(sorted(vals)) if vals else ''
                recommend_summary = df.apply(_rec_mix, axis=1)
                rec_conflict = recommend_summary.apply(lambda v: 'yes' in v.split(', ') and 'no' in v.split(', '))
                conflict_mask = conflict_mask | rec_conflict
    conflict_df = df[conflict_mask].copy()
    if not conflict_df.empty:
        conflict_df['_rating_gap'] = max_gap[conflict_mask]
        conflict_df['_recommendation_mix'] = recommend_summary[conflict_mask]
        conflict_view_cols = [
            unique_key,
            'First Name',
            'Last Name',
        ]
        if score_col:
            conflict_view_cols.append(score_col)
        conflict_view_cols.extend([
            '_rating_gap',
            '_recommendation_mix',
            status_col,
        ])
        conflict_view_cols = [col for col in conflict_view_cols if col in conflict_df.columns]
        st.dataframe(conflict_df[conflict_view_cols], use_container_width=True, hide_index=True)
    else:
        st.caption('No rating or recommendation conflicts detected.')

    st.subheader("Analytics")
    status_counts = df[status_col].value_counts(dropna=False)
    if not status_counts.empty:
        st.bar_chart(status_counts)
    else:
        st.caption('No status data yet.')
    if rating_avg_col and rating_avg_col in df.columns:
        avg_series = pd.to_numeric(df[rating_avg_col], errors='coerce')
        st.metric('Average reviewer rating', f"{avg_series.dropna().mean():.2f}" if not avg_series.dropna().empty else '—')
    role_pref_summary = {}
    for role in ROLE_NAMES:
        pref_col = f"{role}__pref"
        if pref_col in df.columns:
            top_choice = (df[pref_col].astype(str).str.strip() == '1').sum()
            role_pref_summary[role] = int(top_choice)
    if role_pref_summary:
        st.write('Top-choice interest by role:')
        st.bar_chart(pd.Series(role_pref_summary))
    else:
        st.caption('No role preference data yet.')

    st.subheader("Role Shortlists")
    role_tabs = st.tabs(["All"] + ROLE_NAMES)
    with role_tabs[0]:
        display_cols = [
            unique_key,
            "First Name",
            "Last Name",
        ]
        if score_col:
            display_cols.append(score_col)
        display_cols.extend([
            ai_col,
            "ai_score",
            "gpa_flag",
            status_col,
            notes_col,
            "summary",
        ])
        display_cols = [col for col in display_cols if col in df.columns]
        if display_cols:
            ordered = df.copy()
            if "_fit_numeric" in ordered.columns:
                ordered.sort_values("_fit_numeric", ascending=False, inplace=True, ignore_index=True)
            st.dataframe(ordered[display_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No shortlist columns to show.")
    for idx, role in enumerate(ROLE_NAMES, start=1):
        with role_tabs[idx]:
            render_role_table(df, role, cfg)

    st.subheader("Activity Log")
    audit_df = fetch_audit_log(cfg["database"]["path"], limit=200)
    if audit_df.empty:
        st.caption("No review activity recorded yet.")
    else:
        st.dataframe(audit_df, use_container_width=True, hide_index=True)

with staffing_tab:
    if "staffing_css_loaded" not in st.session_state:
        st.markdown(
            """
            <style>
            .roster-card {
                border-radius: 10px;
                padding: 0.85rem;
                margin-bottom: 1rem;
                box-shadow: 0 2px 6px rgba(0,0,0,0.06);
                background: var(--backgroundColor-secondary);
            }
            .roster-card.green {border-left: 6px solid #2f9d5d;}
            .roster-card.yellow {border-left: 6px solid #f4a300;}
            .roster-card.red {border-left: 6px solid #d64545;}
            .roster-card h4 {margin: 0 0 0.35rem 0;}
            .roster-card ul {padding-left: 1.1rem; margin: 0;}
            .seat-grid {display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 0.75rem;}
            .seat-card {padding: 0.75rem; border-radius: 10px; border: 1px solid rgba(120,120,120,0.25); background: var(--backgroundColor-primary); min-height: 92px; display: flex; flex-direction: column; justify-content: space-between;}
            .seat-card.filled {border-left: 5px solid #2f9d5d;}
            .seat-card.empty {border-style: dashed; color: rgba(120,120,120,0.8); text-align: center;}
            .seat-card .badges {font-size: 0.8rem; color: rgba(120,120,120,0.9);}
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.session_state["staffing_css_loaded"] = True

    role_lookup = {role.lower(): role for role in ROLE_NAMES}

    def canonical_role(value: str) -> str:
        value = str(value or "").strip()
        if not value:
            return ""
        return role_lookup.get(value.lower(), value)

    st.subheader("Role Assignment Board")
    filter_cols = st.columns(4)
    with filter_cols[0]:
        assignment_role_filter = st.multiselect(
            "Filter by role",
            options=ROLE_NAMES,
            key="assignment_role_filter",
        )
    family_filter_options = family_names + ["Unassigned"]
    with filter_cols[1]:
        assignment_family_filter = st.multiselect(
            "Filter by family",
            options=family_filter_options,
            key="assignment_family_filter",
        )
    with filter_cols[2]:
        assignment_status_filter = st.multiselect(
            "Filter by status",
            options=sorted(df[status_col].dropna().astype(str).unique()),
            key="assignment_status_filter",
        )
    with filter_cols[3]:
        assignment_unassigned_only = st.checkbox(
            "Unassigned only",
            key="assignment_unassigned_only",
        )

    eligible_statuses = {"yes", "strong yes", "maybe"}
    if assignment_field not in df.columns:
        df[assignment_field] = ""
    assignment_subset = df[df[status_col].astype(str).str.lower().isin(eligible_statuses)].copy()
    if assignment_subset.empty:
        st.caption("No eligible applicants yet. Update review statuses to yes/strong yes/maybe to manage assignments.")
    else:
        assignment_subset[assignment_field] = assignment_subset[assignment_field].apply(canonical_role)
        if family_field in assignment_subset.columns:
            assignment_subset[family_field] = assignment_subset[family_field].apply(normalize_family)
        assignment_view = assignment_subset.copy()
        if assignment_role_filter:
            allowed_roles = {role.lower() for role in assignment_role_filter}
            assignment_view = assignment_view[
                assignment_view[assignment_field].astype(str).str.lower().isin(allowed_roles)
            ]
        if assignment_family_filter and family_field in assignment_view.columns:
            family_series = assignment_view[family_field].astype(str)
            masks = []
            selected_families = {name.lower() for name in assignment_family_filter if name != "Unassigned"}
            if selected_families:
                masks.append(family_series.str.lower().isin(selected_families))
            if "Unassigned" in assignment_family_filter:
                masks.append(family_series == "")
            if masks:
                combined_mask = masks[0]
                for mask in masks[1:]:
                    combined_mask = combined_mask | mask
                assignment_view = assignment_view[combined_mask]
        if assignment_status_filter:
            allowed_statuses = {status.lower() for status in assignment_status_filter}
            assignment_view = assignment_view[
                assignment_view[status_col].astype(str).str.lower().isin(allowed_statuses)
            ]
        if assignment_unassigned_only:
            assignment_view = assignment_view[assignment_view[assignment_field] == ""]

        assignment_columns = [
            unique_key,
            "First Name",
            "Last Name",
            status_col,
        ]
        if score_col:
            assignment_columns.append(score_col)
        assignment_columns.extend([
            assignment_field,
            family_field if family_field in assignment_view.columns else None,
        ])
        assignment_columns = [col for col in assignment_columns if col]
        board_df = assignment_view[assignment_columns].copy()
        if family_field not in board_df.columns:
            board_df[family_field] = ""
        rename_map = {
            "First Name": "First",
            "Last Name": "Last",
            status_col: "Status",
            assignment_field: "Assigned role",
            family_field: "Family",
        }
        if score_col:
            rename_map[score_col] = "Fit score"
        board_df.rename(columns=rename_map, inplace=True)
        board_df["Candidate"] = board_df["First"].fillna("") + " " + board_df["Last"].fillna("")
        board_df.drop(columns=["First", "Last"], inplace=True)
        column_order = [unique_key, "Candidate", "Status"]
        if score_col:
            column_order.append("Fit score")
        column_order.extend(["Assigned role", "Family"])
        board_df = board_df[column_order]
        board_df.set_index(unique_key, inplace=True)
        original_assignments = {
            key: canonical_role(value)
            for key, value in board_df["Assigned role"].to_dict().items()
        }
        st.caption("Use the dropdown to assign each candidate to a role. Leave blank to keep unassigned.")
        edited_df = st.data_editor(
            board_df,
            hide_index=False,
            key="assignment_editor",
            column_config={
                **({"Fit score": st.column_config.NumberColumn(format="%.2f")} if score_col else {}),
                "Assigned role": st.column_config.SelectboxColumn(
                    "Assigned role",
                    options=[""] + ROLE_NAMES,
                    help="Choose a target role or leave blank to keep unassigned.",
                ),
                "Family": st.column_config.TextColumn("Family", disabled=True),
            },
        )
        edited_df["Assigned role"] = edited_df["Assigned role"].apply(canonical_role)
        counts = {role: int((edited_df["Assigned role"].str.lower() == role.lower()).sum()) for role in ROLE_NAMES}
        counts["Unassigned"] = int((edited_df["Assigned role"] == "").sum())
        summary_cols = st.columns(len(counts)) if counts else []
        for col, (label, value) in zip(summary_cols, counts.items()):
            col.metric(label, value)
        edited_assignments = edited_df["Assigned role"].to_dict()
        updates = {
            key: value
            for key, value in edited_assignments.items()
            if original_assignments.get(key, "") != value
        }
        if updates:
            with st.spinner("Saving assignments..."):
                for key, value in updates.items():
                    update_review(
                        db_path=cfg["database"]["path"],
                        table=cfg["database"]["table"],
                        unique_key=unique_key,
                        keyval=key,
                        updates={assignment_field: value},
                        csv_path=csv_write_target if cfg["csv"].get("write_back", True) else None,
                        skiprows=cfg["csv"].get("skiprows", 0),
                        actor=current_user,
                    )
                snapshot_review_data(cfg, event="assignment-update")
                st.session_state["data_version"] += 1
                st.session_state["assignment_message"] = f"Updated {len(updates)} assignments."
                load_dataframe.clear()
                trigger_rerun()

    st.divider()
    st.subheader("Role Rosters")
    role_targets = {role: max(role_targets_cfg.get(role, 0), 0) for role in ROLE_NAMES}
    roster_columns = st.columns(min(len(ROLE_NAMES), 3)) if ROLE_NAMES else []
    for idx, role in enumerate(ROLE_NAMES):
        column = roster_columns[idx % len(roster_columns)] if roster_columns else st
        with column:
            members = df[df[assignment_field].astype(str).str.lower() == role.lower()].copy()
            count = len(members)
            target = role_targets.get(role, count) or count
            if count == target:
                tone = "green"
            elif count < target:
                tone = "yellow" if target - count <= 2 else "red"
            else:
                tone = "red"
            st.markdown(f"<div class='roster-card {tone}'><h4>{role}</h4><p><strong>{count}</strong> of <strong>{target}</strong> spots filled</p>", unsafe_allow_html=True)
            if count:
                members_sorted = (
                    members.sort_values(score_col, ascending=False)
                    if score_col and score_col in members.columns
                    else members
                )
                items = []
                for _, row in members_sorted.iterrows():
                    badges = []
                    if str(row.get(ai_col, '')).upper() == 'YES':
                        badges.append('🤖 AI')
                    if str(row.get('gpa_flag', '')).upper() == 'LOW':
                        badges.append('📉 GPA')
                    badge_text = f" ({', '.join(badges)})" if badges else ''
                    items.append(f"<li>{row.get('First Name', '')} {row.get('Last Name', '')}{badge_text}</li>")
                st.markdown('<ul>' + ''.join(items) + '</ul></div>', unsafe_allow_html=True)
            else:
                st.markdown("<p style='margin:0;'>No assignments yet.</p></div>", unsafe_allow_html=True)

    st.divider()

    st.subheader("Candidate Quick View")
    profile_options = {
        row[unique_key]: f"{row.get('First Name', '')} {row.get('Last Name', '')} • {row.get(assignment_field, '') or 'Unassigned'}"
        for _, row in df.sort_values("Last Name", na_position="last").iterrows()
    }
    selected_profile = st.selectbox(
        "Select candidate",
        options=[None] + list(profile_options.keys()),
        format_func=lambda val: profile_options.get(val, "(select)") if val else "(select)",
        key="staffing_profile_select",
    )
    if selected_profile:
        profile_row = df[df[unique_key] == selected_profile].iloc[0]
        profile_cols = st.columns([2, 1])
        with profile_cols[0]:
            st.markdown(
                f"### {profile_row.get('First Name', '')} {profile_row.get('Last Name', '')}"
            )
            st.write(f"**Status:** {profile_row.get(status_col, '(none)')}")
            role_preferences = []
            for role in ROLE_NAMES:
                pref_val = profile_row.get(f"{role}__pref", "")
                role_preferences.append(f"{role}: {pref_val or '—'}")
            st.write("**Role rankings:** " + ", ".join(role_preferences))
            if assignment_field in profile_row:
                st.write(f"**Assigned role:** {profile_row.get(assignment_field, '—') or '—'}")
            if family_field in profile_row:
                st.write(f"**Family:** {profile_row.get(family_field, '—') or '—'}")
            if family_pair_field in profile_row:
                st.write(f"**Family pod:** {profile_row.get(family_pair_field, '—') or '—'}")
        with profile_cols[1]:
            if score_col:
                st.metric("Fit score", profile_row.get(score_col, '—'))
            st.metric("AI flag", profile_row.get(ai_col, '—'))
            st.metric("GPA flag", profile_row.get("gpa_flag", '—'))
        shared_note = str(profile_row.get(notes_col, "") or "").strip()
        if shared_note:
            st.markdown("**Shared note**")
            st.write(shared_note)
        personal_note = str(profile_row.get(user_note_col, "") or "").strip() if user_note_col else ""
        if personal_note:
            st.markdown("**My notes**")
            st.write(personal_note)
        interview_summary = str(profile_row.get(user_interview_summary_col, "") or "").strip() if user_interview_summary_col else ""
        if interview_summary:
            st.markdown("**Interview summary**")
            st.write(interview_summary)
        st.divider()

    st.divider()
    final_statuses = {"yes", "strong yes"}
    if family_field not in df.columns:
        df[family_field] = ""
    else:
        df[family_field] = df[family_field].apply(normalize_family)
    if family_pair_field and family_pair_field not in df.columns:
        df[family_pair_field] = ""
    family_subset = df[df[status_col].astype(str).str.lower().isin(final_statuses)].copy()

    st.subheader("Family Assignment Board")
    if family_subset.empty:
        st.caption("Once applicants are marked yes/strong yes they will appear here for family grouping.")
    else:
        family_columns = [
            unique_key,
            "First Name",
            "Last Name",
            status_col,
        ]
        if score_col:
            family_columns.append(score_col)
        family_columns.extend([
            assignment_field,
            family_field,
        ])
        if family_pair_field and family_pair_field in family_subset.columns:
            family_columns.append(family_pair_field)
        family_board = family_subset[family_columns].copy()
        rename_map = {
            "First Name": "First",
            "Last Name": "Last",
            status_col: "Status",
            assignment_field: "Assigned role",
            family_field: "Family",
        }
        if family_pair_field and family_pair_field in family_columns:
            rename_map[family_pair_field] = "Family pod"
        family_board.rename(columns=rename_map, inplace=True)
        family_board["Candidate"] = family_board["First"].fillna("") + " " + family_board["Last"].fillna("")
        family_board.drop(columns=["First", "Last"], inplace=True)
        family_board["Family"] = family_board["Family"].apply(normalize_family)
        if "Family pod" in family_board.columns:
            family_board["Family pod"] = family_board["Family pod"].apply(lambda v: str(v or "").strip())
        else:
            family_board["Family pod"] = ""
        column_order = [unique_key, "Candidate", "Status", "Assigned role", "Family", "Family pod"]
        family_board = family_board[column_order]
        family_board.set_index(unique_key, inplace=True)

        family_filter_col, _ = st.columns([1, 3])
        with family_filter_col:
            pairing_family_filter = st.selectbox(
                "Highlight family",
                options=["All"] + family_names,
                key="pairing_family_filter",
            )
        pairing_view = family_board.copy()
        if pairing_family_filter != "All":
            pairing_view = pairing_view[pairing_view["Family"].str.lower() == pairing_family_filter.lower()]

        original_family = {key: row["Family"] for key, row in pairing_view.iterrows()}
        original_pod = {key: row["Family pod"] for key, row in pairing_view.iterrows()}

        edited_family = st.data_editor(
            pairing_view,
            hide_index=False,
            key="family_assignment_editor",
            column_config={
                **({"Fit score": st.column_config.NumberColumn(format="%.2f")} if score_col else {}),
                "Assigned role": st.column_config.TextColumn("Assigned role", disabled=True),
                "Family": st.column_config.SelectboxColumn("Family", options=family_names, help="Choose the family group for this staff member."),
                "Family pod": st.column_config.TextColumn(
                    "Family pod",
                    help="Label pairings or pods within the family (e.g., Counselor Pod 1).",
                ),
            },
        )
        edited_family["Family"] = edited_family["Family"].apply(normalize_family)
        edited_family["Family pod"] = edited_family["Family pod"].apply(lambda v: str(v or "").strip())

        combined_updates = {}
        for key, row in edited_family.iterrows():
            payload = {}
            new_family = row.get("Family", "")
            new_pod = row.get("Family pod", "")
            if original_family.get(key, "") != new_family:
                payload[family_field] = new_family
            if family_pair_field and original_pod.get(key, "") != new_pod:
                payload[family_pair_field] = new_pod
            if payload:
                combined_updates[key] = payload

        if combined_updates:
            with st.spinner("Saving family placements..."):
                for key, update_payload in combined_updates.items():
                    update_review(
                        db_path=cfg["database"]["path"],
                        table=cfg["database"]["table"],
                        unique_key=unique_key,
                        keyval=key,
                        updates=update_payload,
                        csv_path=csv_write_target if cfg["csv"].get("write_back", True) else None,
                        skiprows=cfg["csv"].get("skiprows", 0),
                        actor=current_user,
                    )
                snapshot_review_data(cfg, event="family-update")
                st.session_state["data_version"] += 1
                st.session_state["family_message"] = f"Updated {len(combined_updates)} family placements."
                load_dataframe.clear()
                trigger_rerun()

    st.divider()
    st.subheader("Family Seat Map")
    if family_subset.empty:
        st.caption("Once applicants are marked yes/strong yes they will appear here for family grouping.")
    else:
        for family_name in family_names:
            members = family_subset[family_subset[family_field].astype(str).str.lower() == family_name.lower()].copy()
            capacity = max(len(members), family_capacity_map.get(family_name, family_default_capacity))
            st.markdown(f"### {family_name} ({len(members)}/{capacity})")
            if score_col and score_col in members.columns:
                members.sort_values(score_col, ascending=False, inplace=True)
            seat_cards = []
            accent_color = family_color_map.get(family_name, family_color_map.get(family_name.lower(), "#2f9d5d"))
            for _, row in members.iterrows():
                badges = []
                if str(row.get(ai_col, '')).upper() == 'YES':
                    badges.append('🤖')
                if str(row.get('gpa_flag', '')).upper() == 'LOW':
                    badges.append('📉')
                assignment_badge = format_role_badge(row.get(assignment_field, ''))
                pod_badge = str(row.get(family_pair_field, "").strip()) if family_pair_field in row else ""
                badge_text = ' '.join(badges) + (f" {assignment_badge}" if assignment_badge != '—' else '')
                if pod_badge:
                    badge_text = f"{badge_text} | Pod: {pod_badge}" if badge_text else f"Pod: {pod_badge}"
                seat_cards.append(
                    f"<div class='seat-card filled' style='border-left: 5px solid {accent_color};'>"
                    f"<div><strong>{row.get('First Name', '')} {row.get('Last Name', '')}</strong></div>"
                    f"<div class='badges'>{badge_text.strip() or '—'}</div></div>"
                )
            for _ in range(capacity - len(members)):
                seat_cards.append(
                    f"<div class='seat-card empty' style='border-left: 5px dashed {accent_color}; color: rgba(120,120,120,0.8);'>Open Seat</div>"
                )
            st.markdown("<div class='seat-grid'>" + ''.join(seat_cards) + "</div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("Family vs Role Matrix")
    if family_subset.empty:
        st.caption("No confirmed staff yet.")
    else:
        heatmap_df = pd.crosstab(
            family_subset[family_field].apply(normalize_family),
            family_subset[assignment_field].apply(canonical_role),
        )
        if heatmap_df.empty:
            st.caption("No role assignments yet.")
        else:
            st.dataframe(heatmap_df, use_container_width=True)
