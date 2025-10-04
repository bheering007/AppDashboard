import sqlite3
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


@st.cache_resource
def get_cfg() -> Dict:
    with open("config.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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
    min_fit: float,
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
    if "_fit_numeric" in filtered.columns:
        filtered = filtered[filtered["_fit_numeric"].fillna(-999) >= min_fit]
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
    if '_fit_numeric' in filtered.columns:
        sort_cols.append('_fit_numeric')
        sort_orders.append(False)
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
    score_col = cfg["review"]["rubric_score_field"]
    status_col = cfg["review"]["status_field"]
    recommend_base = cfg["review"].get("recommendation_field", "review_recommendation")
    display_cols = [
        cfg["database"]["unique_key"],
        "First Name",
        "Last Name",
        score_col,
        f"{role}__fit",
        f"{role}__pref",
        status_col,
        cfg["review"]["notes_field"],
        cfg["review"]["ai_flag_field"],
        "gpa_flag",
    ]
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
score_col = cfg["review"]["rubric_score_field"]
rating_base = cfg["review"].get("rating_field", "review_score")
recommend_base = cfg["review"].get("recommendation_field", "review_recommendation")
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
pending_mask_all = reviewer_pending_mask(df, reviewer_personal_cols)
badge_map = badge_columns(df)

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
        st.session_state["last_uploaded_token"] = token
        st.session_state["upload_message"] = f"Imported {len(enriched)} rows from {file.name}."
        st.session_state["data_version"] += 1
        load_dataframe.clear()
        trigger_rerun()

    process_upload(upload)

    st.divider()
    st.header("Filter")
    search_text = st.text_input("Search name/email/notes")
    status_filter = st.multiselect(
        "Review status",
        options=cfg["review"].get("allowed_statuses", ["yes", "maybe", "no"]),
        help="Filter by your decisions. Leave empty to show all."
    )
    ai_filter = st.selectbox("AI flag", options=["All", "Flagged", "Not flagged"], index=0)
    gpa_only = st.checkbox("Only show GPA flag (<3.0)", value=False)
    min_fit = st.slider("Minimum fit score", -2.0, 6.0, 0.0, 0.1)
    selected_badges = st.multiselect(
        "Must include badges", options=list(badge_map.keys()), format_func=lambda key: badge_map[key]
    )
    role_focus = st.selectbox("Role focus", options=["All"] + ROLE_NAMES)
    role_pref_rule = st.selectbox("Preference filter", options=["Any", "#1 only", "#1 or #2"], index=0)
    pending_only = st.checkbox(
        "Only show my pending reviews",
        key="pending_only_filter",
        help="Show applications where you have not saved notes, rating, or recommendation yet.",
    )

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
    min_fit=min_fit,
    badge_filter=selected_badges,
    role_focus=role_focus,
    role_pref_rule=role_pref_rule,
)
pending_only_active = st.session_state.get("pending_only_filter", False)
if pending_only_active and not filtered_df.empty:
    filtered_df = filtered_df[reviewer_pending_mask(filtered_df, reviewer_personal_cols)]

unique_key = cfg["database"]["unique_key"]

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
    score_col,
]
if rating_avg_col:
    default_table_cols.append(rating_avg_col)
default_table_cols.extend([
    ai_col,
    "ai_score",
    "gpa_flag",
    status_col,
    notes_col,
    "summary",
])
available_cols = [col for col in default_table_cols if col in filtered_df.columns]
selected_table_cols = st.multiselect(
    "Table columns",
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
            interview_outcome_value = str(outcome_raw).strip().lower()
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

        info_cols = st.columns([1, 1])
        with info_cols[0]:
            st.write("**Name**", f"{selected_row.get('First Name', '')} {selected_row.get('Last Name', '')}")
            for contact in cfg.get("fields", {}).get("contact_fields", []):
                if contact in selected_row:
                    st.write(f"**{contact}**", selected_row.get(contact, ""))
            st.write("**Fit score**", selected_row.get(score_col, ""))
            st.write(
                "**AI suspect**",
                f"{selected_row.get(ai_col, '')} ({selected_row.get('ai_score', '')})",
            )
            gpa_flag = selected_row.get("gpa_flag", "")
            if gpa_flag:
                st.error("GPA flagged below threshold", icon="??")
            render_badges(selected_row, badge_map)
            st.write("**Summary**")
            st.write(selected_row.get("summary", ""))
        with info_cols[1]:
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
                interview_rating_state = st.session_state.get("interview_rating_display", "") if user_interview_rating_col else ""
                interview_outcome_state = st.session_state.get("interview_outcome_value", "") if user_interview_outcome_col else ""
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
                update_review(
                    db_path=cfg["database"]["path"],
                    table=cfg["database"]["table"],
                    unique_key=unique_key,
                    keyval=selected_id,
                    updates=updates,
                    csv_path=csv_write_target,
                    skiprows=cfg["csv"].get("skiprows", 0),
                    actor=current_user,
                )
                st.session_state["review_snapshot"] = current_snapshot()
                st.session_state["data_version"] += 1
                load_dataframe.clear()
                if toast_message:
                    st.toast(toast_message, icon="??")

            def save_rating() -> None:
                if not user_rating_col:
                    return
                value = float(st.session_state.get("review_rating_value", 0.0))
                value = max(0.0, min(5.0, value))
                formatted = f"{value:.1f}"
                st.session_state["review_rating_str"] = formatted
                persist_updates({user_rating_col: formatted}, "Rating saved")

            if user_rating_col:
                st.slider(
                    "My reviewer score",
                    min_value=0.0,
                    max_value=5.0,
                    step=0.5,
                    key="review_rating_value",
                    on_change=save_rating,
                    help="0 = not ready / 5 = outstanding. Autosaves when you release the slider.",
                )

            if user_recommend_col:
                def save_recommendation() -> None:
                    value = st.session_state.get("review_recommend_value", "")
                    if isinstance(value, str):
                        value = value.lower().strip()
                    persist_updates({user_recommend_col: value}, "Recommendation saved")

                rec_options = ["", "yes", "maybe", "no"]
                current_value = st.session_state.get("review_recommend_value", "")
                if current_value not in rec_options:
                    current_value = ""
                    st.session_state["review_recommend_value"] = current_value
                st.selectbox(
                    "My recommendation",
                    options=rec_options,
                    index=rec_options.index(current_value),
                    key="review_recommend_value",
                    on_change=save_recommendation,
                    format_func=lambda val: val.title() if val else "(none)",
                    help="Share your personal yes/maybe/no recommendation. Autosaves on change.",
                )

            def save_status() -> None:
                value = st.session_state.get("review_status_value", "")
                persist_updates({status_col: value}, "Status saved")

            def save_notes() -> None:
                value = st.session_state.get("review_notes_value", "")
                persist_updates({user_note_col: value}, "Notes saved")

            st.selectbox(
                "Decision",
                options=allowed_statuses,
                index=allowed_statuses.index(st.session_state.get("review_status_value", ""))
                if st.session_state.get("review_status_value", "") in allowed_statuses
                else 0,
                key="review_status_value",
                on_change=save_status,
                help="Pick yes/maybe/no. Changes auto-save.",
            )
            st.text_area(
                "My notes",
                key="review_notes_value",
                on_change=save_notes,
                height=180,
                help="Free-form notes. Saving happens when you click outside the field.",
            )

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
                value = st.session_state.get("interview_outcome_value", "")
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

            st.divider()
            st.subheader("Interview workspace")
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
                        st.session_state["review_notes_value"] = template_text
                        st.session_state["review_snapshot"] = current_snapshot()
                        st.toast("Template added to notes", icon="??")
                st.text_area(
                    "Interview summary",
                    key="interview_summary_value",
                    on_change=save_interview_summary,
                    height=140,
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
                outcome_options = [""] + [opt for opt in interview_outcome_options if opt]
                if user_interview_outcome_col and outcome_options:
                    current_outcome = st.session_state.get("interview_outcome_value", "")
                    if current_outcome not in outcome_options:
                        current_outcome = ""
                        st.session_state["interview_outcome_value"] = current_outcome
                    st.selectbox(
                        "Interview outcome",
                        options=outcome_options,
                        index=outcome_options.index(current_outcome),
                        key="interview_outcome_value",
                        on_change=save_interview_outcome,
                        format_func=lambda val: val.title() if val else "(undecided)",
                    )
                st.text_area(
                    "Resource notes / links",
                    key="interview_resources_value",
                    on_change=save_interview_resources,
                    height=80,
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
                            format_func=lambda val: question_status_labels.get(val, val.title() if val else "(pending)"),
                        )
        st.markdown("**Reviewer notes & ratings**")
        shared_note = selected_row.get(notes_col, "")
        if shared_note:
            st.caption("Shared note")
            st.write(shared_note)

for reviewer, profile in all_reviewers.items():
    note_col = f"{notes_col}__{reviewer}"
    rating_col_user = f"{rating_base}__{reviewer}" if rating_base else None
    recommend_col_user = f"{recommend_base}__{reviewer}" if recommend_base else None
    note_text = selected_row.get(note_col, "")
    rating_text = selected_row.get(rating_col_user, "") if rating_col_user else ""
    recommend_text = selected_row.get(recommend_col_user, "") if recommend_col_user else ""
    interview_summary_col_user = f"{interview_def['summary_field']}__{reviewer}"
    interview_rating_col_user = f"{interview_def['rating_field']}__{reviewer}"
    interview_outcome_col_user = f"{interview_def['outcome_field']}__{reviewer}"
    summary_text = selected_row.get(interview_summary_col_user, "")
    interviewer_rating_text = selected_row.get(interview_rating_col_user, "")
    interview_outcome_text = selected_row.get(interview_outcome_col_user, "")
    prompt_highlights = []
    for prompt in interview_prompts:
        slug = prompt.get("slug")
        column_name = prompt.get("column")
        if not column_name or not slug:
            continue
        value = selected_row.get(f"{column_name}__{reviewer}", "")
        if pd.isna(value) or not str(value).strip():
            continue
        prompt_highlights.append((prompt.get("label", slug.title()), str(value)))
    if pd.isna(note_text):
        note_text = ""
    if pd.isna(rating_text):
        rating_text = ""
    if pd.isna(recommend_text):
        recommend_text = ""
    if pd.isna(summary_text):
        summary_text = ""
    if pd.isna(interviewer_rating_text):
        interviewer_rating_text = ""
    if pd.isna(interview_outcome_text):
        interview_outcome_text = ""
    display = profile.get("display_name", reviewer)
    with st.expander(display, expanded=(reviewer == current_user)):
        if rating_col_user and rating_text:
            st.write(f"Rating: {rating_text}")
        elif rating_col_user:
            st.caption("No rating yet.")
        if recommend_col_user:
            st.write(
                f"Recommendation: {recommend_text.title()}" if recommend_text else "Recommendation: (none)"
            )
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
        st.divider()
        st.subheader("Responses")
        for question in cfg.get("fields", {}).get("text_fields", []):
            answer = selected_row.get(question, "")
            with st.expander(question, expanded=False):
                if str(answer).strip():
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
        score_col,
        '_rating_gap',
        '_recommendation_mix',
        status_col,
    ]
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
        score_col,
        ai_col,
        "ai_score",
        "gpa_flag",
        status_col,
        notes_col,
        "summary",
    ]
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
