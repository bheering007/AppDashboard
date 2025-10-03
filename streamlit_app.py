import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st
import yaml

from app_review import ROLE_NAMES, import_csv_to_db, update_review

st.set_page_config(page_title="NMC Applications", layout="wide")

BADGE_PREFIX = "badge__"




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
            st.experimental_rerun()
        else:
            session["_auth_error"] = "Invalid username or password."
            st.experimental_rerun()
    if session.get("_auth_error"):
        st.error(session["_auth_error"])
    st.stop()


@st.cache_resource
def get_cfg() -> Dict:
    with open("config.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@st.cache_data(show_spinner=False)
def load_dataframe(db_path: str, table: str, order_by: str | None, version: int) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
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
            filtered = filtered[filtered[pref_col].fillna("") != ""]
            if role_pref_rule == "#1 only":
                filtered = filtered[filtered[pref_col] == "1"]
            elif role_pref_rule == "#1 or #2":
                filtered = filtered[filtered[pref_col].isin(["1", "2"])]
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
        data = data[data[pref_col].fillna("") != ""]
    if data.empty:
        st.info("No applicants ranked this role yet.")
        return
    if f"{role}__fit" in data.columns:
        data["_role_fit_numeric"] = pd.to_numeric(data[f"{role}__fit"], errors="coerce")
        data.sort_values("_role_fit_numeric", ascending=False, inplace=True, ignore_index=True)
    st.dataframe(data[display_cols], use_container_width=True, hide_index=True)


cfg = get_cfg()
current_user, current_profile = login(cfg.get("auth", {}))
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
raw_df = load_dataframe(cfg["database"]["path"], cfg["database"]["table"], order_by, st.session_state["data_version"])
df = prepare_dataframe(raw_df, cfg)
badge_map = badge_columns(df)

with st.sidebar:
    display_name = st.session_state.get("_auth_display", current_profile.get("display_name", current_user))
    st.markdown(f"**Logged in as:** {display_name}")
    if st.button("Log out", key="logout_btn"):
        for key in ["_auth_user", "_auth_display", "data_version", "active_submission", "review_status_value", "review_notes_value", "review_rating_value", "review_rating_str", "review_snapshot", "upload_message", "last_uploaded_token"]:
            st.session_state.pop(key, None)
        st.experimental_rerun()
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
        st.experimental_rerun()

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

status_col = cfg["review"]["status_field"]
notes_col = cfg["review"]["notes_field"]
ai_col = cfg["review"]["ai_flag_field"]
score_col = cfg["review"]["rubric_score_field"]
unique_key = cfg["database"]["unique_key"]

col_metrics = st.columns(4)
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

st.subheader("Review Panel")
if filtered_df.empty:
    st.info("No applicants match the current filters.")
else:
    option_labels = {
        row[unique_key]: format_option_label(row, cfg) for _, row in filtered_df.iterrows()
    }
    selected_id = st.selectbox(
        "Choose applicant",
        options=list(option_labels.keys()),
        format_func=lambda sid: option_labels.get(sid, str(sid)),
    )
    if selected_id:
        selected_row = df[df[unique_key] == selected_id].iloc[0]
        user_note_col = f"{notes_col}__{current_user}"
        note_value = selected_row.get(user_note_col, "")
        if pd.isna(note_value):
            note_value = ""
        rating_base = cfg["review"].get("rating_field", "review_score")
        user_rating_col = f"{rating_base}__{current_user}" if rating_base else None
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
        snapshot = (
            selected_row.get(status_col, ""),
            note_value,
            rating_str,
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
                st.error("GPA flagged below threshold", icon="⚠️")
            render_badges(selected_row, badge_map)
            st.write("**Summary**")
            st.write(selected_row.get("summary", ""))
        with info_cols[1]:
            allowed_statuses = [""] + cfg["review"].get("allowed_statuses", [])

            def current_snapshot():
                rating_display = st.session_state.get("review_rating_str", "") if rating_base else ""
                return (
                    st.session_state.get("review_status_value", ""),
                    st.session_state.get("review_notes_value", ""),
                    rating_display,
                )

            def save_rating() -> None:
                if not user_rating_col:
                    return
                value = float(st.session_state.get("review_rating_value", 0.0))
                value = max(0.0, min(5.0, value))
                updates = {user_rating_col: f"{value:.1f}"}
                update_review(
                    db_path=cfg["database"]["path"],
                    table=cfg["database"]["table"],
                    unique_key=unique_key,
                    keyval=selected_id,
                    updates=updates,
                    csv_path=csv_write_target,
                    skiprows=cfg["csv"].get("skiprows", 0),
                )
                st.session_state["review_rating_str"] = f"{value:.1f}"
                st.session_state["review_snapshot"] = current_snapshot()
                st.session_state["data_version"] += 1
                load_dataframe.clear()
                st.toast("Rating saved", icon="💾")

            if user_rating_col:
                st.slider(
                    "My reviewer score",
                    min_value=0.0,
                    max_value=5.0,
                    step=0.5,
                    key="review_rating_value",
                    on_change=save_rating,
                    help="0 = not ready • 5 = outstanding. Autosaves when you release the slider.",
                )

            def save_status() -> None:
                value = st.session_state.get("review_status_value", "")
                updates = {status_col: value}
                update_review(
                    db_path=cfg["database"]["path"],
                    table=cfg["database"]["table"],
                    unique_key=unique_key,
                    keyval=selected_id,
                    updates=updates,
                    csv_path=csv_write_target,
                    skiprows=cfg["csv"].get("skiprows", 0),
                )
                st.session_state["review_snapshot"] = current_snapshot()
                st.session_state["data_version"] += 1
                load_dataframe.clear()
                st.toast("Status saved", icon="💾")

            def save_notes() -> None:
                value = st.session_state.get("review_notes_value", "")
                updates = {user_note_col: value}
                update_review(
                    db_path=cfg["database"]["path"],
                    table=cfg["database"]["table"],
                    unique_key=unique_key,
                    keyval=selected_id,
                    updates=updates,
                    csv_path=csv_write_target,
                    skiprows=cfg["csv"].get("skiprows", 0),
                )
                st.session_state["review_snapshot"] = current_snapshot()
                st.session_state["data_version"] += 1
                load_dataframe.clear()
                st.toast("Notes saved", icon="💾")

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
        st.markdown("**Reviewer notes & ratings**")
        shared_note = selected_row.get(notes_col, "")
        if shared_note:
            st.caption("Shared note")
            st.write(shared_note)
        for reviewer, profile in all_reviewers.items():
            note_col = f"{notes_col}__{reviewer}"
            rating_col_user = f"{rating_base}__{reviewer}" if rating_base else None
            note_text = selected_row.get(note_col, "")
            rating_text = selected_row.get(rating_col_user, "") if rating_col_user else ""
            if pd.isna(note_text):
                note_text = ""
            if pd.isna(rating_text):
                rating_text = ""
            display = profile.get("display_name", reviewer)
            with st.expander(display, expanded=(reviewer == current_user)):
                if rating_col_user and rating_text:
                    st.write(f"Rating: {rating_text}")
                elif rating_col_user:
                    st.caption("No rating yet.")
                if note_text:
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
