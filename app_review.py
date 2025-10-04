"""Utilities for importing and annotating New Maroon Camp applications."""

from __future__ import annotations

import re
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

ROLE_NAMES: List[str] = ["Counselor", "Buddy Staff", "Programming Staff"]

DEFAULT_BADGE_THRESHOLDS = {
    "yes": 2,
    "maybe": 1,
}



DEFAULT_INTERVIEW_PROMPTS = {
    "strengths": "Strengths",
    "concerns": "Concerns",
    "growth": "Growth Opportunities",
    "followups": "Follow-up Items",
}
DEFAULT_INTERVIEW_OUTCOMES = ["strong yes", "yes", "hold", "no"]
DEFAULT_INTERVIEW_RATING_FIELD = "interview_rating"
DEFAULT_INTERVIEW_OUTCOME_FIELD = "interview_outcome"
DEFAULT_INTERVIEW_LOG_FIELD = "interview_log"
DEFAULT_INTERVIEW_RESOURCES_FIELD = "interview_resources"
DEFAULT_INTERVIEW_SUMMARY_FIELD = "interview_summary"

def get_reviewer_profiles(cfg: Dict) -> Dict[str, Dict]:
    return cfg.get("auth", {}).get("users", {}) or {}

def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


# ---------------------------------------------------------------------------
# Database utilities
# ---------------------------------------------------------------------------


def get_interview_definition(cfg: Dict) -> Dict[str, Dict]:
    interview_cfg = cfg.get("interview", {}) or {}
    prompts_cfg = interview_cfg.get("prompts", {}) or {}
    prompt_items = []
    used_slugs = set()
    for key, default_label in DEFAULT_INTERVIEW_PROMPTS.items():
        label = prompts_cfg.get(key, default_label)
        slug = slugify(key) or slugify(label)
        if not slug or slug in used_slugs:
            continue
        prompt_items.append({"key": key, "label": label, "slug": slug, "column": f"interview_{slug}"})
        used_slugs.add(slug)
    for key, label in prompts_cfg.items():
        slug = slugify(key) or slugify(label)
        if not slug or slug in used_slugs:
            continue
        prompt_items.append({"key": key, "label": label, "slug": slug, "column": f"interview_{slug}"})
        used_slugs.add(slug)
    rating_field = interview_cfg.get("rating_field", DEFAULT_INTERVIEW_RATING_FIELD)
    outcome_field = interview_cfg.get("outcome_field", DEFAULT_INTERVIEW_OUTCOME_FIELD)
    log_field = interview_cfg.get("log_field", DEFAULT_INTERVIEW_LOG_FIELD)
    resources_field = interview_cfg.get("resources_field", DEFAULT_INTERVIEW_RESOURCES_FIELD)
    summary_field = interview_cfg.get("summary_field", DEFAULT_INTERVIEW_SUMMARY_FIELD)
    outcome_options = interview_cfg.get("outcome_options", DEFAULT_INTERVIEW_OUTCOMES)
    summary_template = interview_cfg.get("summary_template", [])
    questions_cfg = interview_cfg.get("question_checklist", []) or []
    question_items = []
    question_slugs = set()
    for question in questions_cfg:
        slug = slugify(question)
        if not slug or slug in question_slugs:
            continue
        question_items.append({
            "question": question,
            "slug": slug,
            "column": f"interview_q_{slug}",
        })
        question_slugs.add(slug)
    return {
        "prompts": prompt_items,
        "rating_field": rating_field,
        "outcome_field": outcome_field,
        "log_field": log_field,
        "resources_field": resources_field,
        "summary_field": summary_field,
        "outcome_options": outcome_options,
        "summary_template": summary_template,
        "questions": question_items,
    }


def init_db(db_path: str, table: str, columns: Iterable[str], unique_key: str) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    col_defs = []
    for col in columns:
        decl = '"{col}" TEXT'.format(col=col)
        if col == unique_key:
            decl += " PRIMARY KEY"
        col_defs.append(decl)
    cur.execute(
        'CREATE TABLE IF NOT EXISTS "{table}" ({cols})'.format(
            table=table, cols=", ".join(col_defs)
        )
    )
    conn.commit()
    conn.close()


def ensure_columns(db_path: str, table: str, columns: Iterable[str]) -> None:
    unique_columns = list(dict.fromkeys(columns))
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(f'PRAGMA table_info("{table}")')
    existing = {row[1] for row in cur.fetchall()}
    to_add = [col for col in unique_columns if col not in existing]
    for col in to_add:
        cur.execute(f'ALTER TABLE "{table}" ADD COLUMN "{col}" TEXT')
    if to_add:
        conn.commit()
    conn.close()


def _normalize_value(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):  # type: ignore[arg-type]
        return None
    if isinstance(value, str):
        if value.strip() == "":
            return None
        return value
    return str(value)


def _is_empty_value(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):  # type: ignore[arg-type]
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def upsert_rows(db_path: str, table: str, df: pd.DataFrame, unique_key: str) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    ensure_columns(db_path, table, df.columns)
    for _, row in df.iterrows():
        keyval = str(row.get(unique_key, ""))
        if not keyval:
            continue
        cur.execute(
            f'SELECT COUNT(1) FROM "{table}" WHERE "{unique_key}"=?',
            (keyval,),
        )
        exists = cur.fetchone()[0] == 1
        if exists:
            set_clause = ", ".join([f'"{col}"=?' for col in df.columns])
            values = [_normalize_value(row[col]) for col in df.columns]
            values.append(keyval)
            cur.execute(
                f'UPDATE "{table}" SET {set_clause} WHERE "{unique_key}"=?',
                values,
            )
        else:
            placeholders = ",".join(["?"] * len(df.columns))
            insert_cols = ",".join([f'"{col}"' for col in df.columns])
            values = [_normalize_value(row[col]) for col in df.columns]
            cur.execute(
                f'INSERT INTO "{table}" ({insert_cols}) VALUES ({placeholders})', values
            )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Text analytics and heuristics
# ---------------------------------------------------------------------------

def text_metrics(text: str) -> Dict[str, float]:
    if not text or not isinstance(text, str):
        return {
            "chars": 0.0,
            "words": 0.0,
            "sents": 0.0,
            "avg_sent_len": 0.0,
            "ttr": 0.0,
            "stop_ratio": 0.0,
            "pronoun_ratio": 0.0,
            "punct_variety": 0.0,
        }
    cleaned = text.strip()
    words = re.findall(r"[A-Za-z']+", cleaned.lower())
    sentences = [s for s in re.split(r"[.!?]+", cleaned) if s.strip()]
    stops = set(
        "a an the and or but so if then because as of to in on for with at by from is are was were be been being i me my we our you your he him his she her they them their it's its that which who whom this those these".split()
    )
    pronouns = set("i me my mine we us our ours".split())
    chars = float(len(cleaned))
    n_words = float(len(words))
    n_sents = float(len(sentences))
    avg_sentence_length = n_words / max(1.0, n_sents)
    type_token_ratio = len(set(words)) / max(1.0, n_words)
    stop_ratio = sum(1 for w in words if w in stops) / max(1.0, n_words)
    pronoun_ratio = sum(1 for w in words if w in pronouns) / max(1.0, n_words)
    puncts = re.findall(r"[^A-Za-z0-9\s]", cleaned)
    punct_variety = len(set(puncts)) / max(1.0, len(puncts)) if puncts else 0.0
    return {
        "chars": chars,
        "words": n_words,
        "sents": n_sents,
        "avg_sent_len": avg_sentence_length,
        "ttr": type_token_ratio,
        "stop_ratio": stop_ratio,
        "pronoun_ratio": pronoun_ratio,
        "punct_variety": punct_variety,
    }


def ai_suspect(answers: Iterable[str], flags_on: Dict[str, float], template_phrases: Iterable[str]):
    texts = [a for a in answers if isinstance(a, str) and a.strip()]
    if not texts:
        return False, 0.0
    metrics = [text_metrics(t) for t in texts]
    averages = {k: sum(m[k] for m in metrics) / len(metrics) for k in metrics[0].keys()}
    score = 0
    if averages["ttr"] < flags_on.get("low_lexical_diversity", 0.42):
        score += 1
    if averages["pronoun_ratio"] < flags_on.get("low_personal_pronouns", 0.02):
        score += 1
    if averages["stop_ratio"] > flags_on.get("high_stopword_ratio", 0.72):
        score += 1
    if averages["punct_variety"] < flags_on.get("low_punctuation_variety", 0.04):
        score += 1
    combined = " ".join(texts).lower()
    if any(tp in combined for tp in template_phrases):
        score += 2
    return score >= 3, score / 5.0


def summarize_answers(answers: Iterable[str], max_sentences: int = 4) -> str:
    sentences: List[str] = []
    for answer in answers:
        if not isinstance(answer, str):
            continue
        pieces = re.split(r"(?<=[.!?])\s+", answer.strip())
        for piece in pieces:
            if len(piece.split()) >= 5:
                sentences.append(piece.strip())
    if not sentences:
        return ""
    counts = Counter()
    for sentence in sentences:
        for token in set(re.findall(r"[a-z']+", sentence.lower())):
            counts[token] += 1
    scored = [
        (sum(counts[token] for token in re.findall(r"[a-z']+", sentence.lower())), sentence)
        for sentence in sentences
    ]
    scored.sort(reverse=True)
    chosen: List[str] = []
    seen = set()
    for _, sentence in scored:
        key = sentence[:80]
        if key not in seen:
            chosen.append(sentence)
            seen.add(key)
        if len(chosen) >= max_sentences:
            break
    return " ".join(chosen)


def keyword_score(texts: Iterable[str], positive: Iterable[str], negative: Iterable[str]) -> int:
    text = " ".join([t for t in texts if isinstance(t, str)]).lower()
    return sum(text.count(word.lower()) for word in positive) - sum(
        text.count(word.lower()) for word in negative
    )


def compute_fit(df: pd.DataFrame, cfg: Dict) -> List[float]:
    rubric = cfg.get("rubric", {})
    weights = rubric.get("weights", {})
    pos_keywords = rubric.get("keywords_positive", [])
    neg_keywords = rubric.get("keywords_negative", [])
    text_fields = cfg.get("fields", {}).get("text_fields", [])
    scores: List[float] = []
    for _, row in df.iterrows():
        essays = [str(row.get(col, "")) for col in text_fields]
        kw_score = keyword_score(essays, pos_keywords, neg_keywords)
        motivation = kw_score / 5.0
        exp_cols = [c for c in df.columns if "previous experience" in c.lower()]
        has_exp = any(str(row.get(c, "")).strip().lower() in {"yes", "y", "i have", "yep"} for c in exp_cols)
        availability_cols = [
            c
            for c in df.columns
            if "willing" in c.lower() or "commitments" in c.lower()
        ]
        available = not any("conflict" in str(row.get(c, "")).lower() for c in availability_cols)
        gpa_value = 0.0
        for col in df.columns:
            if "gpa" in col.lower():
                try:
                    gpa_value = max(gpa_value, float(str(row.get(col, "")).split()[0]))
                except Exception:
                    continue
        score = (
            weights.get("motivation_quality", 0.0) * motivation
            + weights.get("experience_bonus", 0.0) * (1.0 if has_exp else 0.0)
            + weights.get("availability", 0.0) * (1.0 if available else 0.0)
            + weights.get("gpa", 0.0) * (gpa_value - 2.5)
        )
        scores.append(score)
    return scores


def extract_gpa_flag(row: pd.Series, threshold: float) -> str:
    gpa_value: Optional[float] = None
    for col in row.index:
        if "gpa" in col.lower():
            try:
                current = float(str(row[col]).split()[0])
            except Exception:
                continue
            gpa_value = max(gpa_value, current) if gpa_value is not None else current
    if gpa_value is None:
        return ""
    return "LOW" if gpa_value < threshold else ""


def infer_role_preference(row: pd.Series, role: str) -> Optional[int]:
    role_lower = role.lower()
    target_words = {"please rank", "rank", "preference", "preferred"}
    cols = [
        col
        for col in row.index
        if role_lower in col.lower() and any(token in col.lower() for token in target_words)
    ]

    def contains_rank(value: str, target: int) -> bool:
        if not value:
            return False
        stripped = value.lower().strip()
        if stripped in {"nan", "n/a", "", "none"}:
            return False
        if stripped.isdigit():
            return int(stripped) == target
        # look for standalone digits, # or ordinal text
        if str(target) in stripped.split():
            return True
        if f"#{target}" in stripped or f"no.{target}" in stripped:
            return True
        if stripped.endswith(str(target)):
            return True
        ordinal = {
            1: {"1st", "first"},
            2: {"2nd", "second"},
            3: {"3rd", "third"},
        }
        for token in ordinal.get(target, set()):
            if token in stripped:
                return True
        return False

    if not cols:
        return None

    for target in (1, 2, 3):
        for col in cols:
            value = str(row.get(col, ""))
            if contains_rank(value, target):
                return target
    return None


def compute_role_fit(base_score: float, pref: Optional[int]) -> float:
    if pref is None:
        return base_score
    return base_score + (3 - pref) * 0.25


def compute_badges(essays: Iterable[str], cfg: Dict[str, Dict]) -> Dict[str, str]:
    badge_cfg = cfg.get("rubric", {}).get("keyword_badges", {})
    text = " ".join([t.lower() for t in essays if isinstance(t, str)])
    badges: Dict[str, str] = {}
    for badge_name, badge_info in badge_cfg.items():
        keywords = [kw.lower() for kw in badge_info.get("keywords", [])]
        hits = sum(text.count(kw) for kw in keywords)
        if hits >= badge_info.get("thresholds", {}).get("yes", DEFAULT_BADGE_THRESHOLDS["yes"]):
            level = "yes"
        elif hits >= badge_info.get("thresholds", {}).get("maybe", DEFAULT_BADGE_THRESHOLDS["maybe"]):
            level = "maybe"
        else:
            level = "no"
        badges[f"badge__{slugify(badge_name)}"] = level
    return badges


# ---------------------------------------------------------------------------
# CSV <-> SQLite synchronisation
# ---------------------------------------------------------------------------

def prepare_enriched_frame(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    df = df.copy()
    text_fields = cfg.get("fields", {}).get("text_fields", [])
    reviewers = get_reviewer_profiles(cfg)
    notes_base = cfg["review"]["notes_field"]
    rating_base = cfg["review"].get("rating_field")
    recommend_base = cfg["review"].get("recommendation_field")
    if recommend_base and recommend_base not in df.columns:
        df[recommend_base] = ""
    for username in reviewers.keys():
        note_col = f"{notes_base}__{username}"
        if note_col not in df.columns:
            df[note_col] = ""
        if rating_base:
            rating_col = f"{rating_base}__{username}"
            if rating_col not in df.columns:
                df[rating_col] = ""
        if recommend_base:
            recommend_col = f"{recommend_base}__{username}"
            if recommend_col not in df.columns:
                df[recommend_col] = ""
    fit_scores = compute_fit(df, cfg)
    df[cfg["review"]["rubric_score_field"]] = [f"{score:.2f}" for score in fit_scores]
    ai_flags: List[str] = []
    ai_scores: List[str] = []
    summaries: List[str] = []
    gpa_flags: List[str] = []
    badges_store: Dict[str, List[str]] = {}
    for role in ROLE_NAMES:
        df[f"{role}__fit"] = ""
        df[f"{role}__pref"] = ""
    for offset, (idx, row) in enumerate(df.iterrows()):
        essays = [str(row.get(col, "")) for col in text_fields]
        flag, score = ai_suspect(
            essays,
            cfg.get("ai_detection", {}).get("flags_on", {}),
            cfg.get("ai_detection", {}).get("flags_on", {}).get("template_phrases", []),
        )
        ai_flags.append("YES" if flag else "NO")
        ai_scores.append(f"{score:.2f}")
        summaries.append(summarize_answers(essays))
        gpa_flags.append(extract_gpa_flag(row, cfg.get("rubric", {}).get("gpa_flag_threshold", 3.0)))
        badges = compute_badges(essays, cfg)
        for key, value in badges.items():
            badges_store.setdefault(key, []).append(value)
        for role in ROLE_NAMES:
            pref = infer_role_preference(row, role)
            df.at[idx, f"{role}__pref"] = "" if pref is None else str(pref)
            df.at[idx, f"{role}__fit"] = f"{compute_role_fit(fit_scores[offset], pref):.2f}"
    df[cfg["review"]["ai_flag_field"]] = ai_flags
    df["ai_score"] = ai_scores
    df["summary"] = summaries
    df["gpa_flag"] = gpa_flags
    for key, values in badges_store.items():
        df[key] = values
    return df


def import_csv_to_db(
    csv_path: str,
    skiprows: int,
    db_path: str,
    table: str,
    unique_key: str,
    cfg: Dict,
    write_back_csv: bool = True,
    write_back_target: Optional[str] = None,
) -> pd.DataFrame:
    csv_path_obj = Path(csv_path)
    if not csv_path_obj.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df_raw = pd.read_csv(csv_path_obj, skiprows=skiprows, engine="python")
    enriched_df = prepare_enriched_frame(df_raw, cfg)
    extra_cols = [
        cfg["review"]["status_field"],
        cfg["review"]["notes_field"],
        cfg["review"]["rubric_score_field"],
        cfg["review"]["ai_flag_field"],
        cfg["review"].get("rating_field", "review_score"),
        cfg["review"].get("recommendation_field", "review_recommendation"),
        "ai_score",
        "summary",
        "gpa_flag",
    ]
    badge_cols = [col for col in enriched_df.columns if col.startswith("badge__")]
    reviewers = get_reviewer_profiles(cfg)
    notes_base = cfg["review"]["notes_field"]
    rating_base = cfg["review"].get("rating_field")
    recommend_base = cfg["review"].get("recommendation_field")
    interview_def = get_interview_definition(cfg)
    interview_prompt_bases = [item["column"] for item in interview_def["prompts"]]
    interview_question_bases = [item["column"] for item in interview_def["questions"]]
    interview_rating_base = interview_def["rating_field"]
    interview_outcome_base = interview_def["outcome_field"]
    interview_log_base = interview_def["log_field"]
    interview_resources_base = interview_def["resources_field"]
    interview_summary_base = interview_def["summary_field"]
    for role in ROLE_NAMES:
        extra_cols.extend([f"{role}__fit", f"{role}__pref"])
    for username in reviewers.keys():
        extra_cols.append(f"{notes_base}__{username}")
        if rating_base:
            extra_cols.append(f"{rating_base}__{username}")
        if recommend_base:
            extra_cols.append(f"{recommend_base}__{username}")
        for base in interview_prompt_bases:
            extra_cols.append(f"{base}__{username}")
        extra_cols.append(f"{interview_rating_base}__{username}")
        extra_cols.append(f"{interview_outcome_base}__{username}")
        extra_cols.append(f"{interview_log_base}__{username}")
        extra_cols.append(f"{interview_resources_base}__{username}")
        extra_cols.append(f"{interview_summary_base}__{username}")
        for base in interview_question_bases:
            extra_cols.append(f"{base}__{username}")
    extra_cols.extend(badge_cols)
    extra_cols = list(dict.fromkeys(extra_cols))

    reviewer_note_fields = [f"{notes_base}__{username}" for username in reviewers.keys()]
    reviewer_rating_fields = [f"{rating_base}__{username}" for username in reviewers.keys()] if rating_base else []
    reviewer_recommend_fields = [f"{recommend_base}__{username}" for username in reviewers.keys()] if recommend_base else []
    interview_prompt_fields = [f"{base}__{username}" for base in interview_prompt_bases for username in reviewers.keys()]
    interview_rating_fields = [f"{interview_rating_base}__{username}" for username in reviewers.keys()]
    interview_outcome_fields = [f"{interview_outcome_base}__{username}" for username in reviewers.keys()]
    interview_log_fields = [f"{interview_log_base}__{username}" for username in reviewers.keys()]
    interview_resource_fields = [f"{interview_resources_base}__{username}" for username in reviewers.keys()]
    interview_summary_fields = [f"{interview_summary_base}__{username}" for username in reviewers.keys()]
    interview_question_fields = [f"{base}__{username}" for base in interview_question_bases for username in reviewers.keys()]

    preserve_columns: set[str] = {
        cfg["review"]["status_field"],
        cfg["review"]["notes_field"],
    }
    if rating_base:
        preserve_columns.add(rating_base)
    if recommend_base:
        preserve_columns.add(recommend_base)
    preserve_columns.update(reviewer_note_fields)
    preserve_columns.update(reviewer_rating_fields)
    preserve_columns.update(reviewer_recommend_fields)
    preserve_columns.update(interview_prompt_fields)
    preserve_columns.update(interview_rating_fields)
    preserve_columns.update(interview_outcome_fields)
    preserve_columns.update(interview_log_fields)
    preserve_columns.update(interview_resource_fields)
    preserve_columns.update(interview_summary_fields)
    preserve_columns.update(interview_question_fields)

    if Path(db_path).exists():
        key_values = (
            enriched_df[unique_key]
            .astype(str)
            .dropna()
            .unique()
            .tolist()
        )
        if key_values:
            placeholders = ",".join(["?"] * len(key_values))
            with sqlite3.connect(db_path) as conn:
                existing_df = pd.read_sql_query(
                    f'SELECT * FROM "{table}" WHERE "{unique_key}" IN ({placeholders})',
                    conn,
                    params=key_values,
                )
            if not existing_df.empty:
                existing_df[unique_key] = existing_df[unique_key].astype(str)
                enriched_df[unique_key] = enriched_df[unique_key].astype(str)
                existing_indexed = existing_df.set_index(unique_key)
                enriched_indexed = enriched_df.set_index(unique_key)
                overlap = enriched_indexed.index.intersection(existing_indexed.index)
                if not overlap.empty:
                    for column in preserve_columns:
                        if column not in enriched_indexed.columns or column not in existing_indexed.columns:
                            continue
                        new_values = enriched_indexed.loc[overlap, column]
                        existing_values = existing_indexed.loc[overlap, column]
                        mask = new_values.apply(_is_empty_value)
                        if mask.any():
                            indices_to_fill = mask[mask].index
                            enriched_indexed.loc[indices_to_fill, column] = existing_values.loc[
                                indices_to_fill
                            ]
                enriched_df = enriched_indexed.reset_index()

    if not Path(db_path).exists():
        init_db(db_path, table, list(enriched_df.columns) + extra_cols, unique_key)
    ensure_columns(db_path, table, list(enriched_df.columns) + extra_cols)
    upsert_rows(db_path, table, enriched_df, unique_key)
    if write_back_csv:
        target_path = Path(write_back_target) if write_back_target else csv_path_obj
        if target_path != csv_path_obj and not target_path.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(csv_path_obj.read_bytes())
        review_fields = [
            cfg["review"]["status_field"],
            cfg["review"]["notes_field"],
            cfg["review"]["rubric_score_field"],
            cfg["review"]["ai_flag_field"],
            cfg["review"].get("rating_field", "review_score"),
            cfg["review"].get("recommendation_field", "review_recommendation"),
            "ai_score",
            "summary",
            "gpa_flag",
        ]
        review_fields += badge_cols
        review_fields += [f"{role}__fit" for role in ROLE_NAMES]
        review_fields += [f"{role}__pref" for role in ROLE_NAMES]
        review_fields += reviewer_note_fields
        review_fields += reviewer_rating_fields
        review_fields += reviewer_recommend_fields
        review_fields += interview_prompt_fields
        review_fields += interview_rating_fields
        review_fields += interview_outcome_fields
        review_fields += interview_log_fields
        review_fields += interview_resource_fields
        review_fields += interview_summary_fields
        review_fields += interview_question_fields
        review_fields = list(dict.fromkeys(review_fields))

        merge_with_master_csv(
            csv_path=str(target_path),
            skiprows=skiprows,
            unique_key=unique_key,
            enriched_df=enriched_df,
            review_fields=review_fields,
        )
    return enriched_df


def merge_with_master_csv(
    csv_path: str,
    skiprows: int,
    unique_key: str,
    enriched_df: pd.DataFrame,
    review_fields: List[str],
) -> None:
    csv_path_obj = Path(csv_path)
    if not csv_path_obj.exists():
        return
    review_fields = list(dict.fromkeys(review_fields))
    # Capture header preamble (lines before header row)
    preamble: List[str] = []
    if skiprows:
        with csv_path_obj.open("r", encoding="utf-8-sig") as handle:
            for _ in range(skiprows):
                line = handle.readline()
                if not line:
                    break
                preamble.append(line.rstrip("\n"))
    master_df = pd.read_csv(csv_path_obj, skiprows=skiprows, engine="python")
    master_df.set_index(unique_key, inplace=True)
    enriched_idx = enriched_df.set_index(unique_key)
    # Ensure review columns exist in master
    for field in review_fields:
        if field not in master_df.columns:
            master_df[field] = ""
    # Update/add rows from enriched data
    for key, row in enriched_idx.iterrows():
        if key in master_df.index:
            for field in review_fields:
                if field in row:
                    value = row.get(field, "")
                    if value is None:
                        continue
                    if isinstance(value, float) and pd.isna(value):
                        continue
                    if isinstance(value, str) and value.strip() == "":
                        continue
                    master_df.at[key, field] = value
        else:
            master_df.loc[key] = row
    master_df.reset_index(inplace=True)
    temp_path = csv_path_obj.with_suffix(".tmp.csv")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        if preamble:
            for line in preamble:
                handle.write(f"{line}\n")
        master_df.to_csv(handle, index=False)
    backup_path = csv_path_obj.with_suffix(f".backup-{datetime.now().strftime('%Y%m%d%H%M%S')}.csv")
    csv_path_obj.replace(backup_path)
    temp_path.replace(csv_path_obj)


def update_review(
    db_path: str,
    table: str,
    unique_key: str,
    keyval: str,
    updates: Dict[str, Optional[str]],
    csv_path: Optional[str] = None,
    skiprows: int = 0,
    actor: Optional[str] = None,
) -> None:
    if not updates:
        return
    ensure_columns(db_path, table, updates.keys())
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    if actor:
        cur.execute(
            'CREATE TABLE IF NOT EXISTS "review_audit" (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, actor TEXT, submission_id TEXT, field TEXT, value TEXT)'
        )
    set_clause = ", ".join([f'"{field}"=?' for field in updates.keys()])
    values = [_normalize_value(value) for value in updates.values()]
    values.append(keyval)
    cur.execute(
        f'UPDATE "{table}" SET {set_clause} WHERE "{unique_key}"=?',
        values,
    )
    if actor:
        from datetime import datetime as _dt

        ts = _dt.utcnow().isoformat()
        for field, value in updates.items():
            cur.execute(
                'INSERT INTO "review_audit" (timestamp, actor, submission_id, field, value) VALUES (?,?,?,?,?)',
                (ts, actor, keyval, field, _normalize_value(value)),
            )
    conn.commit()
    conn.close()
    if csv_path:
        sync_single_row_to_csv(
            csv_path=csv_path,
            skiprows=skiprows,
            unique_key=unique_key,
            keyval=keyval,
            updates=updates,
        )


def sync_single_row_to_csv(
    csv_path: str,
    skiprows: int,
    unique_key: str,
    keyval: str,
    updates: Dict[str, Optional[str]],
) -> None:
    csv_path_obj = Path(csv_path)
    if not csv_path_obj.exists():
        return
    preamble: List[str] = []
    if skiprows:
        with csv_path_obj.open("r", encoding="utf-8-sig") as handle:
            for _ in range(skiprows):
                line = handle.readline()
                if not line:
                    break
                preamble.append(line.rstrip("\n"))
    df = pd.read_csv(csv_path_obj, skiprows=skiprows, engine="python")
    if unique_key not in df.columns:
        return
    if keyval not in df[unique_key].astype(str).values:
        return
    mask = df[unique_key].astype(str) == str(keyval)
    for field, value in updates.items():
        if field not in df.columns:
            df[field] = ""
        df.loc[mask, field] = "" if value is None else value
    temp_path = csv_path_obj.with_suffix(".tmp.csv")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        if preamble:
            for line in preamble:
                handle.write(f"{line}\n")
        df.to_csv(handle, index=False)
    temp_path.replace(csv_path_obj)



def fetch_audit_log(db_path: str, limit: int = 200) -> pd.DataFrame:
    db_file = Path(db_path)
    if not db_file.exists():
        return pd.DataFrame(columns=["timestamp", "actor", "submission_id", "field", "value"])
    with sqlite3.connect(db_path) as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS "review_audit" (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, actor TEXT, submission_id TEXT, field TEXT, value TEXT)')
        query = 'SELECT timestamp, actor, submission_id, field, value FROM "review_audit" ORDER BY id DESC LIMIT ?'
        df = pd.read_sql_query(query, conn, params=(limit,))
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Import applications into SQLite and CSV")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument(
        "--import-csv",
        action="store_true",
        help="Read the configured CSV and sync into SQLite (default action)",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    if args.import_csv:
        import_csv_to_db(
            csv_path=cfg["csv"]["path"],
            skiprows=cfg["csv"].get("skiprows", 0),
            db_path=cfg["database"]["path"],
            table=cfg["database"]["table"],
            unique_key=cfg["database"]["unique_key"],
            cfg=cfg,
            write_back_csv=cfg["csv"].get("write_back", True),
            write_back_target=cfg["csv"].get("write_back_path"),
        )
        print("Imported and synced records")
