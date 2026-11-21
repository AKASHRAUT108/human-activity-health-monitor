from __future__ import annotations

import io
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent
EVALUATION_PATH = ROOT / "models" / "kth_tuned_conf03" / "kth_test_evaluation.json"

st.set_page_config(
    page_title="Activity Monitor | Human Activity Health Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# PROFESSIONAL DASHBOARD STYLING
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #0b1120;
        color: #e5edf7;
    }

    [data-testid="stSidebar"] {
        background: #101a2d;
        border-right: 1px solid #24334b;
    }

    [data-testid="stSidebar"] * {
        color: #dce7f5;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    .main-title {
        font-size: 2rem;
        font-weight: 800;
        color: #f1f5f9;
        letter-spacing: -1px;
        margin-bottom: 3px;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 0.92rem;
        margin-bottom: 25px;
    }

    .metric-card {
        background: linear-gradient(145deg, #17243a, #111c30);
        border: 1px solid #263954;
        border-radius: 14px;
        padding: 20px 21px;
        min-height: 122px;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 1.85rem;
        font-weight: 800;
        margin-top: 9px;
    }

    .metric-caption {
        color: #7dd3c7;
        font-size: 0.78rem;
        margin-top: 4px;
    }

    .section-heading {
        color: #e2e8f0;
        font-size: 1.12rem;
        font-weight: 700;
        margin-top: 12px;
        margin-bottom: 12px;
    }

    .info-panel {
        background: #111c30;
        border: 1px solid #263954;
        border-radius: 13px;
        padding: 18px;
        margin-bottom: 12px;
    }

    .status-online {
        color: #5eead4;
        font-weight: 700;
    }

    .status-offline {
        color: #fca5a5;
        font-weight: 700;
    }

    .small-muted {
        color: #94a3b8;
        font-size: 0.82rem;
    }

    div[data-testid="stFileUploader"] {
        background: #111c30;
        border: 1px dashed #3b587a;
        border-radius: 12px;
        padding: 12px;
    }

    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(90deg, #0d9488, #0f766e);
        color: white;
        border: 0;
        border-radius: 9px;
        font-weight: 700;
        min-height: 42px;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: #14b8a6;
        color: white;
        border: 0;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #263954;
        border-radius: 10px;
        overflow: hidden;
    }

    h1, h2, h3 {
        color: #f1f5f9;
    }

    hr {
        border-color: #263954;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

@st.cache_data
def load_evaluation() -> dict[str, Any]:
    if not EVALUATION_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {EVALUATION_PATH}"
        )

    with EVALUATION_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def metric_card(label: str, value: str, caption: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_api_url() -> str:
    return st.session_state.get(
        "api_url", os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
    ).rstrip("/")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_prediction_windows(result: dict) -> list:
    for key in ("predictions", "windows", "window_predictions", "results"):
        value = result.get(key)
        if isinstance(value, list):
            return value
    return []


def get_dominant_activity(result: dict) -> str:
    for key in (
        "dominant_activity",
        "predicted_activity",
        "activity",
        "overall_prediction",
    ):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value

    counts = result.get("activity_counts")
    if isinstance(counts, dict) and counts:
        return max(counts, key=counts.get)

    windows = get_prediction_windows(result)
    activities = [
        item.get("activity") or item.get("predicted_activity") or item.get("label")
        for item in windows
        if isinstance(item, dict)
    ]
    activities = [item for item in activities if item]
    if activities:
        return max(set(activities), key=activities.count)

    return "Not available"


def get_confidence(result: dict) -> float | None:
    for key in ("confidence", "dominant_confidence", "prediction_confidence"):
        value = result.get(key)
        if value is not None:
            number = safe_float(value, -1)
            if number >= 0:
                return number * 100 if number <= 1 else number

    return None


# ---------------------------------------------------------
# PDF REPORT GENERATION
# ---------------------------------------------------------

def create_pdf_report(
    evaluation: dict,
    prediction: dict | None = None,
) -> bytes:
    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=17 * mm,
        leftMargin=17 * mm,
        topMargin=17 * mm,
        bottomMargin=17 * mm,
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportTitleCustom",
            parent=styles["Title"],
            textColor=colors.HexColor("#0F766E"),
            fontSize=22,
            leading=27,
            alignment=TA_CENTER,
            spaceAfter=8,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SectionCustom",
            parent=styles["Heading2"],
            textColor=colors.HexColor("#0F766E"),
            fontSize=13,
            spaceBefore=12,
            spaceAfter=7,
        )
    )

    story = []

    story.append(Paragraph("HUMAN ACTIVITY MONITOR", styles["ReportTitleCustom"]))
    story.append(
        Paragraph(
            "Model Evaluation and Activity Analysis Report",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M:%S')}",
            styles["Normal"],
        )
    )

    story.append(Paragraph("1. Model Overview", styles["SectionCustom"]))

    accuracy = safe_float(evaluation.get("test_accuracy"))
    model_data = [
        ["Metric", "Value"],
        ["Dataset", str(evaluation.get("dataset", "KTH"))],
        ["Model", str(evaluation.get("model_type", "LSTM"))],
        ["Test accuracy", f"{accuracy * 100:.2f}%"],
        ["Test sequences", str(evaluation.get("test_sequence_count", "N/A"))],
        ["Sequence length", str(evaluation.get("sequence_length", "N/A"))],
        ["Feature dimension", str(evaluation.get("feature_dimension", "N/A"))],
        ["Activity classes", str(len(evaluation.get("class_names", [])))],
    ]

    table = Table(model_data, colWidths=[65 * mm, 95 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F1F5F9")),
                ("PADDING", (0, 0), (-1, -1), 7),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    story.append(table)

    story.append(Paragraph("2. Per-Class Model Performance", styles["SectionCustom"]))

    report = evaluation.get("classification_report", {})
    class_names = evaluation.get("class_names", [])

    class_data = [["Activity", "Precision", "Recall", "F1-score", "Support"]]

    for name in class_names:
        metrics = report.get(name, {})
        class_data.append(
            [
                name,
                f"{safe_float(metrics.get('precision')):.3f}",
                f"{safe_float(metrics.get('recall')):.3f}",
                f"{safe_float(metrics.get('f1-score')):.3f}",
                str(int(safe_float(metrics.get("support")))),
            ]
        )

    class_table = Table(
        class_data,
        colWidths=[48 * mm, 28 * mm, 28 * mm, 28 * mm, 28 * mm],
    )
    class_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F1F5F9")),
            ]
        )
    )
    story.append(class_table)

    if prediction:
        story.append(Paragraph("3. Video Analysis", styles["SectionCustom"]))

        dominant = get_dominant_activity(prediction)
        confidence = get_confidence(prediction)

        prediction_data = [
            ["Field", "Result"],
            ["Dominant activity", dominant],
            [
                "Prediction confidence",
                f"{confidence:.2f}%" if confidence is not None else "Not provided",
            ],
            ["Video information", str(prediction.get("video_info", "See API result"))],
            ["Prediction windows", str(len(get_prediction_windows(prediction)))],
        ]

        prediction_table = Table(
            prediction_data,
            colWidths=[65 * mm, 95 * mm],
        )
        prediction_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                    ("PADDING", (0, 0), (-1, -1), 7),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F1F5F9")),
                ]
            )
        )
        story.append(prediction_table)

        story.append(Spacer(1, 8))
        story.append(
            Paragraph(
                "The activity result is a machine-learning estimate based on "
                "the submitted video. It is not a medical diagnosis, clinical "
                "assessment, or guaranteed health alert.",
                styles["Normal"],
            )
        )

    story.append(Spacer(1, 18))
    story.append(
        Paragraph(
            "This report describes model performance on the KTH test dataset. "
            "Test accuracy does not guarantee the same performance on arbitrary "
            "real-world videos.",
            styles["Italic"],
        )
    )

    document.build(story)
    return buffer.getvalue()


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

if "api_url" not in st.session_state:
    st.session_state.api_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:
    st.markdown("## 📊 ACTIVITY MONITOR")
    st.caption("Human Activity Health Monitor")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Video Analysis",
            "Model Performance",
            "Reports & About",
        ],
        label_visibility="visible",
    )

    st.divider()
    st.markdown("**Backend connection**")

    st.session_state.api_url = st.text_input(
        "FastAPI URL",
        value=st.session_state.api_url,
        help="Address of your running FastAPI service.",
    )

    if st.button("Check API connection", use_container_width=True):
        try:
            response = requests.get(
                f"{get_api_url()}/health",
                timeout=5,
            )
            response.raise_for_status()
            st.success("API is reachable.")
            st.json(response.json())
        except requests.RequestException as exc:
            st.error(f"API connection failed: {exc}")

    st.divider()
    st.markdown(
        """
        <div class="small-muted">
        <b>Project</b><br>
        Human Activity Recognition<br><br>
        <b>Model</b><br>
        Tuned LSTM<br><br>
        <b>Dataset</b><br>
        KTH Actions
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# LOAD MODEL EVALUATION
# ---------------------------------------------------------

try:
    evaluation = load_evaluation()
except Exception as exc:
    st.error(f"Unable to load model evaluation: {exc}")
    st.info(
        "Check that models/kth_tuned_conf03/kth_test_evaluation.json "
        "exists in the project directory."
    )
    st.stop()

accuracy = safe_float(evaluation.get("test_accuracy"))
class_names = evaluation.get("class_names", [])
classification_report = evaluation.get("classification_report", {})
confusion_matrix = evaluation.get("confusion_matrix", [])

# ---------------------------------------------------------
# PAGE: OVERVIEW
# ---------------------------------------------------------

if page == "Overview":

    st.markdown(
        '<div class="main-title">Activity Monitor</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Human activity recognition and model monitoring dashboard</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Test accuracy", f"{accuracy * 100:.2f}%", "KTH test dataset")

    with col2:
        metric_card(
            "Test sequences",
            f"{int(evaluation.get('test_sequence_count', 0)):,}",
            "Evaluated samples",
        )

    with col3:
        metric_card(
            "Activity classes",
            str(len(class_names)),
            "Supported categories",
        )

    with col4:
        metric_card(
            "Model architecture",
            str(evaluation.get("model_type", "LSTM")),
            "Tuned model candidate",
        )

    st.markdown("---")

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown('<div class="section-heading">Per-class F1 performance</div>', unsafe_allow_html=True)

        rows = []
        for name in class_names:
            item = classification_report.get(name, {})
            rows.append(
                {
                    "Activity": name.title(),
                    "F1-score": safe_float(item.get("f1-score")),
                    "Precision": safe_float(item.get("precision")),
                    "Recall": safe_float(item.get("recall")),
                }
            )

        performance_df = pd.DataFrame(rows)

        fig = px.bar(
            performance_df,
            x="Activity",
            y="F1-score",
            range_y=[0, 1],
            text=performance_df["F1-score"].map(lambda x: f"{x:.2f}"),
            color="F1-score",
            color_continuous_scale=["#334155", "#0d9488", "#5eead4"],
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b1120",
            plot_bgcolor="#111c30",
            font_color="#dce7f5",
            yaxis_title="F1-score",
            xaxis_title="",
            coloraxis_showscale=False,
            margin=dict(l=15, r=15, t=20, b=15),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-heading">Model summary</div>', unsafe_allow_html=True)

        macro_f1 = safe_float(
            classification_report.get("macro avg", {}).get("f1-score")
        )
        weighted_f1 = safe_float(
            classification_report.get("weighted avg", {}).get("f1-score")
        )

        metric_card("Macro F1-score", f"{macro_f1 * 100:.2f}%", "Average across classes")
        st.write("")
        metric_card("Weighted F1-score", f"{weighted_f1 * 100:.2f}%", "Support-weighted performance")

    st.markdown("---")
    st.markdown('<div class="section-heading">Supported activities</div>', unsafe_allow_html=True)

    activity_columns = st.columns(3)
    for index, name in enumerate(class_names):
        with activity_columns[index % 3]:
            st.markdown(
                f"""
                <div class="info-panel">
                    <b>{name.replace('_', ' ').title()}</b><br>
                    <span class="small-muted">Recognized activity category</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption(
        "Model metrics are calculated on the KTH test dataset. "
        "They are not a guarantee of real-world prediction performance."
    )


# ---------------------------------------------------------
# PAGE: VIDEO ANALYSIS
# ---------------------------------------------------------

elif page == "Video Analysis":

    st.markdown(
        '<div class="main-title">Video Analysis</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Upload a video and run activity recognition through the FastAPI inference service.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-heading">Upload video</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        help="Supported formats: MP4, AVI, MOV and MKV.",
    )

    if uploaded_file is not None:
        st.info(
            f"Selected file: {uploaded_file.name} "
            f"({uploaded_file.size / (1024 * 1024):.2f} MB)"
        )

        if uploaded_file.size > 200 * 1024 * 1024:
            st.error("File exceeds the 200 MB API upload limit.")
        elif st.button("Analyze video", type="primary", use_container_width=True):
            with st.spinner("Processing video and generating predictions..."):
                try:
                    response = requests.post(
                        f"{get_api_url()}/predict",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type or "application/octet-stream",
                            )
                        },
                        timeout=600,
                    )

                    if not response.ok:
                        try:
                            detail = response.json()
                        except ValueError:
                            detail = response.text

                        st.error(f"API returned HTTP {response.status_code}: {detail}")
                    else:
                        result = response.json()
                        st.session_state.prediction_result = result
                        st.session_state.uploaded_filename = uploaded_file.name
                        st.success("Video analysis completed successfully.")

                except requests.Timeout:
                    st.error("The video analysis timed out. Try a shorter video.")
                except requests.RequestException as exc:
                    st.error(f"Unable to reach the inference API: {exc}")
                except ValueError:
                    st.error("The API returned an invalid JSON response.")

    result = st.session_state.prediction_result

    if result:
        st.markdown("---")
        st.markdown('<div class="section-heading">Analysis result</div>', unsafe_allow_html=True)

        dominant = get_dominant_activity(result)
        confidence = get_confidence(result)

        col1, col2, col3 = st.columns(3)

        with col1:
            metric_card("Dominant activity", dominant.replace("_", " ").title(), "Most frequently predicted category")

        with col2:
            metric_card(
                "Prediction confidence",
                f"{confidence:.2f}%" if confidence is not None else "N/A",
                "Model output confidence, when provided",
            )

        with col3:
            metric_card(
                "Prediction windows",
                str(len(get_prediction_windows(result))),
                "Analyzed sequence windows",
            )

        st.markdown('<div class="section-heading">Video details</div>', unsafe_allow_html=True)

        st.write(f"**File:** {st.session_state.uploaded_filename or 'Uploaded video'}")

        video_info = result.get("video_info", {})
        if isinstance(video_info, dict):
            st.json(video_info)
        elif video_info:
            st.write(video_info)

        windows = get_prediction_windows(result)

        if windows:
            st.markdown('<div class="section-heading">Window-level predictions</div>', unsafe_allow_html=True)

            normalized = []
            for index, item in enumerate(windows):
                if not isinstance(item, dict):
                    continue

                activity = (
                    item.get("activity")
                    or item.get("predicted_activity")
                    or item.get("label")
                    or "Unknown"
                )

                row = {
                    "Window": item.get("window_index", index + 1),
                    "Activity": str(activity).replace("_", " ").title(),
                }

                for key in ("confidence", "start_frame", "end_frame", "start_time", "end_time"):
                    if key in item:
                        row[key.replace("_", " ").title()] = item[key]

                normalized.append(row)

            if normalized:
                st.dataframe(
                    pd.DataFrame(normalized),
                    use_container_width=True,
                    hide_index=True,
                )

            # Plot window predictions as a timeline.
            activity_values = [
                (
                    item.get("activity")
                    or item.get("predicted_activity")
                    or item.get("label")
                    or "Unknown"
                )
                for item in windows
                if isinstance(item, dict)
            ]

            if activity_values:
                timeline = pd.DataFrame(
                    {
                        "Window": list(range(1, len(activity_values) + 1)),
                        "Activity": [str(x).replace("_", " ").title() for x in activity_values],
                    }
                )

                fig = px.scatter(
                    timeline,
                    x="Window",
                    y="Activity",
                    color="Activity",
                    title="Predicted activity across video windows",
                )
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0b1120",
                    plot_bgcolor="#111c30",
                    font_color="#dce7f5",
                )
                st.plotly_chart(fig, use_container_width=True)

        counts = result.get("activity_counts")
        if isinstance(counts, dict) and counts:
            st.markdown('<div class="section-heading">Activity distribution</div>', unsafe_allow_html=True)

            count_df = pd.DataFrame(
                {
                    "Activity": list(counts.keys()),
                    "Windows": list(counts.values()),
                }
            )

            fig = px.pie(
                count_df,
                names="Activity",
                values="Windows",
                hole=0.55,
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b1120",
                font_color="#dce7f5",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Predictions are estimates from the trained machine-learning model. "
            "They are not medical diagnoses or validated health alerts."
        )

        with st.expander("View complete API response"):
            st.json(result)

    else:
        st.info("No video has been analyzed in this dashboard session yet.")


# ---------------------------------------------------------
# PAGE: MODEL PERFORMANCE
# ---------------------------------------------------------

elif page == "Model Performance":

    st.markdown(
        '<div class="main-title">Model Performance</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Evaluation metrics from the held-out KTH test dataset.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card("Test accuracy", f"{accuracy * 100:.2f}%", "Overall test accuracy")

    with col2:
        metric_card(
            "Test samples",
            f"{int(evaluation.get('test_sequence_count', 0)):,}",
            "Sequence-level test samples",
        )

    with col3:
        metric_card(
            "Feature dimension",
            str(evaluation.get("feature_dimension", "N/A")),
            "Features per frame",
        )

    st.markdown("---")
    st.markdown('<div class="section-heading">Classification report</div>', unsafe_allow_html=True)

    rows = []
    for name in class_names:
        item = classification_report.get(name, {})
        rows.append(
            {
                "Activity": name.title(),
                "Precision": safe_float(item.get("precision")),
                "Recall": safe_float(item.get("recall")),
                "F1-score": safe_float(item.get("f1-score")),
                "Support": int(safe_float(item.get("support"))),
            }
        )

    report_df = pd.DataFrame(rows)

    st.dataframe(
        report_df.style.format(
            {
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1-score": "{:.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    fig = go.Figure()

    for metric in ("Precision", "Recall", "F1-score"):
        fig.add_trace(
            go.Bar(
                name=metric,
                x=report_df["Activity"],
                y=report_df[metric],
            )
        )

    fig.update_layout(
        barmode="group",
        title="Precision, recall and F1-score by activity",
        yaxis=dict(title="Score", range=[0, 1]),
        xaxis_title="",
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#111c30",
        font_color="#dce7f5",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-heading">Confusion matrix</div>', unsafe_allow_html=True)

    if confusion_matrix:
        labels = evaluation.get("confusion_matrix_labels", class_names)

        matrix_fig = px.imshow(
            confusion_matrix,
            x=labels,
            y=labels,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Teal",
            labels={
                "x": "Predicted activity",
                "y": "Actual activity",
                "color": "Sequences",
            },
        )

        matrix_fig.update_layout(
            title="KTH test-set confusion matrix",
            template="plotly_dark",
            paper_bgcolor="#0b1120",
            plot_bgcolor="#111c30",
            font_color="#dce7f5",
        )

        st.plotly_chart(matrix_fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-heading">Evaluation observations</div>', unsafe_allow_html=True)

    running = classification_report.get("running", {})
    running_recall = safe_float(running.get("recall"))

    if running:
        st.warning(
            f"Running activity recall is {running_recall * 100:.2f}%. "
            "This indicates that the model misses some running sequences. "
            "Running predictions should therefore be interpreted cautiously."
        )

    macro_f1 = safe_float(
        classification_report.get("macro avg", {}).get("f1-score")
    )
    weighted_f1 = safe_float(
        classification_report.get("weighted avg", {}).get("f1-score")
    )

    st.write(f"**Macro average F1-score:** {macro_f1:.4f}")
    st.write(f"**Weighted average F1-score:** {weighted_f1:.4f}")

    st.caption(
        "These results represent the saved evaluation of the selected KTH model. "
        "They should not be interpreted as clinical or medical validation."
    )


# ---------------------------------------------------------
# PAGE: REPORTS & ABOUT
# ---------------------------------------------------------

elif page == "Reports & About":

    st.markdown(
        '<div class="main-title">Reports & About</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Export evaluation results and review project information.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-heading">PDF report generation</div>', unsafe_allow_html=True)

    st.write(
        "Generate a PDF report containing the saved model evaluation metrics "
        "and, when available, the latest video analysis result."
    )

    include_prediction = st.checkbox(
        "Include latest video analysis in the PDF",
        value=st.session_state.prediction_result is not None,
        disabled=st.session_state.prediction_result is None,
    )

    if st.button("Generate PDF report", use_container_width=True):
        try:
            pdf_bytes = create_pdf_report(
                evaluation,
                st.session_state.prediction_result if include_prediction else None,
            )
            st.session_state.generated_pdf = pdf_bytes
            st.success("PDF report generated.")
        except Exception as exc:
            st.error(f"PDF generation failed: {exc}")

    if st.session_state.get("generated_pdf"):
        st.download_button(
            "Download PDF report",
            data=st.session_state.generated_pdf,
            file_name="human_activity_monitor_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown('<div class="section-heading">Project information</div>', unsafe_allow_html=True)

    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.markdown(
            """
            <div class="info-panel">
                <b>Project name</b><br>
                Human Activity Health Monitor<br><br>
                <b>Recognition approach</b><br>
                Pose-based human activity recognition<br><br>
                <b>Pose estimation</b><br>
                MediaPipe Pose Landmarker
            </div>
            """,
            unsafe_allow_html=True,
        )

    with info_col2:
        st.markdown(
            """
            <div class="info-panel">
                <b>Machine learning model</b><br>
                Long Short-Term Memory (LSTM)<br><br>
                <b>Dataset</b><br>
                KTH Human Action Dataset<br><br>
                <b>Backend</b><br>
                FastAPI inference service<br><br>
                <b>Dashboard</b><br>
                Streamlit and Plotly
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown('<div class="section-heading">Model configuration</div>', unsafe_allow_html=True)

    config_data = {
        "Dataset": evaluation.get("dataset"),
        "Model type": evaluation.get("model_type"),
        "Sequence length": evaluation.get("sequence_length"),
        "Sequence stride": evaluation.get("sequence_stride"),
        "Feature dimension": evaluation.get("feature_dimension"),
        "Test sequences": evaluation.get("test_sequence_count"),
        "Test accuracy": f"{accuracy * 100:.2f}%",
        "Class names": ", ".join(class_names),
    }

    st.json(config_data)

    st.caption(
        "Human Activity Monitor is an experimental activity-recognition project. "
        "It is not a medical device and must not be used for diagnosis, "
        "emergency response, or treatment decisions."
    )

