import os
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import streamlit as st
import torch
import pandas as pd
import matplotlib
import matplotlib.font_manager as fm

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)

from stt import transcribe_with_segments
from predict import analyze_text, analyze_all_and_save
from visualization import draw_pie_chart, draw_histogram

from solapi import SolapiMessageService
from solapi.model import RequestMessage


@dataclass(frozen=True)
class AppConfig:
    enable_sms: bool = False
    min_total_texts: int = 10
    min_risk_ratio: float = 0.5
    ui_update_delay: float = 1.5
    top_k_sentences: int = 5

    api_key: str = ""
    api_secret: str = ""
    sender: str = "-"
    receiver: str = "-"

    upload_dir: str = "uploaded"
    smishing_model_dir: str = "./smishing_model/test"


RISK_DANGER = "위험"
RISK_SUSPICIOUS = "의심"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    text: str
    risk_level: str
    hybrid_score: float
    matched_patterns: List[str]


@dataclass
class DetectionStats:
    danger_count: int = 0
    suspicious_count: int = 0
    pattern_counter: Dict[str, int] = field(default_factory=dict)
    results: List[AnalysisResult] = field(default_factory=list)
    top_dangerous_sentences: List[Dict] = field(default_factory=list)

    def total_count(self) -> int:
        return len(self.results)

    def danger_ratio(self) -> float:
        if self.total_count() == 0:
            return 0.0
        return self.danger_count / self.total_count()


def setup_font():
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    if not os.path.exists(font_path):
        font_path = "C:/Windows/Fonts/malgun.ttf"

    if os.path.exists(font_path):
        font_prop = fm.FontProperties(fname=font_path)
        matplotlib.rc("font", family=font_prop.get_name())


def save_uploaded_file(uploaded_file, upload_dir: str) -> str:
    os.makedirs(upload_dir, exist_ok=True)
    path = os.path.join(upload_dir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.read())
    return path


def update_statistics(
    stats: DetectionStats,
    result: AnalysisResult,
    config: AppConfig
):
    stats.results.append(result)

    for p in result.matched_patterns:
        stats.pattern_counter[p] = stats.pattern_counter.get(p, 0) + 1

    if result.risk_level == RISK_DANGER:
        stats.danger_count += 1
        stats.top_dangerous_sentences.append({
            "text": result.text,
            "hybrid_score": result.hybrid_score
        })
        stats.top_dangerous_sentences = sorted(
            stats.top_dangerous_sentences,
            key=lambda x: x["hybrid_score"],
            reverse=True
        )[:config.top_k_sentences]

    elif result.risk_level == RISK_SUSPICIOUS:
        stats.suspicious_count += 1


def should_trigger_alert(stats: DetectionStats, config: AppConfig) -> bool:
    return (
        stats.total_count() >= config.min_total_texts
        and stats.danger_ratio() >= config.min_risk_ratio
    )


def send_sms_alert(sentences: List[Dict], config: AppConfig):
    sms_content = "\n".join(
        f"{i+1}. {s['text']}" for i, s in enumerate(sentences)
    )
    service = SolapiMessageService(config.api_key, config.api_secret)
    message = RequestMessage(
        to=config.receiver,
        from_=config.sender,
        text=sms_content
    )
    service.send(message)


@st.cache_resource
def load_smishing_model(model_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(device)
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer, device


def predict_smishing(
    texts: List[str],
    model,
    tokenizer,
    device
) -> Tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=-1)
    return torch.argmax(probs, dim=-1), probs[:, 1]


def process_audio_detection(uploaded_file, config: AppConfig):
    audio_path = save_uploaded_file(uploaded_file, config.upload_dir)

    try:
        segments = transcribe_with_segments(audio_path)
    except Exception as e:
        st.error(f"음성 인식 실패: {e}")
        return

    stats = DetectionStats()
    sent_sms = False

    alert_area = st.empty()
    sentence_area = st.empty()
    chart_area = st.empty()

    for idx, seg in enumerate(segments):
        raw = analyze_text(seg["text"])
        result = AnalysisResult(
            text=seg["text"],
            risk_level=raw["risk_level"],
            hybrid_score=raw["hybrid_score"],
            matched_patterns=raw["matched_patterns"].split(", ")
        )

        update_statistics(stats, result, config)

        if should_trigger_alert(stats, config):
            alert_area.error(
                f"보이스피싱 위험 비율 {stats.danger_ratio():.1%} 초과"
            )

            if config.enable_sms and not sent_sms:
                send_sms_alert(stats.top_dangerous_sentences, config)
                sent_sms = True

        with sentence_area.container():
            st.markdown(f"### 통화 내역 {idx + 1}")
            st.write(result.text)
            st.json(raw)

        with chart_area.container():
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(draw_pie_chart(
                    stats.danger_count,
                    stats.suspicious_count,
                    stats.total_count()
                ))
            with col2:
                st.pyplot(draw_histogram(
                    [r.hybrid_score for r in stats.results]
                ))

        time.sleep(config.ui_update_delay)

    analyze_all_and_save(
        [r.__dict__ for r in stats.results],
        stats.pattern_counter
    )
    st.success("보이스피싱 탐지 종료")


def process_smishing_detection(uploaded_file, config: AppConfig):
    lines = uploaded_file.read().decode("utf-8").splitlines()
    lines = [l.strip() for l in lines if l.strip()]

    model, tokenizer, device = load_smishing_model(config.smishing_model_dir)
    labels, probs = predict_smishing(lines, model, tokenizer, device)

    df = pd.DataFrame({
        "text": lines,
        "label": labels.cpu().numpy(),
        "smishing_prob": probs.cpu().numpy()
    })

    st.dataframe(df)

    if (labels == 1).any():
        st.error("스미싱 의심 문자 탐지")


def main():
    config = AppConfig()
    setup_font()

    st.set_page_config(
        page_title="피싱 자동 탐지 시스템",
        layout="wide"
    )
    st.title("보이스피싱 · 스미싱 자동 탐지 시스템")

    uploaded_file = st.file_uploader(
        "파일 업로드 (mp4 = 보이스피싱 / txt = 스미싱)",
        type=["mp4", "txt"]
    )

    if not uploaded_file:
        return

    ext = uploaded_file.name.split(".")[-1].lower()

    if ext == "mp4":
        st.success("통화 파일 감지 → 보이스피싱 탐지 시작")
        process_audio_detection(uploaded_file, config)

    elif ext == "txt":
        st.success("문자 파일 감지 → 스미싱 탐지 시작")
        process_smishing_detection(uploaded_file, config)

    else:
        st.error("지원하지 않는 파일 형식입니다.")


if __name__ == "__main__":
    main()
