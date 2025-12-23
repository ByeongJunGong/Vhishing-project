import streamlit as st
import os
import time
import torch
import pandas as pd
import matplotlib
import matplotlib.font_manager as fm

from stt import transcribe_with_segments
from predict import analyze_text, analyze_all_and_save
from visualization import draw_pie_chart, draw_histogram

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from solapi import SolapiMessageService
from solapi.model import RequestMessage

# ======================================================
# 공통 설정
# ======================================================
ENABLE_SMS = False   # ★ SMS 미사용 시 False
API_KEY = ""
API_SECRET = ""
SENDER = "-"
RECEIVER = "-"

min_total_texts = 10
min_risk_ratio = 0.5

# ======================================================
# 폰트 설정 (기존 코드 유지)
# ======================================================
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if not os.path.exists(font_path):
    font_path = "C:/Windows/Fonts/malgun.ttf"
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    matplotlib.rc("font", family=font_prop.get_name())

# ======================================================
# 페이지 설정
# ======================================================
st.set_page_config(page_title="피싱 자동 탐지 시스템", layout="wide")
st.title("📡 보이스피싱 · 스미싱 자동 탐지 시스템")

# ======================================================
# 세션 상태
# ======================================================
if "confirmed" not in st.session_state:
    st.session_state.confirmed = False
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

# ======================================================
# 파일 업로드 (자동 탐지 핵심)
# ======================================================
uploaded_file = st.file_uploader(
    "파일 업로드 (mp4 = 보이스피싱 / txt = 스미싱)",
    type=["mp4", "txt"]
)

if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()

# ======================================================
# 1️⃣ 보이스피싱 자동 실행 (mp4)
# ======================================================
if uploaded_file and file_ext == "mp4":

    os.makedirs("uploaded", exist_ok=True)
    audio_path = os.path.join("uploaded", uploaded_file.name)
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("📞 통화 파일 감지 → 보이스피싱 탐지 시작")

    segments = transcribe_with_segments(audio_path)

    status_placeholder = st.empty()
    status_placeholder.success("📝 통화 감지. 보이스피싱 탐지 시작...")
    time.sleep(3)

    results = []
    pattern_counter = {}
    danger_count = 0
    suspicious_count = 0
    sent_sms = False
    top_dangerous_sentences = []

    alert_area = st.empty()
    sentence_area = st.empty()
    chart_area = st.empty()

    for i, seg in enumerate(segments):
        text = seg["text"]
        result = analyze_text(text)
        results.append(result)

        for p in result["matched_patterns"].split(", "):
            if p:
                pattern_counter[p] = pattern_counter.get(p, 0) + 1

        if result["risk_level"] == "위험":
            danger_count += 1
            top_dangerous_sentences.append(
                {"text": text, "hybrid_score": result["hybrid_score"]}
            )
            top_dangerous_sentences = sorted(
                top_dangerous_sentences,
                key=lambda x: x["hybrid_score"],
                reverse=True
            )[:5]

        elif result["risk_level"] == "의심":
            suspicious_count += 1

        total_count = i + 1
        danger_ratio = danger_count / total_count

        # 경고 영역 (기존 로직 유지)
        if total_count >= min_total_texts and danger_ratio >= min_risk_ratio:
            alert_area.error(
                f"🚨 보이스피싱 위험 비율 {danger_ratio:.1%} 초과"
            )

            if ENABLE_SMS and not sent_sms:
                sms_content = "\n".join(
                    [f"{i+1}. {s['text']}" for i, s in enumerate(top_dangerous_sentences)]
                )
                service = SolapiMessageService(API_KEY, API_SECRET)
                message = RequestMessage(
                    to=RECEIVER, from_=SENDER, text=sms_content
                )
                service.send(message)
                sent_sms = True

        # 문장 출력
        with sentence_area.container():
            st.markdown(f"### 통화 내역 {i+1}")
            st.write(text)
            st.json(result)

        # 차트
        with chart_area.container():
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(draw_pie_chart(
                    danger_count, suspicious_count, total_count
                ))
            with col2:
                st.pyplot(draw_histogram(
                    [r["hybrid_score"] for r in results]
                ))

        time.sleep(1.5)

    analyze_all_and_save(results, pattern_counter)
    st.success("✅ 보이스피싱 탐지 종료")

# ======================================================
# 2️⃣ 스미싱 자동 실행 (txt)
# ======================================================
elif uploaded_file and file_ext == "txt":

    st.success("💬 문자 파일 감지 → 스미싱 탐지 시작")

    lines = uploaded_file.read().decode("utf-8").splitlines()
    lines = [l.strip() for l in lines if l.strip()]

    model_dir = "./smishing_model/test"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(device)
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model.eval()

    def predict(texts):
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            probs = torch.softmax(out.logits, dim=-1)
        return torch.argmax(probs, dim=-1), probs[:, 1]

    labels, probs = predict(lines)

    df = pd.DataFrame({
        "text": lines,
        "label": labels.cpu().numpy(),
        "smishing_prob": probs.cpu().numpy()
    })

    st.dataframe(df)

    if (labels == 1).any():
        st.error("🚨 스미싱 의심 문자 탐지")

# ======================================================
# 기타 파일
# ======================================================
elif uploaded_file:
    st.error("❌ 지원하지 않는 파일 형식입니다.")
