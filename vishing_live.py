import streamlit as st
import os
import time
import pandas as pd
import matplotlib
import matplotlib.font_manager as fm
from stt import transcribe_with_segments
from predict import analyze_text, analyze_all_and_save
from visualization import draw_pie_chart, draw_histogram
from solapi import SolapiMessageService
from solapi.model import RequestMessage

# 폰트 설정
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if not os.path.exists(font_path):
    font_path = "C:/Windows/Fonts/malgun.ttf"
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    matplotlib.rc("font", family=font_prop.get_name())
else:
    print("한글 폰트 파일을 찾을 수 없습니다.")

# 페이지 설정
st.set_page_config(page_title="보이스피싱 탐지", layout="wide")
st.title("실시간 보이스피싱 탐지 상황")

# 환경 변수 및 세션 초기화
API_KEY = "Api_key"
API_SECRET = "secret_key"
SENDER = "-"
RECEIVER = "-"
min_total_texts = 10
min_risk_ratio = 0.5

if "confirmed" not in st.session_state:
    st.session_state.confirmed = False

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

# 파일 업로드(결과 확인+성능 테스트용)
if not st.session_state.file_uploaded:
    uploaded_file = st.file_uploader("통화 내역(mp4) 업로드", type=["mp4"])
    if uploaded_file:
        os.makedirs("uploaded", exist_ok=True)
        audio_path = os.path.join("uploaded", uploaded_file.name)
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.read())
        st.session_state.audio_path = audio_path
        st.session_state.file_uploaded = True
        st.rerun()

# 분석 시작
if st.session_state.file_uploaded:
    audio_path = st.session_state.audio_path
    segments = transcribe_with_segments(audio_path)

    status_placeholder = st.empty()
    status_placeholder.success("통화 감지. 보이스피싱 탐지 시작...")
    time.sleep(5)

    spinner_html = """
    <div style="display: flex; align-items: center;">
        <div class="loader" style="border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 18px; height: 18px; animation: spin 1s linear infinite; margin-right: 8px;"></div>
        <strong>실시간 보이스피싱 탐지 중...</strong>
    </div>
    <style>
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
    </style>
    """
    status_placeholder.markdown(spinner_html, unsafe_allow_html=True)

    results = []
    pattern_counter = {}
    danger_count = 0
    suspicious_count = 0
    threshold = min_risk_ratio
    sent_sms = False
    top_dangerous_sentences = []

    alert_area = st.empty()
    sentence_area = st.empty()
    chart_area = st.empty()

    for i, seg in enumerate(segments):
        text = seg['text']
        result = analyze_text(text)
        results.append(result)

        for p in result['matched_patterns'].split(', '):
            if p:
                pattern_counter[p] = pattern_counter.get(p, 0) + 1

        if result['risk_level'] == '위험':
            danger_count += 1
            top_dangerous_sentences.append({'text': text, 'hybrid_score': result['hybrid_score']})
            top_dangerous_sentences = sorted(top_dangerous_sentences, key=lambda x: x['hybrid_score'], reverse=True)[:5]
        elif result['risk_level'] == '의심':
            suspicious_count += 1

        total_count = i + 1
        danger_ratio = danger_count / total_count if total_count > 0 else 0

        # 경고 및 문자 발송
        if total_count >= min_total_texts and danger_ratio >= threshold and not st.session_state.confirmed:
            with alert_area.container():
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.error(f"보이스피싱 위험 문장 비율 {danger_ratio:.1%} 초과! 정상 통화일 경우 '정상 통화입니다.' 클릭")
                with col2:
                    if st.button("정상 통화입니다", key=f"confirm_btn_once_{i}"):
                        st.session_state.confirmed = True
                        alert_area.success("사용자가 확인했습니다.")

            if not sent_sms and len(top_dangerous_sentences) > 0:
                summary_text = "\n".join([f"{j+1}. {s['text']}" for j, s in enumerate(top_dangerous_sentences)])
                sms_content = f"[경고] 위험 비율 {danger_ratio:.1%} 감지\nTop5:\n" + summary_text
                try:
                    service = SolapiMessageService(API_KEY, API_SECRET)
                    message = RequestMessage(
                        to=RECEIVER,
                        from_=SENDER,
                        text=sms_content
                    )
                    service.send(message)
                    st.warning("보이스피싱 의심 경고 문자가 발송되었습니다.")
                    sent_sms = True
                except Exception as e:
                    st.error(f"문자 발송 실패: {str(e)}")
        elif st.session_state.confirmed:
            alert_area.success("사용자가 이미 확인했습니다.")
        elif danger_ratio < 0.05:
            alert_area.empty()
        else:
            alert_area.info(f"현재 위험 문장 비율: {danger_ratio:.1%}")

        # 문장 출력
        with sentence_area.container():
            st.markdown(f"### 통화 내역 {i+1}")
            st.markdown(f"**문장 내용:** {text}")
            st.markdown(f"- NLP 확률: {result['nlp_probability']:.3f}")
            st.markdown(f"- Rule 점수: {result['rule_score']}")
            st.markdown(f"- 매칭 패턴: {result['matched_patterns'] or '없음'}")
            st.markdown(f"- Hybrid 점수: {result['hybrid_score']:.3f}")
            st.markdown(f"- 위험 등급: **{result['risk_level']}**")

        # 차트 출력
        with chart_area.container():
            col_left, col_right = st.columns([1, 1])
            with col_left:
                st.markdown("#### 위험도 분포")
                fig1 = draw_pie_chart(danger_count, suspicious_count, total_count, figsize=(1.5, 2.5), fontsize=8)
                st.pyplot(fig1)
            with col_right:
                st.markdown("#### 위험 점수 분포")
                fig2 = draw_histogram([r['hybrid_score'] for r in results], figsize=(3.5, 2), fontsize=8)
                st.pyplot(fig2)

        time.sleep(max(len(text) * 0.1, 2.5))

    df, top5, risk_ratio, charts, pattern_counter_df = analyze_all_and_save(results, pattern_counter)
    status_placeholder.success("통화 및 보이스피싱 탐지 종료.")
    st.success(f"결과 저장 완료: vishing_predictions/result_*.csv")

    st.subheader("최종 Top 5 위험 문장")
    if top5 is not None and len(top5) > 0:
        for idx, row in top5.iterrows():
            st.markdown(f"**{idx+1}.** {row['text']}")
            st.markdown(f"- 점수: {row['hybrid_score']} | 등급: {row['risk_level']}")
    else:
        st.markdown("(위험 문장이 감지되지 않았습니다.)")
