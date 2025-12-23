AI 기반 보이스피싱·스미싱 탐지 시스템

(Hybrid NLP + Rule-based + STT Pipeline)

**1. 프로젝트 개요**

본 프로젝트는 보이스피싱(Vishing) 및 스미싱(Smishing)을 탐지하기 위한
자연어 처리(NLP) 기반 하이브리드 탐지 시스템을 구현한 연구·포트폴리오 프로젝트입니다.

문자(SMS) 및 음성(STT 변환 텍스트) 데이터를 대상으로

딥러닝 모델 + 규칙 기반(Rule-based) 로직을 결합한 하이브리드 방식

위험도 예측, 통계 분석, 시각화까지 포함한 End-to-End 파이프라인을 구현했습니다.



**2. 주요 기능**

문자/음성 텍스트 기반 피싱 탐지

한국어: KoBERT / KC-BERT / KLUE-BERT / ELECTRA 
영어: Distillbert/Bert/RoBERTa/Electra/Deberta-v3-small
등 다중 모델 비교

딥러닝 + Rule-based Hybrid 예측

STT 기반 음성 파일(mp4/wav) 처리

위험도 분포, 상위 위험 문장, 패턴 통계 시각화

실험 결과 자동 저장 (CSV / PNG)



**3. git 폴더 구조**
 
Vhishing-project/
 - audio_utils.py                # 분석용 오디오 조각 생성
 - best.pt                       # 보이스피싱 탐지용 학습 모델
 - predict.py                    # 하이브리드 탐지 모델(NLP + Rule base)
 - rule_definition_table2.csv    # 규칙 기반 탐지 용 규칙 테이블
 - smishing_datam model.txt      # 스미싱 탐지용 모델, 데이터 구글 드라이브 링크
 - smishing_data_preprocess.py   # 스미싱 데이터 전처리
 - sms.py                        # 스미싱(텍스트) 탐지 및 sms 발송
 - stt.py                        # 음성 → Text 변환(whisper)
 - test.mp4                      # 보이스피싱 탐지 시뮬레이션 용 파일(출처: 금융감독원)
 - vishing_live.py               # 보이스피싱 탐지 메인 코드(streamlit 대시보드)
 - visualization.py              # 결과 시각화
 - data/                         # 보이스피싱 학습·실험 데이터 (CSV)



**4. 사용 기술 스택**

Language: Python 3.10

NLP / DL

PyTorch

Transformer-based Language Models (Hugging Face Transformers)
  - BERT
  - DistilBERT
  - RoBERTa
  - ELECTRA
  - DeBERTa-v3-small
  - KoBERT / KC-BERT / KLUE-BERT (Korean-specific models)

Speech

STT 기반 음성 텍스트 변환

Data / Visualization

Pandas, NumPy

Matplotlib

Version Control

Git / GitHub

Git LFS (대용량 데이터 관리)



**5. 실행 방법**

환경 설정
pip install -r requirements.txt

음성 파일 기반 탐지
streamlit run vishing_live.py

mp4 / wav 파일 입력

STT → 텍스트 변환 → 피싱 위험도 예측



**6. 결과 예시**

위험도 히스토그램

위험 수준 비율 파이차트

Hybrid 점수 분포

Top-5 위험 문장
CSV 기반 정량 성능 지표

모든 결과는 실행 시 자동으로 vishing_predictions/에 저장됩니다.



**7. 프로젝트 특징**

단순 분류가 아닌 Hybrid 탐지 구조 설계

STT 기반 음성 텍스트 변환

다양한 한국어(보이스피싱)/영어(스미싱) 언어모델 비교 실험

실험 재현 가능한 결과 자동 저장 구조

연구/논문/실서비스 확장 가능 구조



**8. 참고 사항**

일부 대용량 데이터 및 모델 파일은 Git LFS와 구글 드라이브(외부 링크)로 관리됩니다.

본 프로젝트는 연구 및 학습 목적으로 제작되었습니다.



**9. Author**

이름: 공병준

분야: AI / NLP / 사용자 특화

관심 주제: 자연어 처리, 멀티모달 AI, 비전

프로젝트 논문: 자연어처리 기반 지능형 피싱 탐지 및 경보 시스템 개발(공학 석사 학위 논문)
