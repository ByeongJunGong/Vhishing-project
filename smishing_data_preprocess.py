import pandas as pd
import re

file_path = "data/text/analysisdataset.csv"

try:
    analysis_df = pd.read_csv(file_path, encoding="utf-8")
except UnicodeDecodeError:
    print("⚠ UTF-8 로딩 실패 → cp949로 재시도")
    try:
        analysis_df = pd.read_csv(file_path, encoding="cp949")
    except UnicodeDecodeError:
        print("⚠ cp949 로딩 실패 → latin1로 재시도")
        analysis_df = pd.read_csv(file_path, encoding="latin1")

print(f"✅ analysisdataset.csv 로딩 완료! 데이터 크기: {analysis_df.shape}")


analysis_df["label"] = analysis_df["Phishing"].apply(lambda x: 1 if x == 1 else 0)
analysis_df = analysis_df[["MainText", "label"]].rename(columns={"MainText": "text"})

smish_df = pd.read_csv("data/text/SMSSmishCollection.txt", sep="\t", header=None, names=["type", "text"])
smish_df["label"] = smish_df["type"].apply(lambda x: 1 if x == "smish" else 0)
smish_df = smish_df[["text", "label"]]

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"http\S+", "URL", text)          # URL 마스킹
    text = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", text) # 특수문자 제거
    text = re.sub(r"\s+", " ", text).strip()        # 공백 정리
    return text

analysis_df["text"] = analysis_df["text"].apply(clean_text)
smish_df["text"] = smish_df["text"].apply(clean_text)

combined_df = pd.concat([analysis_df, smish_df], ignore_index=True)


combined_df.drop_duplicates(subset=["text"], inplace=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

combined_df.to_csv("combined_smishing_dataset.csv", index=False, encoding="utf-8-sig")

print("✅ 병합 데이터셋 저장 완료!")
print(f"총 샘플 수: {len(combined_df)}")
print(f"스미싱 비율: {combined_df['label'].mean():.2%}")
print(combined_df.head())
