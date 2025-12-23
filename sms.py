import torch
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from solapi import SolapiMessageService
from solapi.model import RequestMessage


model_dir = "./smishing_model/test" 
model = DistilBertForSequenceClassification.from_pretrained(model_dir)
tokenizer = DistilBertTokenizer.from_pretrained(model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

txt_file = "smstest.txt"
with open(txt_file, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

print(f"✅ {len(lines)}개의 문장을 읽었습니다.")

def predict_smishing(texts):
    encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    encodings = {k: v.to(device) for k, v in encodings.items()}
    with torch.no_grad():
        outputs = model(**encodings)
        probs = torch.softmax(outputs.logits, dim=-1)
    pred_labels = torch.argmax(probs, dim=-1).cpu().numpy()
    return pred_labels, probs.cpu().numpy()

pred_labels, probs = predict_smishing(lines)

results = []
for text, label, p in zip(lines, pred_labels, probs):
    smishing_prob = p[1]
    results.append({
        "text": text,
        "label": int(label),  # 0=Normal, 1=Smishing
        "smishing_prob": float(smishing_prob)
    })

df = pd.DataFrame(results)
csv_file = "./text/sms/smishing_results.csv"
df.to_csv(csv_file, index=False, encoding="utf-8-sig")
print(f"예측 결과 CSV 저장 완료: '{csv_file}'")

API_KEY = "api_key"
API_SECRET = "Secret_key"
SENDER = "phone numnber"
RECEIVER = "phone number"

service = SolapiMessageService(API_KEY, API_SECRET)

smishing_texts = [row["text"] for row in results if row["label"] == 1]

if len(smishing_texts) > 0:
    sms_content = "사용자가 스미싱 의심 문자를 받았습니다.\n"
    sms_content += "\n".join([f"- {t}" for t in smishing_texts])
    sms_content += "\n보호자의 확인이 필요합니다."

    service = SolapiMessageService(API_KEY, API_SECRET)
    message = RequestMessage(
        to=RECEIVER,
        from_=SENDER,
        text=sms_content
    )
    try:
        service.send(message)
        print(f"스미싱 문장 전체를 하나의 문자로 발송 성공!")
        print(sms_content)
    except Exception as e:
        print(f"문자 발송 실패: {e}")
    else:
        print("스미싱 문장이 없어서 발송할 메시지가 없습니다.")
