# predict_sentence.py
import torch
import torch.nn as nn
import pandas as pd
from transformers import BertTokenizer, BertModel
from konlpy.tag import Okt
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RULE_SCORE_MAX = 50
WEIGHT_NLP = 0.8
WEIGHT_RULE = 0.2
THRESHOLD_DANGER = 0.8
THRESHOLD_SUSPICIOUS = 0.5

class KoBERTClassifier(nn.Module):
    def __init__(self, dropout=0.3):
        super(KoBERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("monologg/kobert", trust_remote_code=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        dropped = self.dropout(pooled)
        return self.classifier(dropped)

model = KoBERTClassifier().to(DEVICE)
model.load_state_dict(torch.load("saved_model/best.pt", map_location=DEVICE))
model.eval()
tokenizer = BertTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
okt = Okt()
rule_table = pd.read_csv('data/rule_definition_table2.csv', encoding='utf-8-sig')

def calculate_rule_score(text, rule_table):
    score = 0
    matched_patterns = []
    nouns = okt.nouns(text)
    tokens = set(nouns)

    for idx, row in rule_table.iterrows():
        pattern = row['pattern']
        pattern_type = row['rule_type']
        pattern_score = row['default_score']

        if pattern_type == 'keyword' and pattern in tokens:
            score += pattern_score
            matched_patterns.append(pattern)
        elif pattern_type == 'pattern' and pattern in text:
            score += pattern_score
            matched_patterns.append(pattern)
    return score, matched_patterns

def calculate_hybrid_score(nlp_prob, rule_score, rule_score_max=50, weight_nlp=0.7, weight_rule=0.3):
    rule_score_norm = min(rule_score / rule_score_max, 1.0)
    return (weight_nlp * nlp_prob) + (weight_rule * rule_score_norm)

def classify_risk_level(hybrid_score):
    if hybrid_score >= THRESHOLD_DANGER:
        return "위험"
    elif hybrid_score >= THRESHOLD_SUSPICIOUS:
        return "의심"
    else:
        return "정상"

def analyze_text(text):
    encoding = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        nlp_prob = torch.sigmoid(logits).item()

    rule_score, matched_patterns = calculate_rule_score(text, rule_table)
    hybrid_score = calculate_hybrid_score(nlp_prob, rule_score, RULE_SCORE_MAX, WEIGHT_NLP, WEIGHT_RULE)
    risk_level = classify_risk_level(hybrid_score)

    return {
        "text": text,
        "nlp_probability": round(nlp_prob, 4),
        "rule_score": rule_score,
        "matched_patterns": ', '.join(matched_patterns),
        "hybrid_score": round(hybrid_score, 4),
        "risk_level": risk_level
    }

def analyze_all_and_save(results, pattern_counter):
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("vishing_predictions", exist_ok=True)
    df.to_csv(f"vishing_predictions/result_{timestamp}.csv", index=False, encoding='utf-8-sig')

    # Top 5
    danger_df = df[df['risk_level'] == '위험']
    top5 = danger_df.sort_values(by='hybrid_score', ascending=False).head(5)

    # 위험 비율 계산
    risk_ratio = len(danger_df) / len(df) if len(df) > 0 else 0

    # Charts
    chart_dir = f"vishing_predictions/figs_{timestamp}"
    os.makedirs(chart_dir, exist_ok=True)

    plt.figure(figsize=(10,6))
    sns.histplot(df['hybrid_score'], bins=20, kde=True)
    plt.title('Hybrid Score Distribution')
    plt.savefig(f"{chart_dir}/hybrid_hist.png")
    plt.close()

    plt.figure(figsize=(5,5))
    counts = df['risk_level'].value_counts()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
    plt.title("Risk Level")
    plt.savefig(f"{chart_dir}/risk_pie.png")
    plt.close()

    pattern_counter_df = pd.DataFrame(sorted(pattern_counter.items(), key=lambda x: x[1], reverse=True),
                                      columns=['pattern', 'count'])

    return df, top5, risk_ratio, {
        "hybrid": f"{chart_dir}/hybrid_hist.png",
        "pie": f"{chart_dir}/risk_pie.png"
    }, pattern_counter_df
