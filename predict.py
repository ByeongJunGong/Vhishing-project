import os
from dataclasses import dataclass
from typing import List, Tuple, Dict
from datetime import datetime

import torch
import torch.nn as nn
import pandas as pd
from transformers import BertTokenizer, BertModel
from konlpy.tag import Okt

import matplotlib.pyplot as plt
import seaborn as sns


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RULE_SCORE_MAX = 50
WEIGHT_NLP = 0.8
WEIGHT_RULE = 0.2

THRESHOLD_DANGER = 0.8
THRESHOLD_SUSPICIOUS = 0.5


class KoBERTClassifier(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            "monologg/kobert",
            trust_remote_code=True
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = outputs.pooler_output
        dropped = self.dropout(pooled)
        return self.classifier(dropped)

@dataclass
class AnalysisResult:
    text: str
    nlp_probability: float
    rule_score: int
    matched_patterns: List[str]
    hybrid_score: float
    risk_level: str

def load_model(model_path: str, device: torch.device):
    model = KoBERTClassifier().to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.eval()
    return model


def load_tokenizer():
    return BertTokenizer.from_pretrained(
        "monologg/kobert",
        trust_remote_code=True
    )


def load_rule_table(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def load_nlp_resources(
    model_path: str,
    rule_table_path: str
):
    return {
        "model": load_model(model_path, DEVICE),
        "tokenizer": load_tokenizer(),
        "okt": Okt(),
        "rule_table": load_rule_table(rule_table_path)
    }

def calculate_rule_score(
    text: str,
    rule_table: pd.DataFrame,
    okt: Okt
) -> Tuple[int, List[str]]:
    score = 0
    matched_patterns = []

    nouns = okt.nouns(text)
    tokens = set(nouns)

    for _, row in rule_table.iterrows():
        pattern = row["pattern"]
        pattern_type = row["rule_type"]
        pattern_score = row["default_score"]

        if pattern_type == "keyword" and pattern in tokens:
            score += pattern_score
            matched_patterns.append(pattern)

        elif pattern_type == "pattern" and pattern in text:
            score += pattern_score
            matched_patterns.append(pattern)

    return score, matched_patterns


def calculate_hybrid_score(
    nlp_prob: float,
    rule_score: int
) -> float:
    rule_score_norm = min(rule_score / RULE_SCORE_MAX, 1.0)
    return (
        WEIGHT_NLP * nlp_prob
        + WEIGHT_RULE * rule_score_norm
    )


def classify_risk_level(hybrid_score: float) -> str:
    if hybrid_score >= THRESHOLD_DANGER:
        return "위험"
    elif hybrid_score >= THRESHOLD_SUSPICIOUS:
        return "의심"
    return "정상"

def infer_nlp_probability(
    text: str,
    model,
    tokenizer
) -> float:
    encoding = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        return torch.sigmoid(logits).item()


def analyze_text(
    text: str,
    resources: Dict
) -> AnalysisResult:
    nlp_prob = infer_nlp_probability(
        text,
        resources["model"],
        resources["tokenizer"]
    )

    rule_score, matched_patterns = calculate_rule_score(
        text,
        resources["rule_table"],
        resources["okt"]
    )

    hybrid_score = calculate_hybrid_score(
        nlp_prob,
        rule_score
    )

    risk_level = classify_risk_level(hybrid_score)

    return AnalysisResult(
        text=text,
        nlp_probability=round(nlp_prob, 4),
        rule_score=rule_score,
        matched_patterns=matched_patterns,
        hybrid_score=round(hybrid_score, 4),
        risk_level=risk_level
    )

def analyze_all_and_save(
    results: List[AnalysisResult],
    pattern_counter: Dict[str, int]
):
    df = pd.DataFrame([r.__dict__ for r in results])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"vishing_predictions/{timestamp}"
    os.makedirs(base_dir, exist_ok=True)

    df.to_csv(
        f"{base_dir}/results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    danger_df = df[df["risk_level"] == "위험"]
    top5 = danger_df.sort_values(
        by="hybrid_score",
        ascending=False
    ).head(5)

    risk_ratio = (
        len(danger_df) / len(df)
        if len(df) > 0 else 0
    )

    # Charts
    plt.figure(figsize=(10, 6))
    sns.histplot(df["hybrid_score"], bins=20, kde=True)
    plt.title("Hybrid Score Distribution")
    plt.savefig(f"{base_dir}/hybrid_hist.png")
    plt.close()

    plt.figure(figsize=(5, 5))
    counts = df["risk_level"].value_counts()
    plt.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=140
    )
    plt.title("Risk Level")
    plt.savefig(f"{base_dir}/risk_pie.png")
    plt.close()

    pattern_df = pd.DataFrame(
        sorted(pattern_counter.items(), key=lambda x: x[1], reverse=True),
        columns=["pattern", "count"]
    )

    return df, top5, risk_ratio, pattern_df

