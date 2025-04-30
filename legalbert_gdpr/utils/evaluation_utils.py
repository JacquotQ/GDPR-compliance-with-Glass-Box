import matplotlib.pyplot as plt
import seaborn as sns
import shap
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def print_classification_report(y_true, y_pred, target_names):
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)


def compute_shap_explanations(model, tokenizer, texts, max_samples=100):
    # Use a sample subset to reduce SHAP computation time
    sample_texts = texts[:max_samples]
    tokens = tokenizer(sample_texts, padding=True, truncation=True, return_tensors="pt")

    def f(X):
        input_ids = torch.tensor(X, dtype=torch.long)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits.detach().numpy()

    explainer = shap.Explainer(f, tokens['input_ids'])
    shap_values = explainer(tokens['input_ids'])
    shap.plots.text(shap_values)
