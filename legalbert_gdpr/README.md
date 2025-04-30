# LegalBERT for GDPR Clause Classification

This project fine-tunes a LegalBERT model to classify GDPR-related clauses in legal texts.

## 📁 Project Structure

```
legalbert_gdpr/
│
├── data/
│   └── FINAL_dataset.csv               # Your input data
│
├── models/
│   └── train_legalbert.py              # Training script
│
├── utils/
│   ├── data_utils.py                   # Data loading and preprocessing
│   ├── model_utils.py                  # (optional - modular model helpers)
│   └── evaluation_utils.py             # Evaluation and SHAP analysis
│
├── main.py                             # Main entry point (training/evaluation)
├── requirements.txt                    # Python dependencies
└── README.md                           # Project overview
```

## ✅ Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

To train and evaluate the model:

```bash
python main.py --data_path data/FINAL_dataset.csv --do_train --do_eval
```

You can change the pretrained model using `--model_name`, e.g., `nlpaueb/legal-bert-base-uncased`.

## 📊 Evaluation

- Classification report
- Confusion matrix
- SHAP interpretability support
