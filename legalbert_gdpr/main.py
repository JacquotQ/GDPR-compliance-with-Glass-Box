import argparse
import os
from utils.data_utils import load_and_prepare_data
from models.train_legalbert import train_model
from utils.evaluation_utils import plot_confusion_matrix, print_classification_report
from sklearn.metrics import classification_report


def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate LegalBERT on GDPR Dataset")
    parser.add_argument('--data_path', type=str, required=True, help='Path to FINAL_dataset.csv')
    parser.add_argument('--model_name', type=str, default='nlpaueb/legal-bert-base-uncased', help='Pretrained model name')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save the model and tokenizer')
    parser.add_argument('--do_train', action='store_true', help='Flag to train model')
    parser.add_argument('--do_eval', action='store_true', help='Flag to evaluate model')

    args = parser.parse_args()

    train_texts, val_texts, train_labels, val_labels, label_encoder = load_and_prepare_data(args.data_path)

    if args.do_train:
        trainer, model, tokenizer = train_model(train_texts, val_texts, train_labels, val_labels, args.model_name, args.output_dir)
    else:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)

    if args.do_eval:
        from transformers import Trainer
        from datasets import Dataset

        def tokenize(tokenizer, texts, labels):
            encodings = tokenizer(texts, truncation=True, padding=True)
            return Dataset.from_dict({
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask'],
                'labels': labels
            })

        val_dataset = tokenize(tokenizer, val_texts, val_labels)
        trainer = Trainer(model=model)
        predictions = trainer.predict(val_dataset)
        y_pred = predictions.predictions.argmax(-1)

        print_classification_report(val_labels, y_pred, label_encoder.classes_)
        plot_confusion_matrix(val_labels, y_pred, label_encoder.classes_)


if __name__ == '__main__':
    main()
