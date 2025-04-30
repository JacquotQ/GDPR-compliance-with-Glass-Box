def load_and_prepare_data(filepath, text_column='gdpr_clause', label_column='legal_area'):
    df = pd.read_csv(filepath)

    # clean na
    df = df.dropna(subset=[text_column, label_column])

    # label encoder
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df[label_column])

    # split test and val
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df[text_column].tolist(),
        df['label'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )

    return train_texts, val_texts, train_labels, val_labels, label_encoder
