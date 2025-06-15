"""Code used to inference the dev file with T5"""
import os
import json
import pandas as pd

from transformers import T5Tokenizer, T5ForConditionalGeneration

from src.models.utils import generate_sql_t5, compute_bleu


if __name__ == "__main__":
    model_name = "mrm8488/t5-small-finetuned-wikiSQL"
    dev_path = "../../data/dev.json"
    out_csv = "../../predictions/t5_without_finetuning.csv"

    # Crear carpeta del output
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Cargar modelo y tokenizador
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Cargar datos de dev
    with open(dev_path, 'r', encoding='utf-8') as f:
        validation_data = json.load(f)

    results = []
    for sample in validation_data[:1]:
        question = sample["question"]
        true_sql = sample["query"]
        generated_sql = generate_sql_t5(question, model, tokenizer)
        bleu_score = compute_bleu(true_sql, generated_sql)

        results.append({
            "question": question,
            "true_sql": true_sql,
            "generated_sql": generated_sql,
            "bleu_score": bleu_score,
        })

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"Predictions saved to {out_csv}")