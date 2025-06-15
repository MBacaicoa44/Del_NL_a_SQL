"""Code used to inference the dev file with T5 finetuned"""
import os
import json
import zipfile
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from src.models.utils import generate_sql_t5, compute_bleu


if __name__ == "__main__":
    zip_path = "../../finetuned_models/t5-WikiSQL-finetuned.zip"
    model_dir = "../../finetuned_models/t5-WikiSQL-finetuned"
    dev_path = "../../data/dev.json"
    out_csv = "../../predictions/t5_finetuned.csv"

    # Crear carpeta de predicciones
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Crear carpeta del modelo si no existe
    os.makedirs(model_dir, exist_ok=True)

    # Descomprimir modelo fine-tuneado
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("../../finetuned_models/")

    # Cargar tokenizador y modelo desde la carpeta extraída
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)

    # Cargar datos de dev
    with open(dev_path, 'r', encoding='utf-8') as f:
        validation_data = json.load(f)

    # Inferencia y cálculo BLEU
    results = []
    for item in validation_data:
        question = item["question"]
        true_sql = item["query"]
        generated_sql = generate_sql_t5(question, model, tokenizer)
        bleu = compute_bleu(true_sql, generated_sql)
        results.append({
            "question": question,
            "true_sql": true_sql,
            "generated_sql": generated_sql,
            "bleu_score": bleu
        })

    # Guardar en CSV
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"Predictions saved to {out_csv}")
