"""Adaptation of the code used to evaluate the dev file with CPU."""

import os
import json
import zipfile
import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.models.utils import compute_bleu
from src.models.infer_gemma2b import generate_sql_gemma, extract_sql_from_code_tag

if __name__ == "__main__":
    zip_path = "../../finetuned_models/gemma2b-finetuned.zip"
    model_dir = "../../finetuned_models/gemma2b-finetuned"
    dev_path = "../../data/dev.json"
    out_csv = "../../predictions/gemma_finetuned.csv"

    # Crear carpetas necesarias
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall("../../finetuned_models/")

    # Cargar datos de validación
    with open(dev_path, 'r', encoding='utf-8') as f:
        validation_data = json.load(f)

    # Preparar modelo y tokenizer en el dispositivo disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carga del tokenizador y modelo base de Gemma-2b desde Hugging Face
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2b"
    )

    # Aplicar el adaptador LoRA que está en model_dir sobre el modelo base
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.config.use_cache = False
    model.to(device)

    # Inferencia y extracción de SQL
    results = []
    for sample in validation_data:
        question = sample["question"]
        true_sql = sample["query"]
        generated_sql = generate_sql_gemma(question, model, tokenizer, device)
        cleaned_sql = extract_sql_from_code_tag(generated_sql)
        bleu_score = compute_bleu(true_sql, cleaned_sql)
        results.append({
            "question": question,
            "true_sql": true_sql,
            "generated_sql": generated_sql,
            "sql_cleaned": cleaned_sql,
            "bleu_score": bleu_score,
        })

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"Predictions saved to {out_csv}")
