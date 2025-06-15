"""Code used to inference the dev file with Gemma2b"""
import os
import json
import re
import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM

from src.models.utils import compute_bleu

def generate_sql_gemma(question, model, tokenizer, device):
    """
    Genera SQL con Gemma-2b para una pregunta dada.
    """
    prompt = (
        "Translate the following English question into exactly one SQL query. "
        "Do NOT output any additional examples or explanations. "
        f"Question: {question}\n </s>"
    )
    # Tokenizar y mover tensores al device
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        num_beams=5,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def extract_sql_from_code_tag(text):
    """
    1) Quita todo antes de un marcador de solución (si existe).
    2) Extrae el contenido de <code>…</code> (si lo hay).
    3) Si no empieza por SELECT, corta hasta el primer SELECT como último recurso.
    """
    # 1) Recortar antes de un marcador
    for marker in ['<h2>Step-by-Step</h2>', 'Solution:', 'Solución:', '<h2>Expert Answer</h2>']:
        if marker in text:
            text = text.split(marker, 1)[1]
            break

    # 2) Extraer <code>…</code>
    m = re.search(r'<code>([\s\S]*?)</code>', text)
    if m:
        text = m.group(1)

    # 3) Cortar hasta el primer SELECT si no empieza por él
    upp = text.upper()
    pos = upp.find('SELECT')
    if pos != -1:
        text = text[pos:]

    return text.strip()

if __name__ == "__main__":
    dev_path = "../../data/dev.json"
    out_csv = "../../predictions/gemma_without_finetuning.csv"
    model_name = "google/gemma-2b"

    # Crear carpeta de salida si no existe
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    with open(dev_path, 'r', encoding='utf-8') as f:
        validation_data = json.load(f)

    # Preparar modelo/tokenizer en GPU si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

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
