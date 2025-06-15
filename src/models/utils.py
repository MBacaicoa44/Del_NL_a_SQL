import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def generate_sql_t5(query, model, tokenizer):
    """Genera SQL a partir de una pregunta usando T5."""
    input_text = f"translate English to SQL: {query} </s>"
    tokens = tokenizer([input_text], return_tensors="pt", padding=True, truncation=True)
    out = model.generate(
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
        max_length=100,
        num_beams=5,
        early_stopping=True,
        repetition_penalty=1.2
    )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()

def tokenize_sql(sql):
    """Aplica los 15 pasos de limpieza y tokenización al SQL."""
    if sql is None:
        return ""
        # Paso 1: minúsculas y limpieza inicial
    sql = sql.lower().strip()
    # Paso 2: eliminar punto y coma final
    sql = sql.rstrip(';')
    # Paso 3: normalizar comillas tipográficas y dobles
    sql = sql.replace("''", "'").replace('"', "'")
    sql = sql.replace("“", "'").replace("”", "'").replace("‘", "'").replace("’", "'")
    # Paso 4: eliminar backticks
    sql = sql.replace('`', '')
    # Paso 5: quitar espacios múltiples
    sql = re.sub(r'\s+', ' ', sql)
    # Eliminar comas en los números
    sql = re.sub(r'(?<=\d),(?=\d)', '', sql)
    # Paso 6: quitar espacios alrededor de símbolos comunes (excepto comillas simples)
    sql = re.sub(r"\s*([,()=<>])\s*", r"\1", sql)
    # Paso 7: añadir espacio si después de símbolo va letra, número o comilla simple (sin afectar funciones)
    sql = re.sub(r"([,)=<>])([a-zA-Z0-9'])", r"\1 \2", sql)
    # Paso 8: añadir espacio si antes de =, < o > va una letra o número
    sql = re.sub(r"([a-zA-Z0-9])([=<>])", r"\1 \2", sql)
    # Paso 9: añadir espacio entre paréntesis de cierre y operadores
    sql = re.sub(r"(\))([=<>])", r"\1 \2", sql)
    # Paso 10: eliminar paréntesis redundantes como (a = b)
    sql = re.sub(r'\((\s*\w+\s*[=<>!]+\s*\w+\s*)\)', r'\1', sql)
    # Paso 11: añadir espacio después de comillas de cierre si están pegadas a palabras
    sql = re.sub(r"('(?:[^']+)')(?=[a-zA-Z])", r"\1 ", sql)
    # Paso 12: añadir espacio entre operadores y paréntesis de apertura
    sql = re.sub(r"([=<>])(\()", r"\1 \2", sql)
    # Paso 13: añadir espacio antes de paréntesis de apertura si está pegado a palabra, EXCEPTO funciones comunes
    def espacio_antes_parentesis(match):
        palabra = match.group(1)
        funciones = {"avg", "count", "sum", "max", "min"}
        return palabra + "(" if palabra in funciones else palabra + " ("
    sql = re.sub(r"([a-zA-Z_][a-zA-Z0-9_]*)\(", espacio_antes_parentesis, sql)
    # Paso 14: eliminar espacios dentro de strings entre comillas simples (ej. ' ca ' → 'ca')
    sql = re.sub(r"' *([^']*?) *'", lambda m: f"'{m.group(1).strip()}'", sql)
    # Paso 15: limpieza final de espacios
    sql = re.sub(r"\s+", " ", sql).strip()
    return sql


def compute_bleu(true_sql, generated_sql):
    """Calcula BLEU con suavizado usando tokenize_sql."""
    smooth = SmoothingFunction().method1
    return sentence_bleu(
        [tokenize_sql(true_sql)],
        tokenize_sql(generated_sql),
        smoothing_function=smooth
    )