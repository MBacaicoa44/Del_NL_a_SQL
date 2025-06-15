"""Code used to classify the samples"""
import re
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")

def clasificar_sql(query: str) -> str:
    # Pasamos la consulta a minúsculas para simplificar la detección de patrones
    q = query.lower()

    # Palabras clave que indican agregación y operaciones complejas
    aggregations = ["count(", "sum(", "avg(", "min(", "max("]
    complex_ops = ["union", "intersect", "except", "case when"]

    # Flags booleanos para detectar características de la consulta
    has_join = "join" in q
    has_group_by = "group by" in q
    has_agg = any(agg in q for agg in aggregations)
    has_where = "where" in q
    has_having = "having" in q
    has_subquery = len(re.findall(r"\(\s*select", q)) >= 1
    has_nested_subquery = len(re.findall(r"\(\s*select", q)) >= 2
    has_complex_operator = any(op in q for op in complex_ops)

    # Subconsultas dentro de IN, EXISTS, etc.
    has_in_select = any(
        re.search(pat, q) for pat in [
            r"\bin\s*\(\s*select",
            r"\bnot\s+in\s*\(\s*select",
            r"\bexists\s*\(\s*select",
            r"\bnot\s+exists\s*\(\s*select"
        ]
    )
    has_subquery_with_agg = has_subquery and has_agg

    # Clasificación por niveles de complejidad (jerarquía descendente)
    if (
        has_nested_subquery
        or (has_join and has_subquery)
        or (has_join and has_group_by and has_subquery)
        or has_complex_operator
    ):
        return "7. Consulta compleja y anidada"
    if has_join and has_agg:
        return "6. JOIN con agregación"
    if has_join:
        return "5. JOIN sin agregación"
    if (
        (has_agg and (has_group_by or has_having or has_in_select))
        or has_in_select
        or has_subquery_with_agg
    ):
        return "4. Agregación con lógica adicional o subconsulta simple"
    if has_agg:
        return "3. Consulta con agregación simple"
    if has_where:
        return "2. Consulta con condiciones"
    return "1. Consulta simple"

def categorize_question(text):
    """
    Clasifica la pregunta en uno de los 7 niveles de complejidad lingüística.
    """

    # Procesamos el texto con spaCy
    doc = nlp(text)
    tokens = [t for t in doc if not t.is_punct]
    lower_tokens = {t.lower_ for t in doc}
    num_sents = len(list(doc.sents))
    num_ents = len(doc.ents)
    num_tokens = len(tokens)

    # 1. Pregunta muy corta
    if num_tokens <= 7:
        return "1. Consulta directa y corta"

    # 2. Contiene cuantificadores
    if lower_tokens & {"all", "any", "every", "none"}:
        return "2. Consulta con cuantificadores"

    # Detección de conectores, cláusulas y estructuras relevantes
    has_and_or = any(t.lower_ in {"and", "or"} for t in doc)
    has_between_having_where = any(t.lower_ in {"between", "having", "where"} for t in doc)
    has_join_group_order = any(t.lower_ in {"join", "group", "order"} for t in doc)
    has_relcl = any(t.dep_ == "relcl" for t in doc) or bool(lower_tokens & {"who", "which", "that"})
    has_if = 'if' in lower_tokens

    # 7. Estructura gramatical compleja
    if num_sents > 1 or num_ents >= 2:
        return "7. Gramaticalmente compleja"

    # 6. Pregunta condicional o extensa
    if has_if or num_tokens > 15:
        return "6. Condicional o larga/cargada"

    # 5. Contiene cláusula relativa
    if has_relcl:
        return "5. Cláusula relativa"

    # 4. Estructura ligeramente compuesta
    if has_and_or or has_between_having_where or has_join_group_order:
        return "4. Ligeramente compuesta"

    # 3. Caso general
    return "3. Afirmativa básica"


def process_file(path: str):
    """
    Carga el CSV en 'path', añade 'sql_level' Y "question_level", luego lo guarda
    sobre el mismo fichero.
    """
    df = pd.read_csv(path)

    # Aplicar las funciónes de clasificación
    df["sql_level"] = df["true_sql"].fillna("").apply(clasificar_sql)
    df["question_level"] = df["question"].fillna("").apply(categorize_question)


    # Guardar de nuevo el CSV
    df.to_csv(path, index=False)
    print(f"Procesado y guardado: {path}")


if __name__ == "__main__":
    files = [
        "../../predictions/t5_without_finetuning.csv",
        "../../predictions/t5_finetuned.csv",
        "../../predictions/gemma_without_finetuning.csv",
        "../../predictions/gemma_finetuned.csv"
    ]
    for f in files:
        process_file(f)

