import json
import os

DATA_DIR = "../../data/"
TRAIN_FILE = os.path.join(DATA_DIR, "train_spider.json")
OTHERS_FILE = os.path.join(DATA_DIR, "train_others.json")
OUTPUT_DIR = "../../processed/"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "train_data.json")


def load_json(path):
    """
    Carga un archivo JSON y devuelve su lista de objetos.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def combine_data(train, others):
    """
    Combina dos listas de datos en una sola.
    """
    return train + others

def simplify(data):
    """
    Extraer datos relevantes.
    """
    return [
        {"db_id": item["db_id"],
         "question": item["question"],
         "query": item["query"]}
        for item in data
    ]

def save_json(data, path):
    """
    Guarda la lista de objetos en un archivo csv.
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_output_dir(path):
    """
    Crea la carpeta de salida si no existe.
    """
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == "__main__":
    # Preparar directorio de salida
    ensure_output_dir(OUTPUT_DIR)

    # Cargar datos
    train = load_json(TRAIN_FILE)
    others = load_json(OTHERS_FILE)

    # Combinar
    combined = combine_data(train, others)

    # Filtrar
    simplified = simplify(combined)

    # Guardar resultado
    save_json(simplified, OUTPUT_FILE)

    print(f"Datos combinados guardados en: {OUTPUT_FILE}")
