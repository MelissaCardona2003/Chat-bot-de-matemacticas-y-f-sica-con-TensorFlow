"""
download_gsm8k.py — Descarga y procesa el dataset GSM8K.

GSM8K (Grade School Math 8K) es un dataset de Google/OpenAI con 8,819
problemas de matemáticas de nivel escolar CON soluciones paso a paso.

Fuente: https://github.com/openai/grade-school-math

Formato original (JSONL):
    {"question": "...", "answer": "Step 1... Step 2... #### 42"}

Formato de salida (JSON):
    [{"problem": "...", "solution": "...", "topic": "grade_school_math", "level": "elementary"}]

Uso:
    python data/download_gsm8k.py
"""

import json
import os
import sys
from pathlib import Path

# Intentar importar requests, si no está disponible usar urllib
try:
    import requests
    USE_REQUESTS = True
except ImportError:
    import urllib.request
    USE_REQUESTS = False


# Directorio de datos
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR


# URLs oficiales del dataset GSM8K
URLS = {
    "train": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
    "test": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
}


def download_file(url: str) -> str:
    """
    Descarga un archivo desde una URL y retorna su contenido como string.

    Args:
        url: URL del archivo a descargar.

    Returns:
        Contenido del archivo como string.
    """
    print(f"  Descargando: {url}")

    if USE_REQUESTS:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        return response.text
    else:
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read().decode("utf-8")


def parse_jsonl(content: str) -> list:
    """
    Parsea contenido JSONL (una línea JSON por línea).

    Args:
        content: String con contenido JSONL.

    Returns:
        Lista de diccionarios parseados.
    """
    records = []
    for line_num, line in enumerate(content.strip().split("\n"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  ⚠️ Error en línea {line_num}: {e}")
    return records


def format_for_pipeline(raw_records: list) -> list:
    """
    Convierte registros GSM8K al formato del proyecto.

    GSM8K format:  {"question": "...", "answer": "..."}
    Nuestro format: {"problem": "...", "solution": "...", "topic": "...", "level": "..."}

    Args:
        raw_records: Lista de registros en formato GSM8K.

    Returns:
        Lista de registros en formato del proyecto.
    """
    formatted = []
    for record in raw_records:
        formatted.append({
            "problem": record["question"].strip(),
            "solution": record["answer"].strip(),
            "topic": "grade_school_math",
            "level": "elementary",
        })
    return formatted


def download_gsm8k():
    """
    Pipeline principal: descarga GSM8K train y test, guarda en formato JSON.
    """
    print("=" * 60)
    print("  DESCARGA DEL DATASET GSM8K")
    print("  Grade School Math 8K (OpenAI)")
    print("=" * 60)

    for split, url in URLS.items():
        print(f"\n📥 Descargando split '{split}'...")

        # Descargar
        try:
            content = download_file(url)
        except Exception as e:
            print(f"  ❌ Error al descargar {split}: {e}")
            continue

        # Parsear JSONL
        raw_records = parse_jsonl(content)
        print(f"  Registros parseados: {len(raw_records)}")

        # Formatear para nuestro pipeline
        records = format_for_pipeline(raw_records)

        # Guardar
        output_file = DATA_DIR / f"gsm8k_{split}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        print(f"  ✅ Guardado en: {output_file}")
        print(f"  Total: {len(records)} problemas")

        # Mostrar ejemplo
        if records:
            ex = records[0]
            print(f"\n  📄 Ejemplo:")
            print(f"  Problema: {ex['problem'][:120]}...")
            print(f"  Solución: {ex['solution'][:180]}...")

        # Estadísticas
        avg_prob = sum(len(r["problem"]) for r in records) / len(records)
        avg_sol = sum(len(r["solution"]) for r in records) / len(records)
        print(f"\n  📊 Longitud promedio:")
        print(f"     Problema: {avg_prob:.1f} chars")
        print(f"     Solución: {avg_sol:.1f} chars")

    print(f"\n{'=' * 60}")
    print("  ✅ DESCARGA COMPLETADA")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    download_gsm8k()
