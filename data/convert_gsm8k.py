"""
convert_gsm8k.py — Descarga GSM8K y lo convierte al esquema unificado.

GSM8K (Grade School Math 8K) contiene ~8.8K problemas de mates
con soluciones paso a paso y respuesta numérica final.

Descarga desde HuggingFace Datasets y produce:
  - data/gsm8k_train_clean.json
  - data/gsm8k_test_clean.json

Uso:
    python data/convert_gsm8k.py
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def download_gsm8k() -> Dict[str, List[Dict]]:
    """
    Descarga GSM8K usando la librería `datasets` de HuggingFace.

    Returns:
        Dict con claves "train" y "test", valores = listas de dicts
        con campos "question" y "answer" del GSM8K original.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Instalando 'datasets'...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
        from datasets import load_dataset

    print("Descargando GSM8K desde HuggingFace...")
    ds = load_dataset("openai/gsm8k", "main")

    result = {}
    for split in ["train", "test"]:
        result[split] = [
            {"question": row["question"], "answer": row["answer"]}
            for row in ds[split]
        ]
        print(f"  {split}: {len(result[split])} problemas")

    return result


def normalize_gsm8k_solution(raw_answer: str) -> str:
    """
    Convierte una solución GSM8K al formato Step/Answer unificado.

    GSM8K ya trae soluciones paso a paso con formato:
      "Línea de razonamiento 1.\nLínea 2.\n#### RESPUESTA_NUMÉRICA"

    Transformamos a:
      "Step 1: Línea 1.\nStep 2: Línea 2.\nAnswer: RESPUESTA_NUMÉRICA"

    Args:
        raw_answer: Campo "answer" crudo de GSM8K.

    Returns:
        Solución normalizada con formato Step/Answer.
    """
    # GSM8K usa "#### NÚMERO" como separador de respuesta final
    parts = raw_answer.split("####")

    if len(parts) == 2:
        reasoning = parts[0].strip()
        final_answer = parts[1].strip()
        # Limpiar comas de formato numérico (1,200 → 1200)
        final_answer = final_answer.replace(",", "")
    else:
        # Fallback: toda la cadena es la solución
        reasoning = raw_answer.strip()
        final_answer = ""
        # Intentar extraer número de la última línea
        lines = reasoning.split("\n")
        for line in reversed(lines):
            num_match = re.search(r"=\s*([\d.,]+)\s*$", line)
            if num_match:
                final_answer = num_match.group(1).replace(",", "")
                break

    # Dividir razonamiento en pasos
    lines = [l.strip() for l in reasoning.split("\n") if l.strip()]

    # Numerar cada paso
    steps = []
    for i, line in enumerate(lines, 1):
        # Quitar prefijos existentes tipo "Step 1:" o "1."
        clean = re.sub(r"^(?:Step\s+)?\d+[.):]\s*", "", line).strip()
        if clean:
            steps.append(f"Step {i}: {clean}")

    # Construir solución final
    if steps and final_answer:
        return "\n".join(steps) + f"\nAnswer: {final_answer}"
    elif steps:
        return "\n".join(steps)
    else:
        return f"Answer: {final_answer}" if final_answer else raw_answer


def detect_topic(problem: str) -> str:
    """
    Intenta detectar el tema de un problema GSM8K por palabras clave.

    Args:
        problem: Texto del problema.

    Returns:
        Tema estimado (str).
    """
    text = problem.lower()

    # Porcentajes
    if any(w in text for w in ["percent", "%", "discount", "tax", "tip", "interest"]):
        return "percentages"

    # Fracciones
    if any(w in text for w in ["half", "third", "quarter", "fraction",
                                "1/2", "1/3", "1/4", "3/4"]):
        return "fractions"

    # Geometría
    if any(w in text for w in ["area", "perimeter", "rectangle", "square",
                                "circle", "triangle", "radius", "diameter",
                                "length", "width", "height"]):
        return "geometry"

    # Probabilidad
    if any(w in text for w in ["probability", "chance", "likely", "dice", "coin"]):
        return "probability"

    # Operaciones aritméticas generales con contexto word-problem
    return "word_problems"


def convert_split(raw_data: List[Dict], split: str) -> List[Dict]:
    """
    Convierte un split de GSM8K al esquema unificado.

    Args:
        raw_data: Lista de dicts con "question" y "answer".
        split: "train" o "test".

    Returns:
        Lista de dicts con el esquema unificado.
    """
    converted = []
    skipped = 0

    for entry in raw_data:
        problem = entry["question"].strip()
        solution = normalize_gsm8k_solution(entry["answer"])

        # Filtrar entradas vacías
        if not problem or not solution:
            skipped += 1
            continue

        converted.append({
            "problem": problem,
            "solution": solution,
            "domain": "math",
            "source": "GSM8K",
            "topic": detect_topic(problem),
            "split": split,
        })

    if skipped > 0:
        print(f"  Omitidos {skipped} problemas con campos vacíos")

    return converted


def main():
    """Pipeline completo: descarga GSM8K → esquema unificado."""
    print("=" * 60)
    print("  CONVERSIÓN GSM8K → ESQUEMA UNIFICADO")
    print("=" * 60)

    # Descargar
    raw = download_gsm8k()

    # Convertir cada split
    for split in ["train", "test"]:
        data = raw[split]
        print(f"\nConvirtiendo {split}...")
        converted = convert_split(data, split)

        # Guardar
        output_path = DATA_DIR / f"gsm8k_{split}_clean.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)

        print(f"  Guardado: {output_path} ({len(converted)} problemas)")

        # Ejemplo
        if converted:
            ex = converted[0]
            print(f"\n  Ejemplo (primero del {split}):")
            print(f"    problem: {ex['problem'][:100]}...")
            print(f"    solution: {ex['solution'][:120]}...")
            print(f"    domain: {ex['domain']}, topic: {ex['topic']}")

    # Validar con schema
    try:
        from transformer_math_physics_tutor.data.schema import validate_dataset
        all_data = []
        for split in ["train", "test"]:
            path = DATA_DIR / f"gsm8k_{split}_clean.json"
            with open(path, "r") as f:
                all_data.extend(json.load(f))
        stats = validate_dataset(all_data)
        print(f"\n  Validación del esquema: {stats['pct_valid']}% válidos "
              f"({stats['valid']}/{stats['total']})")
        if stats["errors"]:
            for err, cnt in stats["errors"].items():
                print(f"    ⚠ {err}: {cnt}")
    except Exception as e:
        print(f"\n  (Validación omitida: {e})")

    print(f"\n{'=' * 60}")
    print("  CONVERSIÓN GSM8K COMPLETADA")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
