"""
convert_math_combined.py — Combina GSM8K + MATH_LLM en un solo dataset de mates.

Produce: data/math_combined.json con splits train/val/test.

Estrategia de splits:
  - GSM8K train → 90% train, 10% val
  - GSM8K test  → test
  - MATH_LLM   → 90% train, 10% val (si viene sin split)

Uso:
    python data/convert_math_combined.py
"""

import json
import random
from pathlib import Path
from collections import Counter
from typing import List, Dict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

RANDOM_SEED = 42


def load_if_exists(path: Path) -> List[Dict]:
    """Carga un JSON si existe, o devuelve lista vacía."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Cargado: {path.name} ({len(data)} problemas)")
        return data
    print(f"  No encontrado: {path.name}")
    return []


def assign_splits(data: List[Dict], val_ratio: float = 0.1,
                   seed: int = RANDOM_SEED) -> List[Dict]:
    """
    Asigna splits train/val a datos que no los tengan.
    Los que ya tienen split se dejan intactos.
    """
    needs_split = [d for d in data if not d.get("split")]
    has_split = [d for d in data if d.get("split")]

    if needs_split:
        rng = random.Random(seed)
        rng.shuffle(needs_split)
        n_val = max(1, int(len(needs_split) * val_ratio))

        for d in needs_split[:n_val]:
            d["split"] = "val"
        for d in needs_split[n_val:]:
            d["split"] = "train"

    return has_split + needs_split


def main():
    print("=" * 60)
    print("  COMBINAR DATASETS DE MATEMÁTICAS")
    print("=" * 60)

    all_data: List[Dict] = []

    # 1. GSM8K
    print("\n[1] Cargando GSM8K...")
    gsm8k_train = load_if_exists(DATA_DIR / "gsm8k_train_clean.json")
    gsm8k_test = load_if_exists(DATA_DIR / "gsm8k_test_clean.json")

    # Asignar splits a GSM8K train → train/val
    if gsm8k_train:
        rng = random.Random(RANDOM_SEED)
        rng.shuffle(gsm8k_train)
        n_val = max(1, int(len(gsm8k_train) * 0.1))

        for d in gsm8k_train[:n_val]:
            d["split"] = "val"
        for d in gsm8k_train[n_val:]:
            d["split"] = "train"
        all_data.extend(gsm8k_train)

    # GSM8K test → test
    for d in gsm8k_test:
        d["split"] = "test"
    all_data.extend(gsm8k_test)

    # 2. MATH_LLM
    print("\n[2] Cargando MATH_LLM...")
    math_llm = load_if_exists(DATA_DIR / "math_llm_solved.json")
    if math_llm:
        math_llm = assign_splits(math_llm, val_ratio=0.1)
        all_data.extend(math_llm)

    # 3. Estadísticas
    if not all_data:
        print("\n⚠ No hay datos disponibles.")
        print("  Ejecuta primero: python data/convert_gsm8k.py")
        return

    split_counts = Counter(d.get("split", "unknown") for d in all_data)
    source_counts = Counter(d.get("source", "unknown") for d in all_data)
    topic_counts = Counter(d.get("topic", "unknown") for d in all_data)

    print(f"\n{'=' * 60}")
    print(f"  DATASET COMBINADO DE MATEMÁTICAS")
    print(f"{'=' * 60}")
    print(f"  Total: {len(all_data)} problemas")
    print(f"\n  Por split:")
    for s, c in sorted(split_counts.items()):
        print(f"    {s}: {c}")
    print(f"\n  Por fuente:")
    for s, c in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"    {s}: {c}")
    print(f"\n  Por tema (top 10):")
    for t, c in topic_counts.most_common(10):
        print(f"    {t}: {c}")

    # Guardar
    output_path = DATA_DIR / "math_combined.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Guardado: {output_path}")

    print(f"\n{'=' * 60}")
    print("  COMBINACIÓN COMPLETADA")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
