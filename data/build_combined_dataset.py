"""
build_combined_dataset.py — Construye el dataset final combinado Math + Physics.

Lee los datasets parciales ya convertidos y genera:
  - data/combined_math_physics.json (todo junto, con campo split)

Este es el archivo que usa run_training.py para entrenar el modelo.

Uso:
    python data/build_combined_dataset.py
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
        print(f"  ✓ {path.name}: {len(data)} problemas")
        return data
    print(f"  ✗ {path.name}: no encontrado")
    return []


def quality_filter(data: List[Dict],
                   min_problem_len: int = 20,
                   min_solution_len: int = 30,
                   max_problem_len: int = 500,
                   max_solution_len: int = 1000) -> List[Dict]:
    """
    Filtro final de calidad sobre el dataset combinado.

    Elimina:
      - Problemas o soluciones vacías / muy cortas.
      - Entradas excesivamente largas.
      - Duplicados exactos de problem.
    """
    seen = set()
    filtered = []
    reasons = Counter()

    for d in data:
        p = d.get("problem", "").strip()
        s = d.get("solution", "").strip()

        if len(p) < min_problem_len:
            reasons["problem_too_short"] += 1
            continue
        if len(s) < min_solution_len:
            reasons["solution_too_short"] += 1
            continue
        if len(p) > max_problem_len:
            reasons["problem_too_long"] += 1
            continue
        if len(s) > max_solution_len:
            reasons["solution_too_long"] += 1
            continue

        # Deduplicar por hash del problema
        h = hash(p.lower().strip())
        if h in seen:
            reasons["duplicate"] += 1
            continue
        seen.add(h)

        filtered.append(d)

    if reasons:
        print(f"\n  Filtrado de calidad:")
        for r, c in reasons.most_common():
            print(f"    {r}: {c} eliminados")

    return filtered


def main():
    print("=" * 60)
    print("  CONSTRUCCIÓN DEL DATASET FINAL COMBINADO")
    print("  Math + Physics → combined_math_physics.json")
    print("=" * 60)

    all_data: List[Dict] = []

    # ── 1. Cargar datasets parciales ──────────────────────

    print("\n[Cargando datasets parciales...]")

    # Matemáticas
    math_combined = load_if_exists(DATA_DIR / "math_combined.json")
    all_data.extend(math_combined)

    # Física
    physics_combined = load_if_exists(DATA_DIR / "physics_combined.json")
    all_data.extend(physics_combined)

    # Fallback: si no hay datos combinados, intentar archivos individuales
    if not math_combined:
        print("\n  Intentando archivos individuales de mates...")
        gsm8k_train = load_if_exists(DATA_DIR / "gsm8k_train_clean.json")
        gsm8k_test = load_if_exists(DATA_DIR / "gsm8k_test_clean.json")
        math_llm = load_if_exists(DATA_DIR / "math_llm_solved.json")

        for d in gsm8k_train:
            d.setdefault("split", "train")
        for d in gsm8k_test:
            d.setdefault("split", "test")
        for d in math_llm:
            d.setdefault("split", "train")

        all_data.extend(gsm8k_train)
        all_data.extend(gsm8k_test)
        all_data.extend(math_llm)

    if not physics_combined:
        print("\n  Intentando archivos individuales de física...")
        camel = load_if_exists(DATA_DIR / "camel_physics_clean.json")
        manual = load_if_exists(DATA_DIR / "physics_problems.json")

        for d in camel:
            d.setdefault("split", "train")
            d.setdefault("domain", "physics")
        for d in manual:
            d.setdefault("split", "train")
            d.setdefault("domain", "physics")

        all_data.extend(camel)
        all_data.extend(manual)

    if not all_data:
        print("\n⚠ No hay datos disponibles.")
        print("  Ejecuta en orden:")
        print("    python data/convert_gsm8k.py")
        print("    python data/convert_math_combined.py")
        print("    python data/convert_camel_physics.py")
        print("    python data/physics_combined.py")
        return

    # ── 2. Asegurar campos obligatorios ──────────────────

    for d in all_data:
        d.setdefault("domain", "math")
        d.setdefault("source", "unknown")
        d.setdefault("topic", "general")
        d.setdefault("split", "train")

    # ── 3. Filtro de calidad ─────────────────────────────

    print(f"\n  Total antes de filtrado: {len(all_data)}")
    all_data = quality_filter(all_data)
    print(f"  Total después de filtrado: {len(all_data)}")

    # ── 4. Shuffle determinístico ─────────────────────────

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(all_data)

    # ── 5. Estadísticas finales ───────────────────────────

    domain_counts = Counter(d["domain"] for d in all_data)
    split_counts = Counter(d["split"] for d in all_data)
    source_counts = Counter(d["source"] for d in all_data)
    topic_counts = Counter(d["topic"] for d in all_data)

    # Longitudes
    p_lens = [len(d["problem"]) for d in all_data]
    s_lens = [len(d["solution"]) for d in all_data]

    print(f"\n{'=' * 60}")
    print(f"  DATASET FINAL COMBINADO")
    print(f"{'=' * 60}")
    print(f"  Total: {len(all_data)} problemas")

    print(f"\n  Por dominio:")
    for d, c in sorted(domain_counts.items()):
        pct = c / len(all_data) * 100
        print(f"    {d}: {c} ({pct:.1f}%)")

    print(f"\n  Por split:")
    for s, c in sorted(split_counts.items()):
        print(f"    {s}: {c}")

    print(f"\n  Por fuente:")
    for s, c in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"    {s}: {c}")

    print(f"\n  Top temas:")
    for t, c in topic_counts.most_common(15):
        print(f"    {t}: {c}")

    print(f"\n  Longitudes (caracteres):")
    print(f"    Problema: media={sum(p_lens)/len(p_lens):.0f}, "
          f"min={min(p_lens)}, max={max(p_lens)}")
    print(f"    Solución: media={sum(s_lens)/len(s_lens):.0f}, "
          f"min={min(s_lens)}, max={max(s_lens)}")

    # ── 6. Guardar ────────────────────────────────────────

    output_path = DATA_DIR / "combined_math_physics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Guardado: {output_path}")

    # ── 7. Ejemplo de cada dominio ────────────────────────

    for domain in ["math", "physics"]:
        examples = [d for d in all_data if d["domain"] == domain]
        if examples:
            ex = examples[0]
            print(f"\n  Ejemplo ({domain}):")
            print(f"    P: {ex['problem'][:100]}...")
            print(f"    S: {ex['solution'][:120]}...")

    print(f"\n{'=' * 60}")
    print("  DATASET COMBINADO LISTO PARA ENTRENAR")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
