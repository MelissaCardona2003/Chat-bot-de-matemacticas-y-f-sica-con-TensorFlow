"""
physics_combined.py — Combina todos los datasets de física en uno solo.

Fuentes:
  1. camel_physics_clean.json (camel-ai/physics, convertido)
  2. physics_problems.json (problemas manuales, 20 problemas)

Produce: data/physics_combined.json con splits train/val/test.

Uso:
    python data/physics_combined.py
"""

import json
import random
import re
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


def normalize_manual_physics(problems: List[Dict]) -> List[Dict]:
    """
    Convierte los 20 problemas manuales de physics_problems.json
    al esquema unificado con Steps + Answer.

    Los originales tienen soluciones cortas tipo:
      "F = m * a = 5 * 3 = 15 N"
    Necesitamos:
      "Step 1: Use Newton's second law: F = m * a.
       Step 2: Substitute m = 5 kg and a = 3 m/s^2: F = 5 * 3 = 15 N.
       Answer: 15 N"
    """
    converted = []

    for p in problems:
        problem = p.get("problem", "").strip()
        solution = p.get("solution", "").strip()
        topic = p.get("topic", "general_physics")

        if not problem or not solution:
            continue

        # Si ya tiene formato Step/Answer, mantener
        if "Step" in solution and "Answer:" in solution:
            pass
        else:
            # Parsear la solución corta
            lines = [l.strip() for l in solution.split(",") if l.strip()]
            if len(lines) == 1:
                lines = [l.strip() for l in solution.split(";") if l.strip()]

            # Extraer número final
            num_match = re.search(r"=\s*([-\d.,]+\s*\w*/?(?:s\^?\d*)?)\s*$", solution)
            answer = num_match.group(1).strip() if num_match else solution.split("=")[-1].strip()

            # Construir pasos
            steps = []
            parts = solution.split("=")
            if len(parts) >= 2:
                steps.append(f"Step 1: Identify the formula: {parts[0].strip()} = ...")
                for i, part in enumerate(parts[1:-1], 2):
                    steps.append(f"Step {i}: Calculate: {part.strip()}")
                steps.append(f"Step {len(parts)}: Get the result: {parts[-1].strip()}")
            else:
                steps.append(f"Step 1: {solution}")

            solution = "\n".join(steps) + f"\nAnswer: {answer}"

        converted.append({
            "problem": problem,
            "solution": solution,
            "domain": "physics",
            "source": "PhysicsManual",
            "topic": topic,
            "split": "train",
        })

    return converted


def main():
    print("=" * 60)
    print("  COMBINAR DATASETS DE FÍSICA")
    print("=" * 60)

    all_data: List[Dict] = []

    # 1. camel-ai/physics
    print("\n[1] Cargando camel-ai/physics...")
    camel = load_if_exists(DATA_DIR / "camel_physics_clean.json")
    all_data.extend(camel)

    # 2. Problemas manuales
    print("\n[2] Cargando problemas manuales de física...")
    manual = load_if_exists(DATA_DIR / "physics_problems.json")
    if manual:
        converted_manual = normalize_manual_physics(manual)
        print(f"  Convertidos al esquema unificado: {len(converted_manual)}")
        all_data.extend(converted_manual)

    if not all_data:
        print("\n⚠ No hay datos de física disponibles.")
        print("  Ejecuta primero: python data/convert_camel_physics.py")
        return

    # 3. Asignar splits
    # Datos sin split → 85% train, 10% val, 5% test
    rng = random.Random(RANDOM_SEED)
    needs_split = [d for d in all_data if not d.get("split")]
    has_split = [d for d in all_data if d.get("split")]

    if needs_split:
        rng.shuffle(needs_split)
        n = len(needs_split)
        n_test = max(1, int(n * 0.05))
        n_val = max(1, int(n * 0.10))

        for d in needs_split[:n_test]:
            d["split"] = "test"
        for d in needs_split[n_test:n_test + n_val]:
            d["split"] = "val"
        for d in needs_split[n_test + n_val:]:
            d["split"] = "train"

    # Re-asignar los que ya tenían split=train → incluir val/test
    train_only = [d for d in has_split if d.get("split") == "train"]
    if train_only and not any(d["split"] == "val" for d in all_data):
        rng.shuffle(train_only)
        n_val = max(1, int(len(train_only) * 0.10))
        n_test = max(1, int(len(train_only) * 0.05))
        for d in train_only[:n_test]:
            d["split"] = "test"
        for d in train_only[n_test:n_test + n_val]:
            d["split"] = "val"

    all_data = has_split + needs_split

    # 4. Estadísticas
    split_counts = Counter(d.get("split", "unknown") for d in all_data)
    source_counts = Counter(d.get("source", "unknown") for d in all_data)
    topic_counts = Counter(d.get("topic", "unknown") for d in all_data)

    print(f"\n{'=' * 60}")
    print(f"  DATASET COMBINADO DE FÍSICA")
    print(f"{'=' * 60}")
    print(f"  Total: {len(all_data)} problemas")
    print(f"\n  Por split:")
    for s, c in sorted(split_counts.items()):
        print(f"    {s}: {c}")
    print(f"\n  Por fuente:")
    for s, c in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"    {s}: {c}")
    print(f"\n  Por tema:")
    for t, c in sorted(topic_counts.items(), key=lambda x: -x[1]):
        print(f"    {t}: {c}")

    # Guardar
    output_path = DATA_DIR / "physics_combined.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Guardado: {output_path}")

    print(f"\n{'=' * 60}")
    print("  COMBINACIÓN COMPLETADA")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
