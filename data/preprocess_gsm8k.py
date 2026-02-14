"""
preprocess_gsm8k.py — Limpieza y preparación del dataset GSM8K.

Toma los archivos gsm8k_train.json y gsm8k_test.json descargados,
limpia las soluciones (convierte "#### 42" → "Answer: 42"), filtra
por longitud, y genera math_clean.json y math_clean_test.json listos
para entrenamiento.

Transformaciones:
1. Reemplaza "#### N" por "Answer: N"
2. Limpia espacios/saltos de línea extras
3. Filtra ejemplos con soluciones demasiado largas (> max_solution_len)
4. Genera estadísticas de distribución

Uso:
    python data/preprocess_gsm8k.py
"""

import json
import re
import os
from pathlib import Path
from collections import Counter


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR


def clean_gsm8k_solution(solution: str) -> str:
    """
    Limpia una solución GSM8K para consumo del modelo.

    Transformaciones:
    1. "#### 42" → "Answer: 42"
    2. Líneas vacías removidas
    3. Espacios en blanco normalizados
    4. Símbolos LaTeX residuales convertidos

    Args:
        solution: Solución en formato GSM8K original.

    Returns:
        Solución limpia.
    """
    # 1. Reemplazar #### N con Answer: N
    solution = re.sub(r"####\s*(-?[\d,\.]+)", r"Answer: \1", solution)

    # 2. Limpiar notaciones de dinero con formato
    #    "$1,200" → "1200 dollars", etc. (simplificar para el modelo)
    #    Mantenemos $ como está para no perder contexto

    # 3. Normalizar operaciones con <<>>
    #    GSM8K usa <<5+3=8>> para cálculos intermedios
    solution = re.sub(r"<<[^>]*>>", "", solution)

    # 4. Limpiar líneas vacías y espacios
    lines = []
    for line in solution.split("\n"):
        line = line.strip()
        if line:
            lines.append(line)

    return "\n".join(lines)


def clean_problem(problem: str) -> str:
    """
    Limpia un problema GSM8K.

    Args:
        problem: Texto del problema.

    Returns:
        Problema limpio.
    """
    # Normalizar espacios
    problem = " ".join(problem.split())
    return problem.strip()


def compute_stats(records: list, label: str) -> dict:
    """
    Calcula y muestra estadísticas de un conjunto de datos.

    Args:
        records: Lista de diccionarios con 'problem' y 'solution'.
        label: Etiqueta para imprimir (ej: "Train", "Test").

    Returns:
        Diccionario con estadísticas.
    """
    prob_lens = [len(r["problem"]) for r in records]
    sol_lens = [len(r["solution"]) for r in records]

    stats = {
        "count": len(records),
        "avg_problem_len": sum(prob_lens) / len(prob_lens) if prob_lens else 0,
        "avg_solution_len": sum(sol_lens) / len(sol_lens) if sol_lens else 0,
        "max_problem_len": max(prob_lens) if prob_lens else 0,
        "max_solution_len": max(sol_lens) if sol_lens else 0,
        "min_problem_len": min(prob_lens) if prob_lens else 0,
        "min_solution_len": min(sol_lens) if sol_lens else 0,
    }

    print(f"\n  📊 Estadísticas [{label}]:")
    print(f"     Total registros: {stats['count']}")
    print(f"     Problema — avg: {stats['avg_problem_len']:.1f}, "
          f"min: {stats['min_problem_len']}, max: {stats['max_problem_len']} chars")
    print(f"     Solución — avg: {stats['avg_solution_len']:.1f}, "
          f"min: {stats['min_solution_len']}, max: {stats['max_solution_len']} chars")

    return stats


def preprocess_gsm8k(
    max_solution_len: int = 350,
    max_problem_len: int = 300,
    min_solution_len: int = 10,
):
    """
    Pipeline completo de preprocesamiento de GSM8K.

    Args:
        max_solution_len: Longitud máxima de solución (filtrar más largas).
        max_problem_len: Longitud máxima de problema.
        min_solution_len: Longitud mínima de solución (filtrar muy cortas).
    """
    print("=" * 60)
    print("  PREPROCESAMIENTO GSM8K")
    print("=" * 60)

    results = {}

    for split in ["train", "test"]:
        input_file = DATA_DIR / f"gsm8k_{split}.json"

        if not input_file.exists():
            print(f"\n  ⚠️ No se encontró {input_file}")
            print(f"     Ejecuta primero: python data/download_gsm8k.py")
            continue

        # Cargar datos
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"\n{'─' * 40}")
        print(f"  Split: {split} — {len(data)} registros originales")

        # Limpiar
        cleaned = []
        for record in data:
            cleaned.append({
                "problem": clean_problem(record["problem"]),
                "solution": clean_gsm8k_solution(record["solution"]),
                "topic": record.get("topic", "grade_school_math"),
                "level": record.get("level", "elementary"),
            })

        # Filtrar por longitud
        filtered = []
        filtered_out = {"too_long_sol": 0, "too_long_prob": 0, "too_short_sol": 0}

        for record in cleaned:
            sol_len = len(record["solution"])
            prob_len = len(record["problem"])

            if sol_len > max_solution_len:
                filtered_out["too_long_sol"] += 1
                continue
            if prob_len > max_problem_len:
                filtered_out["too_long_prob"] += 1
                continue
            if sol_len < min_solution_len:
                filtered_out["too_short_sol"] += 1
                continue

            filtered.append(record)

        print(f"  Después de filtrar: {len(filtered)} registros")
        if any(v > 0 for v in filtered_out.values()):
            print(f"  Filtrados: {filtered_out}")

        # Estadísticas
        stats = compute_stats(filtered, split.upper())

        # Guardar
        if split == "train":
            output_file = DATA_DIR / "math_clean.json"
        else:
            output_file = DATA_DIR / "math_clean_test.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(filtered, f, indent=2, ensure_ascii=False)

        print(f"  ✅ Guardado en: {output_file}")

        results[split] = {"count": len(filtered), "stats": stats}

        # Mostrar ejemplos
        print(f"\n  📄 Ejemplo limpio:")
        if filtered:
            ex = filtered[0]
            print(f"     Problema: {ex['problem'][:150]}")
            print(f"     Solución: {ex['solution'][:200]}")

    # Resumen comparativo
    print(f"\n{'=' * 60}")
    print("  RESUMEN COMPARATIVO")
    print(f"{'=' * 60}")
    print(f"\n  {'Métrica':<30} {'Antes (MATH)':<15} {'Ahora (GSM8K)':<15}")
    print(f"  {'─' * 60}")
    print(f"  {'Ejemplos train':<30} {'1,581':<15} {results.get('train', {}).get('count', '?'):,}")
    print(f"  {'Ejemplos test':<30} {'175':<15} {results.get('test', {}).get('count', '?'):,}")

    if "train" in results:
        avg_sol = results["train"]["stats"]["avg_solution_len"]
        print(f"  {'Avg solución (chars)':<30} {'2.6':<15} {avg_sol:.1f}")
        print(f"  {'Tiene pasos':<30} {'❌ No':<15} {'✅ Sí':<15}")

    print(f"\n  ✅ PREPROCESAMIENTO COMPLETADO")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    preprocess_gsm8k()
