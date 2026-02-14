"""
filter_dataset.py — Filtra el MATH dataset por dificultad, tema y longitud.

Aplica filtros al dataset MATH descargado (o al dataset local) para
quedarnos solo con problemas que el modelo Transformer pueda manejar:
- Nivel de dificultad básico (Level 1)
- Temas específicos (algebra, prealgebra, counting_and_probability)
- Longitudes razonables para encoder/decoder

Produce math_filtered.json (train) y math_filtered_test.json (test).

Uso:
    python data/filter_dataset.py
"""

import json
import os
import random
from pathlib import Path
from collections import Counter
from typing import List, Dict, Optional, Tuple


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MATH_DIR = DATA_DIR / "MATH"

# ── Criterios de filtrado ──────────────────────────────────
ALLOWED_TOPICS = [
    "algebra",
    "prealgebra",
]
MAX_LEVEL = 1
MAX_PROBLEM_WORDS = 100
MAX_SOLUTION_WORDS = 150
TEST_SPLIT = 0.1
RANDOM_SEED = 42


def load_math_dataset() -> Tuple[List[Dict], str]:
    """
    Carga el MATH dataset desde archivos consolidados o directorio.

    Intenta cargar en este orden:
    1. math_train_raw.json (generado por download_dataset.py)
    2. El directorio MATH/ directamente
    3. math_training_data.json (fallback manual)

    Returns:
        Tupla de (lista de problemas, nombre de la fuente).
    """
    # Opción 1: Archivo consolidado de HuggingFace (download via datasets lib)
    consolidated = DATA_DIR / "math_consolidated.json"
    if consolidated.exists():
        with open(consolidated, "r", encoding="utf-8") as f:
            problems = json.load(f)
        return problems, f"math_consolidated.json ({len(problems)} problemas)"

    # Opción 2: Archivos consolidados de download_dataset.py
    raw_train = DATA_DIR / "math_train_raw.json"
    raw_test = DATA_DIR / "math_test_raw.json"

    if raw_train.exists():
        problems = []
        with open(raw_train, "r", encoding="utf-8") as f:
            problems.extend(json.load(f))
        if raw_test.exists():
            with open(raw_test, "r", encoding="utf-8") as f:
                problems.extend(json.load(f))
        return problems, "MATH dataset (archivos consolidados)"

    # Opción 2: Directorio MATH/ con JSONs individuales
    if MATH_DIR.exists():
        problems = []
        for json_file in MATH_DIR.rglob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                level_str = data.get("level", "Level 1")
                try:
                    level = int(level_str.replace("Level ", ""))
                except (ValueError, AttributeError):
                    level = 0

                problems.append({
                    "problem": data.get("problem", ""),
                    "solution": data.get("solution", ""),
                    "topic": data.get("type", "unknown").lower().replace(" ", "_"),
                    "level": level,
                })
            except (json.JSONDecodeError, KeyError):
                continue

        if problems:
            return problems, f"MATH dataset (directorio, {len(problems)} archivos)"

    # Opción 3: Fallback a dataset manual
    fallback_path = DATA_DIR / "math_training_data.json"
    if fallback_path.exists():
        with open(fallback_path, "r", encoding="utf-8") as f:
            problems = json.load(f)
        # Asegurar que tienen los campos necesarios
        for p in problems:
            p.setdefault("topic", "arithmetic")
            p.setdefault("level", 1)
        return problems, f"math_training_data.json (fallback, {len(problems)} problemas)"

    return [], "ninguno"


def filter_problems(
    problems: List[Dict],
    allowed_topics: List[str] = ALLOWED_TOPICS,
    max_level: int = MAX_LEVEL,
    max_problem_words: int = MAX_PROBLEM_WORDS,
    max_solution_words: int = MAX_SOLUTION_WORDS,
) -> List[Dict]:
    """
    Aplica filtros a la lista de problemas.

    Criterios:
    1. El tema debe estar en allowed_topics
    2. El nivel debe ser <= max_level
    3. El problema no debe tener más de max_problem_words palabras
    4. La solución no debe tener más de max_solution_words palabras

    Args:
        problems: Lista de diccionarios con problem, solution, topic, level.
        allowed_topics: Lista de temas permitidos.
        max_level: Nivel máximo de dificultad (inclusive).
        max_problem_words: Máximo de palabras en el problema.
        max_solution_words: Máximo de palabras en la solución.

    Returns:
        Lista filtrada de problemas.
    """
    filtered = []
    reasons = Counter()

    for prob in problems:
        topic = prob.get("topic", "unknown").lower().replace(" ", "_")
        level = prob.get("level", 0)
        problem_text = prob.get("problem", "")
        solution_text = prob.get("solution", "")

        problem_words = len(problem_text.split())
        solution_words = len(solution_text.split())

        # Aplicar filtros
        if topic not in allowed_topics:
            reasons["tema_no_permitido"] += 1
            continue
        if level > max_level:
            reasons["nivel_alto"] += 1
            continue
        if problem_words > max_problem_words:
            reasons["problema_largo"] += 1
            continue
        if solution_words > max_solution_words:
            reasons["solucion_larga"] += 1
            continue
        if not problem_text.strip() or not solution_text.strip():
            reasons["texto_vacio"] += 1
            continue

        filtered.append({
            "problem": problem_text,
            "solution": solution_text,
            "topic": topic,
            "level": level,
        })

    # Mostrar razones de eliminación
    if reasons:
        print("\n  Razones de eliminación:")
        for reason, count in reasons.most_common():
            print(f"    {reason}: {count}")

    return filtered


def split_train_test(
    problems: List[Dict],
    test_ratio: float = TEST_SPLIT,
    seed: int = RANDOM_SEED
) -> Tuple[List[Dict], List[Dict]]:
    """
    Divide los problemas en conjuntos de entrenamiento y test.

    Args:
        problems: Lista de problemas filtrados.
        test_ratio: Proporción para el conjunto de test.
        seed: Semilla para reproducibilidad.

    Returns:
        Tupla de (train_problems, test_problems).
    """
    random.seed(seed)
    shuffled = problems.copy()
    random.shuffle(shuffled)

    n_test = max(1, int(len(shuffled) * test_ratio))
    test_data = shuffled[:n_test]
    train_data = shuffled[n_test:]

    return train_data, test_data


def print_statistics(
    original: List[Dict],
    filtered: List[Dict],
    source_name: str
) -> None:
    """
    Imprime estadísticas detalladas del filtrado.

    Args:
        original: Lista original de problemas.
        filtered: Lista filtrada de problemas.
        source_name: Nombre de la fuente de datos.
    """
    print(f"\n{'=' * 60}")
    print(f"  ESTADÍSTICAS DE FILTRADO")
    print(f"{'=' * 60}")
    print(f"  Fuente: {source_name}")
    print(f"  Original: {len(original)} problemas")
    print(f"  Filtrado: {len(filtered)} problemas")
    print(f"  Eliminados: {len(original) - len(filtered)} "
          f"({(len(original) - len(filtered)) / max(len(original), 1) * 100:.1f}%)")

    if filtered:
        # Distribución por tema
        topics = Counter(p["topic"] for p in filtered)
        print(f"\n  Distribución por tema:")
        for topic, count in topics.most_common():
            bar = "█" * (count * 30 // max(topics.values()))
            print(f"    {topic:30s}: {count:4d} {bar}")

        # Distribución por nivel
        levels = Counter(p["level"] for p in filtered)
        print(f"\n  Distribución por nivel:")
        for level, count in sorted(levels.items()):
            print(f"    Nivel {level}: {count}")

        # Estadísticas de longitud
        prob_lens = [len(p["problem"].split()) for p in filtered]
        sol_lens = [len(p["solution"].split()) for p in filtered]
        print(f"\n  Longitud del problema (palabras):")
        print(f"    Media: {sum(prob_lens)/len(prob_lens):.1f}")
        print(f"    Min: {min(prob_lens)}, Max: {max(prob_lens)}")
        print(f"\n  Longitud de la solución (palabras):")
        print(f"    Media: {sum(sol_lens)/len(sol_lens):.1f}")
        print(f"    Min: {min(sol_lens)}, Max: {max(sol_lens)}")


def main():
    """Pipeline principal de filtrado."""
    print("=" * 60)
    print("  FILTRADO DEL MATH DATASET")
    print("=" * 60)

    # Cargar datos
    print("\nCargando dataset...")
    problems, source_name = load_math_dataset()

    if not problems:
        print("\nNo se encontró ningún dataset.")
        print("Opciones:")
        print("  1. Ejecutar: python data/download_dataset.py")
        print("  2. Verificar que math_training_data.json existe en data/")
        return

    print(f"  Fuente: {source_name}")
    print(f"  Problemas cargados: {len(problems)}")

    # Filtrar
    print(f"\nAplicando filtros:")
    print(f"  Temas permitidos: {ALLOWED_TOPICS}")
    print(f"  Nivel máximo: {MAX_LEVEL}")
    print(f"  Máx palabras problema: {MAX_PROBLEM_WORDS}")
    print(f"  Máx palabras solución: {MAX_SOLUTION_WORDS}")

    filtered = filter_problems(problems)

    if not filtered:
        print("\nNo quedaron problemas tras el filtrado.")
        print("Considera relajar los criterios (aumentar MAX_LEVEL, etc.)")

        # Usar todos los problemas como fallback si son del dataset manual
        if len(problems) <= 300:
            print("\nUsando todos los problemas del dataset sin filtrar...")
            filtered = problems

    # Estadísticas
    print_statistics(problems, filtered, source_name)

    # Dividir train/test
    train_data, test_data = split_train_test(filtered)
    print(f"\n  División train/test:")
    print(f"    Train: {len(train_data)} problemas")
    print(f"    Test:  {len(test_data)} problemas")

    # Guardar
    train_path = DATA_DIR / "math_filtered.json"
    test_path = DATA_DIR / "math_filtered_test.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Guardado: {train_path}")

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"  Guardado: {test_path}")

    # Ejemplo
    if filtered:
        print(f"\n--- Ejemplo de problema filtrado ---")
        ex = filtered[0]
        print(f"  Tema: {ex['topic']} | Nivel: {ex['level']}")
        print(f"  Problema: {ex['problem'][:120]}")
        print(f"  Solución: {ex['solution'][:120]}")

    print(f"\n{'=' * 60}")
    print(f"  FILTRADO COMPLETADO")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
