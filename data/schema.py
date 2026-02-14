"""
schema.py — Esquema unificado de dataset para problemas de matemáticas y física.

Define la estructura JSON estándar que TODOS los problemas deben seguir,
independientemente de su fuente (GSM8K, MATH, PhysQA, camel-ai, etc.).

═══════════════════════════════════════════════════════════════════════
ESQUEMA UNIFICADO DE PROBLEMA
═══════════════════════════════════════════════════════════════════════

Cada entrada del dataset es un diccionario JSON con estos campos:

  {
    "problem":  str   — OBLIGATORIO. Enunciado del problema en inglés, texto
                        plano (sin LaTeX). Debe ser autocontenido.

    "solution": str   — OBLIGATORIO. Solución paso a paso en texto plano.
                        FORMATO REQUERIDO:
                          Step 1: <explicación del primer paso>
                          Step 2: <explicación del segundo paso>
                          ...
                          Answer: <resultado final, con unidades si aplica>
                        La última línea SIEMPRE debe empezar con "Answer: ".

    "domain":   str   — OBLIGATORIO. Dominio del problema.
                        Valores permitidos: "math", "physics".

    "source":   str   — OPCIONAL. Origen del dato.
                        Ejemplos: "GSM8K", "MATH_LLM", "CamelPhysics",
                                  "PhysicsManual", "custom".

    "topic":    str   — OPCIONAL. Tema específico dentro del dominio.
                        Math:    "arithmetic", "algebra", "geometry",
                                 "probability", "number_theory",
                                 "word_problems", "fractions", "percentages".
                        Physics: "kinematics", "dynamics", "energy",
                                 "electricity", "thermodynamics", "optics",
                                 "waves", "fluids", "general_physics".

    "split":    str   — OPCIONAL. Partición de datos.
                        Valores: "train", "val", "test".
                        Si no se pone, se asigna durante la combinación.
  }

═══════════════════════════════════════════════════════════════════════
FORMATO OBLIGATORIO DE SOLUTION
═══════════════════════════════════════════════════════════════════════

Todas las soluciones deben seguir este patrón:

  Step 1: <razonamiento>
  Step 2: <razonamiento>
  ...
  Answer: <resultado final>

Reglas:
  1. Mínimo 1 paso (Step 1) + línea Answer.
  2. Los pasos van en orden numérico (Step 1, Step 2, ...).
  3. Cada paso debe ser una oración clara y breve.
  4. La línea Answer: debe ser la ÚLTIMA línea no vacía.
  5. Para física, incluir UNIDADES en la respuesta (e.g., "Answer: 120 km").
  6. Para matemáticas, dar el valor numérico o expresión (e.g., "Answer: 42").
  7. NO usar LaTeX en ningún campo.

═══════════════════════════════════════════════════════════════════════
EJEMPLO MATH
═══════════════════════════════════════════════════════════════════════

  {
    "problem": "Betty has $50. She buys a book for $12 and a pen for $3. How much money does she have left?",
    "solution": "Step 1: Betty starts with $50.\\nStep 2: She spends $12 + $3 = $15 in total.\\nStep 3: She has $50 - $15 = $35 left.\\nAnswer: 35",
    "domain": "math",
    "source": "GSM8K",
    "topic": "arithmetic",
    "split": "train"
  }

═══════════════════════════════════════════════════════════════════════
EJEMPLO PHYSICS
═══════════════════════════════════════════════════════════════════════

  {
    "problem": "A car accelerates from rest at 3 m/s^2 for 5 seconds. What is its final velocity?",
    "solution": "Step 1: The car starts from rest, so initial velocity v0 = 0 m/s.\\nStep 2: Using v = v0 + a*t, we get v = 0 + 3*5 = 15 m/s.\\nAnswer: 15 m/s",
    "domain": "physics",
    "source": "PhysicsManual",
    "topic": "kinematics",
    "split": "train"
  }

═══════════════════════════════════════════════════════════════════════
"""

import re
from typing import Dict, List, Optional, Tuple

# ── Constantes del esquema ──────────────────────────────────────────

REQUIRED_FIELDS = ("problem", "solution", "domain")
OPTIONAL_FIELDS = ("source", "topic", "split")

VALID_DOMAINS = ("math", "physics")
VALID_SPLITS = ("train", "val", "test")

MATH_TOPICS = (
    "arithmetic", "algebra", "geometry", "probability",
    "number_theory", "word_problems", "fractions", "percentages",
    "grade_school_math", "prealgebra", "counting_and_probability",
    "intermediate_algebra", "precalculus",
)

PHYSICS_TOPICS = (
    "kinematics", "dynamics", "energy", "electricity",
    "thermodynamics", "optics", "waves", "fluids",
    "general_physics", "magnetism", "gravitation",
)

ALL_TOPICS = MATH_TOPICS + PHYSICS_TOPICS

# Regex para validar formato de solución
ANSWER_PATTERN = re.compile(r"^Answer:\s*.+", re.MULTILINE)
STEP_PATTERN = re.compile(r"^Step\s+\d+:", re.MULTILINE)


def validate_entry(entry: Dict, strict: bool = False) -> Tuple[bool, List[str]]:
    """
    Valida que una entrada cumple el esquema unificado.

    Args:
        entry: Diccionario con los campos del problema.
        strict: Si True, exige Steps numerados. Si False, acepta
                soluciones libres siempre que tengan Answer:.

    Returns:
        (es_válido, lista_de_errores)
    """
    errors = []

    # Campos obligatorios
    for field in REQUIRED_FIELDS:
        if field not in entry or not entry[field]:
            errors.append(f"Falta campo obligatorio: '{field}'")

    if errors:
        return False, errors

    # Validar domain
    if entry["domain"] not in VALID_DOMAINS:
        errors.append(f"domain='{entry['domain']}' no válido. Usar: {VALID_DOMAINS}")

    # Validar split si existe
    if "split" in entry and entry["split"] not in VALID_SPLITS:
        errors.append(f"split='{entry['split']}' no válido. Usar: {VALID_SPLITS}")

    # Validar formato de solución
    solution = entry["solution"]
    if not ANSWER_PATTERN.search(solution):
        errors.append("La solución no contiene línea 'Answer: ...'")

    if strict and not STEP_PATTERN.search(solution):
        errors.append("La solución no contiene pasos 'Step N: ...'")

    return len(errors) == 0, errors


def normalize_solution(solution: str) -> str:
    """
    Normaliza una solución al formato estándar Step/Answer.

    Si la solución ya tiene Steps + Answer, la devuelve tal cual.
    Si tiene razonamiento libre + Answer, la reformatea.
    Si no tiene Answer:, intenta extraer la última línea numérica.

    Args:
        solution: Texto de la solución (puede estar en varios formatos).

    Returns:
        Solución normalizada con formato Step/Answer.
    """
    solution = solution.strip()

    # Si ya tiene el formato correcto, devolver
    if STEP_PATTERN.search(solution) and ANSWER_PATTERN.search(solution):
        return solution

    # Separar en líneas no vacías
    lines = [l.strip() for l in solution.split("\n") if l.strip()]

    if not lines:
        return solution

    # Verificar si la última línea ya es Answer:
    has_answer = bool(ANSWER_PATTERN.match(lines[-1]))

    if has_answer:
        # Solo faltan los Steps — numerar las líneas de razonamiento
        steps = lines[:-1]
        answer_line = lines[-1]
    else:
        # Intentar extraer respuesta numérica de la última línea
        steps = lines[:-1] if len(lines) > 1 else []
        last = lines[-1]

        # Buscar patrones comunes de respuesta final
        num_match = re.search(r"=\s*([-\d.,/]+(?:\s*\w+)?)\s*$", last)
        if num_match:
            answer_line = f"Answer: {num_match.group(1).strip()}"
            steps.append(last)
        else:
            answer_line = f"Answer: {last}"

    # Numerar los pasos si no están numerados
    numbered_steps = []
    for i, step in enumerate(steps, 1):
        if re.match(r"^Step\s+\d+:", step):
            numbered_steps.append(step)
        else:
            numbered_steps.append(f"Step {i}: {step}")

    return "\n".join(numbered_steps + [answer_line])


def extract_answer(solution: str) -> Optional[str]:
    """
    Extrae la respuesta final de una solución.

    Busca la línea que empieza con 'Answer:' y devuelve lo que hay después.

    Args:
        solution: Texto de la solución completa.

    Returns:
        La respuesta final (sin 'Answer:'), o None si no se encuentra.
    """
    match = ANSWER_PATTERN.search(solution)
    if match:
        answer = match.group(0).replace("Answer:", "").strip()
        return answer
    return None


def validate_dataset(data: List[Dict], strict: bool = False) -> Dict:
    """
    Valida un dataset completo y muestra estadísticas.

    Args:
        data: Lista de entradas del dataset.
        strict: Si True, exige Steps numerados.

    Returns:
        Diccionario con estadísticas de validación.
    """
    total = len(data)
    valid = 0
    invalid = 0
    error_counts: Dict[str, int] = {}

    for entry in data:
        ok, errs = validate_entry(entry, strict=strict)
        if ok:
            valid += 1
        else:
            invalid += 1
            for e in errs:
                error_counts[e] = error_counts.get(e, 0) + 1

    stats = {
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "pct_valid": round(valid / total * 100, 1) if total > 0 else 0,
        "errors": error_counts,
    }

    return stats


if __name__ == "__main__":
    print("=" * 60)
    print("  ESQUEMA UNIFICADO DE DATASET")
    print("=" * 60)

    # Ejemplo de validación
    good = {
        "problem": "What is 2 + 3?",
        "solution": "Step 1: Add 2 and 3.\nAnswer: 5",
        "domain": "math",
        "source": "custom",
        "topic": "arithmetic",
        "split": "train",
    }

    bad = {
        "problem": "What is 2 + 3?",
        "solution": "The answer is 5",
        "domain": "science",  # inválido
    }

    ok1, errs1 = validate_entry(good)
    print(f"\n  Ejemplo correcto: valid={ok1}, errors={errs1}")

    ok2, errs2 = validate_entry(bad)
    print(f"  Ejemplo incorrecto: valid={ok2}, errors={errs2}")

    # Ejemplo de normalización
    raw = "Betty has $50.\nShe spends $12 + $3 = $15.\nShe has $50 - $15 = $35 left.\nAnswer: 35"
    normalized = normalize_solution(raw)
    print(f"\n  Normalizado:\n{normalized}")
