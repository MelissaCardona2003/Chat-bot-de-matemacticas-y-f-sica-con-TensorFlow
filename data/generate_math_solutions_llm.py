"""
generate_math_solutions_llm.py — Genera soluciones paso a paso con un LLM
para problemas del dataset MATH de Hendrycks et al.

Este script es OFFLINE: solo se usa para crear/actualizar datos de entrenamiento.
NUNCA se llama desde la inferencia ni desde la demo.

Pipeline:
  1. Lee problemas de math_consolidated.json (MATH dataset).
  2. Filtra por dificultad y tema razonable.
  3. Limpia LaTeX del enunciado con preprocess_latex.clean_latex().
  4. Para cada problema, construye un prompt y llama a un LLM.
  5. Guarda las soluciones en data/math_llm_solved.json.

Para usar con una API externa (OpenAI, Anthropic, Ollama, etc.):
  - Implementa la función `call_llm(prompt)` con tu API.
  - Ejecuta: python data/generate_math_solutions_llm.py

Sin API configurada, el script genera un TEMPLATE con soluciones placeholder
que puedes completar manualmente o con batch processing.

Uso:
    python data/generate_math_solutions_llm.py [--api openai|ollama|none]
"""

import json
import re
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Callable

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Agregar parent al path para imports
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from data.preprocess_latex import clean_latex
from data.schema import normalize_solution, validate_entry


# ── Configuración de filtrado ──────────────────────────────────────

ALLOWED_TOPICS = [
    "algebra", "prealgebra", "counting_and_probability",
    "number_theory", "geometry",
]
MAX_LEVEL = 3          # Niveles 1-3 (evitar olímpicas nivel 4-5)
MAX_PROBLEM_CHARS = 500  # Enunciados no demasiado largos


# ── Prompt template ────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert math tutor for high school students.
Solve problems step by step, in plain English, using short numbered steps.
At the end, output exactly one line starting with "Answer:" followed by the final result.
Do not use LaTeX or special formatting.
Keep explanations clear and concise."""

USER_PROMPT_TEMPLATE = """Solve the following math problem step by step.
Use the format:
Step 1: ...
Step 2: ...
Answer: <final result>

Problem: {problem}"""


def load_math_problems() -> List[Dict]:
    """
    Carga problemas del MATH dataset y filtra por dificultad/tema.

    Returns:
        Lista de problemas filtrados con campos problem, topic, level.
    """
    # Intentar math_consolidated.json primero
    consolidated = DATA_DIR / "math_consolidated.json"
    if not consolidated.exists():
        # Fallback a math_filtered.json
        filtered = DATA_DIR / "math_filtered.json"
        if filtered.exists():
            with open(filtered, "r", encoding="utf-8") as f:
                return json.load(f)
        print("⚠ No se encontró math_consolidated.json ni math_filtered.json")
        return []

    with open(consolidated, "r", encoding="utf-8") as f:
        all_problems = json.load(f)

    print(f"  Problemas totales en MATH: {len(all_problems)}")

    # Filtrar
    filtered = []
    for p in all_problems:
        topic = p.get("topic", "").lower().replace(" ", "_")
        level = p.get("level", 99)

        if topic not in ALLOWED_TOPICS:
            continue
        if level > MAX_LEVEL:
            continue
        if len(p.get("problem", "")) > MAX_PROBLEM_CHARS:
            continue
        if not p.get("problem", "").strip():
            continue

        # Limpiar LaTeX del enunciado
        clean_problem = clean_latex(p["problem"])
        if len(clean_problem) < 10:
            continue

        filtered.append({
            "problem": clean_problem,
            "topic": topic,
            "level": level,
            "original_solution": p.get("solution", ""),
        })

    print(f"  Problemas tras filtrado (nivel<={MAX_LEVEL}, temas={ALLOWED_TOPICS}): "
          f"{len(filtered)}")

    return filtered


def build_prompt(problem: str) -> str:
    """Construye el prompt para el LLM."""
    return USER_PROMPT_TEMPLATE.format(problem=problem)


# ── Backends de LLM ───────────────────────────────────────────────

def call_llm_none(prompt: str) -> str:
    """
    Backend 'none': no llama a ningún LLM.
    Intenta convertir la solución original del MATH dataset.
    """
    return ""  # Will be handled by fallback


def call_llm_openai(prompt: str) -> str:
    """
    Backend OpenAI: usa la API de OpenAI (GPT-4o-mini o similar).
    Requiere: pip install openai, OPENAI_API_KEY en env.
    """
    import openai
    import os

    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def call_llm_ollama(prompt: str) -> str:
    """
    Backend Ollama: usa un modelo local vía Ollama.
    Requiere: Ollama instalado y modelo descargado (ej: llama3.2).
    """
    import requests

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 512},
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


LLM_BACKENDS = {
    "none": call_llm_none,
    "openai": call_llm_openai,
    "ollama": call_llm_ollama,
}


def fallback_from_original(original_solution: str) -> str:
    """
    Intenta crear una solución Step/Answer a partir de la solución
    original del MATH dataset (que está en LaTeX).

    Args:
        original_solution: Solución original con LaTeX.

    Returns:
        Solución normalizada, o cadena vacía si no se puede extraer.
    """
    cleaned = clean_latex(original_solution)
    if not cleaned:
        return ""

    # Intentar extraer respuesta de \\boxed{...}
    boxed_match = re.search(r"boxed\{([^}]+)\}", original_solution)
    answer = ""
    if boxed_match:
        answer = clean_latex(boxed_match.group(1))

    # Dividir en líneas no vacías
    lines = [l.strip() for l in cleaned.split("\n") if l.strip()]
    if not lines:
        return ""

    # Construir pasos
    steps = []
    for i, line in enumerate(lines, 1):
        steps.append(f"Step {i}: {line}")

    if answer:
        return "\n".join(steps) + f"\nAnswer: {answer}"
    else:
        return "\n".join(steps)


def generate_solutions(
    problems: List[Dict],
    llm_fn: Callable[[str], str],
    max_problems: Optional[int] = None,
    delay: float = 0.5,
) -> List[Dict]:
    """
    Genera soluciones para una lista de problemas.

    Args:
        problems: Lista de problemas filtrados.
        llm_fn: Función que llama al LLM (prompt → respuesta).
        max_problems: Limitar número de problemas (None = todos).
        delay: Segundos entre llamadas al LLM.

    Returns:
        Lista de dicts con esquema unificado.
    """
    if max_problems:
        problems = problems[:max_problems]

    results = []
    errors = 0

    for i, prob in enumerate(problems):
        if (i + 1) % 100 == 0:
            print(f"  Procesando {i+1}/{len(problems)}...")

        prompt = build_prompt(prob["problem"])

        try:
            solution = llm_fn(prompt)
        except Exception as e:
            print(f"  ⚠ Error en problema {i+1}: {e}")
            solution = ""
            errors += 1

        # Si el LLM no devolvió nada, usar fallback
        if not solution:
            solution = fallback_from_original(prob.get("original_solution", ""))

        if not solution:
            continue

        # Normalizar formato
        solution = normalize_solution(solution)

        entry = {
            "problem": prob["problem"],
            "solution": solution,
            "domain": "math",
            "source": "MATH_LLM",
            "topic": prob.get("topic", "algebra"),
            "split": "train",
        }

        # Validar
        ok, _ = validate_entry(entry)
        if ok:
            results.append(entry)

        if delay > 0 and llm_fn != call_llm_none:
            time.sleep(delay)

    print(f"  Generados: {len(results)} / {len(problems)} "
          f"(errores: {errors})")

    return results


def main():
    parser = argparse.ArgumentParser(description="Genera soluciones MATH con LLM")
    parser.add_argument("--api", choices=list(LLM_BACKENDS.keys()),
                        default="none",
                        help="Backend de LLM a usar (default: none = fallback)")
    parser.add_argument("--max", type=int, default=None,
                        help="Máximo de problemas a procesar")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay entre llamadas al LLM (segundos)")
    args = parser.parse_args()

    print("=" * 60)
    print("  GENERACIÓN DE SOLUCIONES MATH CON LLM")
    print(f"  Backend: {args.api}")
    print("=" * 60)

    # Cargar problemas
    print("\nCargando problemas MATH...")
    problems = load_math_problems()

    if not problems:
        print("No hay problemas para procesar.")
        return

    # Seleccionar backend
    llm_fn = LLM_BACKENDS[args.api]

    # Generar
    print(f"\nGenerando soluciones ({args.api})...")
    solved = generate_solutions(
        problems, llm_fn,
        max_problems=args.max,
        delay=args.delay,
    )

    if not solved:
        print("No se generaron soluciones.")
        return

    # Guardar
    output_path = DATA_DIR / "math_llm_solved.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(solved, f, ensure_ascii=False, indent=2)
    print(f"\nGuardado: {output_path} ({len(solved)} problemas)")

    # Estadísticas
    topics = {}
    for p in solved:
        t = p.get("topic", "unknown")
        topics[t] = topics.get(t, 0) + 1
    print("\nDistribución por tema:")
    for t, c in sorted(topics.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")

    print(f"\n{'=' * 60}")
    print("  GENERACIÓN COMPLETADA")
    print(f"  ⚠ Si usaste --api none, las soluciones son fallback del")
    print(f"    dataset MATH original (pueden necesitar revisión).")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
