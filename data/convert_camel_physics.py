"""
convert_camel_physics.py — Descarga camel-ai/physics y lo convierte al esquema unificado.

El dataset camel-ai/physics contiene ~20K problemas de física con soluciones
detalladas generadas por un LLM, organizados por subtema.

Descarga desde HuggingFace y produce:
  - data/camel_physics_clean.json

Los problemas se filtran por longitud y se convierten al formato Step/Answer.

Uso:
    python data/convert_camel_physics.py [--max 5000]
"""

import json
import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def download_camel_physics(max_raw: int = 10000) -> List[Dict]:
    """
    Descarga el dataset camel-ai/physics desde HuggingFace (streaming).

    Usa streaming=True para evitar descargar/procesar las 20K entradas
    completas, limitando a max_raw entradas.

    Args:
        max_raw: Máximo de entradas crudas a descargar.

    Returns:
        Lista de dicts con campos del dataset original.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Instalando 'datasets'...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
        from datasets import load_dataset

    print("Descargando camel-ai/physics desde HuggingFace (streaming)...")
    ds = load_dataset("camel-ai/physics", split="train", streaming=True)

    result = []
    for i, row in enumerate(ds):
        entry = dict(row)
        result.append(entry)
        if (i + 1) % 2000 == 0:
            print(f"  Descargados {i+1} entradas...")
        if len(result) >= max_raw:
            break

    print(f"  Descargados: {len(result)} entradas")
    return result


def extract_qa_from_camel(entry: Dict) -> Optional[Dict]:
    """
    Extrae pregunta y respuesta de una entrada camel-ai.

    El formato de camel-ai puede variar; típicamente tiene
    pares de mensajes de un sistema role-play.

    Returns:
        Dict con "problem" y "raw_solution", o None si no se puede extraer.
    """
    # Intentar varias estructuras conocidas de camel-ai
    problem = ""
    solution = ""

    # Formato 1: message_1 / message_2
    if "message_1" in entry and "message_2" in entry:
        problem = entry["message_1"]
        solution = entry["message_2"]

    # Formato 2: Cada fila es un solo turno de conversación
    elif "role_1" in entry and "role_2" in entry:
        if "topic" in entry:
            problem = entry.get("message_1", entry.get("content", ""))
            solution = entry.get("message_2", "")

    # Formato 3: conversation-style
    elif "conversations" in entry:
        convs = entry["conversations"]
        if len(convs) >= 2:
            problem = convs[0].get("value", "")
            solution = convs[1].get("value", "")

    if not problem or not solution:
        return None

    # Limpiar prefijos de role-play típicos de camel-ai
    # E.g., "Human: ...", "Assistant: ...", "Physics Teacher: ..."
    for prefix in ["Human:", "Assistant:", "User:", "AI:", "Teacher:",
                    "Physics Teacher:", "Student:", "Physicist:"]:
        if problem.startswith(prefix):
            problem = problem[len(prefix):].strip()
        if solution.startswith(prefix):
            solution = solution[len(prefix):].strip()

    return {"problem": problem, "raw_solution": solution}


def detect_physics_topic(text: str) -> str:
    """
    Detecta el tema de física de un problema por palabras clave.

    Args:
        text: Texto del problema.

    Returns:
        Tema estimado.
    """
    t = text.lower()

    if any(w in t for w in ["velocity", "acceleration", "speed", "distance",
                             "displacement", "motion", "trajectory", "projectile"]):
        return "kinematics"
    if any(w in t for w in ["force", "newton", "friction", "tension",
                             "weight", "mass", "momentum", "torque"]):
        return "dynamics"
    if any(w in t for w in ["energy", "work", "power", "kinetic", "potential",
                             "conservation of energy", "joule"]):
        return "energy"
    if any(w in t for w in ["electric", "voltage", "current", "resistance",
                             "ohm", "circuit", "capacitor", "charge", "coulomb"]):
        return "electricity"
    if any(w in t for w in ["magnet", "magnetic", "inductor", "flux",
                             "electromagnetic"]):
        return "magnetism"
    if any(w in t for w in ["temperature", "heat", "thermal", "entropy",
                             "thermodynamic", "celsius", "kelvin", "specific heat"]):
        return "thermodynamics"
    if any(w in t for w in ["wave", "frequency", "wavelength", "amplitude",
                             "sound", "oscillation", "pendulum", "hertz"]):
        return "waves"
    if any(w in t for w in ["lens", "mirror", "refraction", "reflection",
                             "light", "photon", "optic"]):
        return "optics"
    if any(w in t for w in ["fluid", "pressure", "buoyancy", "bernoulli",
                             "density", "pascal", "archimedes"]):
        return "fluids"
    if any(w in t for w in ["gravity", "gravitational", "orbit", "satellite",
                             "kepler", "planet"]):
        return "gravitation"

    return "general_physics"


def normalize_physics_solution(raw: str, max_steps: int = 8) -> str:
    """
    Convierte una solución de física al formato Step/Answer.

    Las soluciones de camel-ai suelen ser largas y detalladas.
    Las condensamos en pasos numerados + Answer.

    Args:
        raw: Solución cruda (puede ser muy larga).
        max_steps: Máximo de pasos a mantener.

    Returns:
        Solución normalizada.
    """
    # Limpiar
    text = raw.strip()

    # Quitar asteriscos de markdown
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)

    # Dividir en líneas/párrafos significativos
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n|\n', text) if p.strip()]

    if not paragraphs:
        return text

    # Intentar extraer una respuesta final
    answer = ""
    # Buscar patrones de respuesta al final
    for line in reversed(paragraphs[-3:]):
        # Patrones comunes de respuesta
        patterns = [
            r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+(.+?)\.?\s*$",
            r"(?:therefore|thus|so|hence),?\s+(.+?)\s*\.?\s*$",
            r"=\s*([-\d.,]+\s*\w*)\s*\.?\s*$",
            r"^Answer:\s*(.+)$",
        ]
        for pat in patterns:
            m = re.search(pat, line, re.IGNORECASE)
            if m:
                answer = m.group(1).strip().rstrip(".")
                break
        if answer:
            break

    # Construir pasos (condensar si hay demasiados)
    steps = []
    for p in paragraphs:
        # Saltar líneas muy cortas o solo números
        if len(p) < 15:
            continue
        # Saltar si es la línea de la respuesta
        if answer and answer in p and ("answer" in p.lower() or "therefore" in p.lower()):
            continue
        # Limpiar numeración existente
        clean = re.sub(r'^(?:Step\s+)?\d+[.):]\s*', '', p).strip()
        if clean:
            steps.append(clean)

    # Limitar pasos
    if len(steps) > max_steps:
        # Mantener inicio y fin, condensar medio
        steps = steps[:3] + steps[-(max_steps-3):]

    # Construir resultado
    numbered = [f"Step {i}: {s}" for i, s in enumerate(steps, 1)]

    if answer:
        return "\n".join(numbered) + f"\nAnswer: {answer}"
    elif numbered:
        return "\n".join(numbered)
    else:
        return text


def convert_camel_data(
    raw_data: List[Dict],
    max_problem_chars: int = 400,
    max_solution_chars: int = 800,
    max_problems: Optional[int] = None,
) -> List[Dict]:
    """
    Convierte datos brutos de camel-ai al esquema unificado.

    Args:
        raw_data: Datos descargados.
        max_problem_chars: Máx. caracteres del enunciado.
        max_solution_chars: Máx. caracteres de la solución final.
        max_problems: Limitar número de problemas.

    Returns:
        Lista con esquema unificado.
    """
    converted = []
    skipped = {"no_extract": 0, "too_long_problem": 0,
               "too_long_solution": 0, "no_answer": 0}

    for entry in raw_data:
        qa = extract_qa_from_camel(entry)
        if not qa:
            skipped["no_extract"] += 1
            continue

        problem = qa["problem"].strip()
        if len(problem) > max_problem_chars:
            skipped["too_long_problem"] += 1
            continue

        solution = normalize_physics_solution(qa["raw_solution"])
        if len(solution) > max_solution_chars:
            skipped["too_long_solution"] += 1
            continue

        # Verificar que tenga Answer:
        if "Answer:" not in solution:
            skipped["no_answer"] += 1
            continue

        converted.append({
            "problem": problem,
            "solution": solution,
            "domain": "physics",
            "source": "CamelPhysics",
            "topic": detect_physics_topic(problem),
            "split": "train",
        })

        if max_problems and len(converted) >= max_problems:
            break

    print(f"\n  Convertidos: {len(converted)}")
    print(f"  Omitidos:")
    for reason, count in skipped.items():
        if count > 0:
            print(f"    {reason}: {count}")

    return converted


def main():
    parser = argparse.ArgumentParser(description="Convertir camel-ai/physics")
    parser.add_argument("--max", type=int, default=5000,
                        help="Máximo de problemas a incluir (default: 5000)")
    args = parser.parse_args()

    print("=" * 60)
    print("  CONVERSIÓN CAMEL-AI/PHYSICS → ESQUEMA UNIFICADO")
    print("=" * 60)

    # Descargar (streaming, limitar crudas a 2x max para tener margen de filtrado)
    raw_data = download_camel_physics(max_raw=min(args.max * 3, 15000))

    if not raw_data:
        print("No se pudieron descargar datos.")
        return

    # Convertir
    print(f"\nConvirtiendo (max {args.max} problemas)...")
    converted = convert_camel_data(raw_data, max_problems=args.max)

    if not converted:
        print("No se convirtieron problemas.")
        return

    # Estadísticas de temas
    topics = {}
    for p in converted:
        t = p.get("topic", "unknown")
        topics[t] = topics.get(t, 0) + 1

    print(f"\n  Distribución por tema:")
    for t, c in sorted(topics.items(), key=lambda x: -x[1]):
        print(f"    {t}: {c}")

    # Guardar
    output_path = DATA_DIR / "camel_physics_clean.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    print(f"\n  Guardado: {output_path}")

    # Ejemplo
    if converted:
        ex = converted[0]
        print(f"\n  Ejemplo:")
        print(f"    problem: {ex['problem'][:120]}...")
        print(f"    solution: {ex['solution'][:150]}...")
        print(f"    topic: {ex['topic']}")

    print(f"\n{'=' * 60}")
    print("  CONVERSIÓN COMPLETADA")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
