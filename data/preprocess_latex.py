"""
preprocess_latex.py — Limpieza de notación LaTeX para tokenización.

Convierte notación LaTeX matemática a texto plano más legible,
facilitando la tokenización a nivel de carácter.

Transformaciones principales:
- \\frac{a}{b} → (a/b)
- \\sqrt{x} → sqrt(x)
- \\textbf{x}, \\emph{x}, \\text{x} → x
- $...$ → ...
- \\boxed{x} → x
- \\cdot, \\times → *
- \\leq → <=, \\geq → >=, \\neq → !=

Uso:
    python data/preprocess_latex.py

    # O como módulo:
    from transformer_math_physics_tutor.data.preprocess_latex import clean_latex
    clean = clean_latex("\\frac{3}{4}")  # → "(3/4)"
"""

import re
import json
from pathlib import Path
from typing import List, Dict


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def clean_latex(text: str) -> str:
    """
    Limpia notación LaTeX de un texto matemático.

    Aplica transformaciones en orden específico para manejar
    correctamente las dependencias entre reglas.

    Args:
        text: Texto con notación LaTeX.

    Returns:
        Texto limpio sin LaTeX.
    """
    if not text:
        return text

    result = text

    # ── 1. Quitar math mode delimiters ────────────────────
    # $$...$$ (display math) antes que $...$
    result = re.sub(r'\$\$(.*?)\$\$', r'\1', result, flags=re.DOTALL)
    # $...$ (inline math)
    result = re.sub(r'\$(.*?)\$', r'\1', result)
    # \[...\] y \(...\)
    result = re.sub(r'\\\[(.*?)\\\]', r'\1', result, flags=re.DOTALL)
    result = re.sub(r'\\\((.*?)\\\)', r'\1', result)

    # ── 2. Quitar \boxed{...} (respuesta final en MATH dataset) ─
    result = _replace_command(result, "boxed")

    # ── 3. Quitar comandos de formato ─────────────────────
    for cmd in ["textbf", "textit", "emph", "text", "mathrm",
                "mathbf", "mathit", "mathcal", "mathbb", "operatorname"]:
        result = _replace_command(result, cmd)

    # ── 4. Convertir fracciones ───────────────────────────
    # \frac{a}{b} → (a/b)  — manejar anidadas con múltiples pasadas
    for _ in range(5):  # Varias pasadas para fracciones anidadas
        new_result = re.sub(
            r'\\frac\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
            r'\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
            r'(\1/\2)',
            result
        )
        if new_result == result:
            break
        result = new_result

    # Forma alternativa: \dfrac, \tfrac
    for frac_cmd in ["dfrac", "tfrac"]:
        for _ in range(5):
            new_result = re.sub(
                rf'\\{frac_cmd}\s*\{{([^{{}}]*(?:\{{[^{{}}]*\}}[^{{}}]*)*)\}}'
                rf'\s*\{{([^{{}}]*(?:\{{[^{{}}]*\}}[^{{}}]*)*)\}}',
                r'(\1/\2)',
                result
            )
            if new_result == result:
                break
            result = new_result

    # ── 5. Convertir raíces ───────────────────────────────
    # \sqrt[n]{x} → nroot(x, n)
    result = re.sub(
        r'\\sqrt\s*\[([^\]]+)\]\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
        r'nroot(\2, \1)',
        result
    )
    # \sqrt{x} → sqrt(x)
    result = _replace_command(result, "sqrt", wrapper="sqrt(", closer=")")

    # ── 6. Operadores comunes ─────────────────────────────
    replacements = {
        r'\\cdot': '*',
        r'\\times': '*',
        r'\\div': '/',
        r'\\pm': '+-',
        r'\\mp': '-+',
        r'\\leq': '<=',
        r'\\le': '<=',
        r'\\geq': '>=',
        r'\\ge': '>=',
        r'\\neq': '!=',
        r'\\ne': '!=',
        r'\\approx': '~=',
        r'\\equiv': '===',
        r'\\rightarrow': '->',
        r'\\leftarrow': '<-',
        r'\\Rightarrow': '=>',
        r'\\implies': '=>',
        r'\\to': '->',
        r'\\iff': '<=>',
        r'\\in': 'in',
        r'\\notin': 'not in',
        r'\\subset': 'subset',
        r'\\cup': 'union',
        r'\\cap': 'intersect',
    }

    for pattern, replacement in replacements.items():
        result = re.sub(pattern + r'(?![a-zA-Z])', replacement, result)

    # ── 7. Constantes y funciones ─────────────────────────
    constants = {
        r'\\pi': 'pi',
        r'\\infty': 'inf',
        r'\\alpha': 'alpha',
        r'\\beta': 'beta',
        r'\\gamma': 'gamma',
        r'\\delta': 'delta',
        r'\\theta': 'theta',
        r'\\lambda': 'lambda',
        r'\\mu': 'mu',
        r'\\sigma': 'sigma',
        r'\\omega': 'omega',
        r'\\epsilon': 'epsilon',
        r'\\phi': 'phi',
        r'\\psi': 'psi',
    }

    for pattern, replacement in constants.items():
        result = re.sub(pattern + r'(?![a-zA-Z])', replacement, result)

    # Funciones trigonométricas (quitar \)
    for func in ["sin", "cos", "tan", "cot", "sec", "csc",
                 "arcsin", "arccos", "arctan",
                 "log", "ln", "exp", "lim", "max", "min",
                 "gcd", "lcm", "det", "dim"]:
        result = re.sub(rf'\\{func}(?![a-zA-Z])', func, result)

    # ── 8. Quitar \left y \right ──────────────────────────
    result = re.sub(r'\\left\s*', '', result)
    result = re.sub(r'\\right\s*', '', result)
    result = re.sub(r'\\big\s*', '', result)
    result = re.sub(r'\\Big\s*', '', result)
    result = re.sub(r'\\bigg\s*', '', result)
    result = re.sub(r'\\Bigg\s*', '', result)

    # ── 9. Entornos LaTeX ─────────────────────────────────
    result = re.sub(r'\\begin\{[^}]*\}', '', result)
    result = re.sub(r'\\end\{[^}]*\}', '', result)
    result = re.sub(r'\\\\', '\n', result)  # Newline en LaTeX
    result = re.sub(r'\\&', '&', result)

    # ── 10. Limpiar comandos LaTeX restantes ──────────────
    # Quitar \comando{contenido} genérico no procesado
    result = re.sub(r'\\[a-zA-Z]+\s*\{([^{}]*)\}', r'\1', result)
    # Quitar \comando solo (sin argumentos)
    result = re.sub(r'\\[a-zA-Z]+', '', result)
    # Quitar \ solitarios
    result = re.sub(r'\\(?![a-zA-Z])', '', result)

    # ── 11. Limpiar llaves sueltas ────────────────────────
    result = result.replace('{', '').replace('}', '')

    # ── 12. Normalizar espacios ───────────────────────────
    result = re.sub(r'[ \t]+', ' ', result)
    result = re.sub(r'\n\s*\n', '\n', result)
    result = result.strip()

    return result


def _replace_command(
    text: str,
    command: str,
    wrapper: str = "",
    closer: str = ""
) -> str:
    """
    Reemplaza un comando LaTeX \\command{contenido} por wrapper+contenido+closer.

    Maneja llaves anidadas correctamente buscando la llave de cierre matching.

    Args:
        text: Texto a procesar.
        command: Nombre del comando LaTeX (sin \\).
        wrapper: Texto a poner antes del contenido.
        closer: Texto a poner después del contenido.

    Returns:
        Texto con el comando reemplazado.
    """
    pattern = f"\\{command}"
    while pattern in text:
        idx = text.find(pattern)
        # Buscar la llave de apertura
        after = idx + len(pattern)
        # Saltar espacios
        while after < len(text) and text[after] == ' ':
            after += 1

        if after >= len(text) or text[after] != '{':
            # No hay llave, quitar solo el comando
            text = text[:idx] + text[after:]
            continue

        # Encontrar la llave de cierre matching
        depth = 0
        end = after
        for i in range(after, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break

        content = text[after + 1:end]
        text = text[:idx] + wrapper + content + closer + text[end + 1:]

    return text


def process_dataset(
    input_path: Path,
    output_path: Path
) -> List[Dict]:
    """
    Procesa un dataset JSON aplicando clean_latex a problem y solution.

    Args:
        input_path: Ruta al archivo JSON de entrada.
        output_path: Ruta al archivo JSON de salida.

    Returns:
        Lista de problemas procesados.
    """
    print(f"  Cargando: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        problems = json.load(f)

    print(f"  Problemas a procesar: {len(problems)}")

    cleaned = []
    for prob in problems:
        clean_prob = prob.copy()
        clean_prob["problem"] = clean_latex(prob["problem"])
        clean_prob["solution"] = clean_latex(prob["solution"])

        # Filtrar problemas vacíos tras limpieza
        if clean_prob["problem"].strip() and clean_prob["solution"].strip():
            cleaned.append(clean_prob)

    print(f"  Problemas tras limpieza: {len(cleaned)}")

    # Guardar
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    print(f"  Guardado: {output_path}")

    return cleaned


def main():
    """Pipeline principal de preprocesamiento LaTeX."""
    print("=" * 60)
    print("  PREPROCESAMIENTO LATEX")
    print("=" * 60)

    # Buscar datasets para procesar
    processed_any = False

    # 1. Procesar math_filtered.json → math_clean.json
    filtered_path = DATA_DIR / "math_filtered.json"
    if filtered_path.exists():
        print(f"\nProcesando math_filtered.json...")
        clean_data = process_dataset(
            filtered_path,
            DATA_DIR / "math_clean.json"
        )
        processed_any = True

        # También procesar test set si existe
        filtered_test = DATA_DIR / "math_filtered_test.json"
        if filtered_test.exists():
            print(f"\nProcesando math_filtered_test.json...")
            process_dataset(
                filtered_test,
                DATA_DIR / "math_clean_test.json"
            )

    # 2. Procesar math_training_data.json si no hay filtrado
    if not processed_any:
        training_path = DATA_DIR / "math_training_data.json"
        if training_path.exists():
            print(f"\nNo se encontró math_filtered.json.")
            print(f"Procesando math_training_data.json como alternativa...")
            clean_data = process_dataset(
                training_path,
                DATA_DIR / "math_clean.json"
            )
            processed_any = True

    if not processed_any:
        print("\nNo se encontró ningún dataset para procesar.")
        print("Ejecutar primero:")
        print("  python data/download_dataset.py")
        print("  python data/filter_dataset.py")
        return

    # Mostrar ejemplos
    if clean_data:
        print(f"\n--- Ejemplos de limpieza ---")
        for i, prob in enumerate(clean_data[:3], 1):
            print(f"\n  Ejemplo {i}:")
            print(f"    Problema: {prob['problem'][:100]}")
            print(f"    Solución: {prob['solution'][:100]}")

    print(f"\n{'=' * 60}")
    print(f"  PREPROCESAMIENTO COMPLETADO")
    print(f"{'=' * 60}")


def run_tests():
    """Ejecuta tests unitarios de clean_latex."""
    print("\n" + "=" * 60)
    print("  TESTS UNITARIOS: clean_latex()")
    print("=" * 60)

    test_cases = [
        # (input, expected_output, description)
        (r"\frac{3}{4}", "(3/4)", "Fracción simple"),
        (r"\frac{a+b}{c}", "(a+b/c)", "Fracción con expresión"),
        (r"\sqrt{16}", "sqrt(16)", "Raíz cuadrada"),
        (r"\sqrt[3]{27}", "nroot(27, 3)", "Raíz cúbica"),
        (r"\textbf{hello}", "hello", "Negrita"),
        (r"\emph{world}", "world", "Énfasis"),
        (r"\text{answer}", "answer", "Texto"),
        (r"$x^2 + 1$", "x^2 + 1", "Math mode inline"),
        (r"$$x = 5$$", "x = 5", "Math mode display"),
        (r"\boxed{42}", "42", "Boxed"),
        (r"\boxed{x = 5}", "x = 5", "Boxed con expresión"),
        (r"a \cdot b", "a * b", "cdot"),
        (r"a \times b", "a * b", "times"),
        (r"a \div b", "a / b", "div"),
        (r"x \leq 5", "x <= 5", "leq"),
        (r"x \geq 3", "x >= 3", "geq"),
        (r"x \neq 0", "x != 0", "neq"),
        (r"\pi r^2", "pi r^2", "pi"),
        (r"x \to \infty", "x -> inf", "Infinito"),  # \to → ->
        (r"\sin(x)", "sin(x)", "Seno"),
        (r"\cos(\theta)", "cos(theta)", "Coseno con theta"),
        (r"\log_2(8)", "log_2(8)", "Logaritmo"),
        (r"\left(\frac{1}{2}\right)", "((1/2))", "left/right con fracción"),
        (r"3 + 5 = 8", "3 + 5 = 8", "Sin LaTeX (pass-through)"),
        ("", "", "String vacío"),
    ]

    passed = 0
    failed = 0

    for latex_in, expected, desc in test_cases:
        result = clean_latex(latex_in)
        ok = result == expected

        if ok:
            passed += 1
            status = "OK"
        else:
            failed += 1
            status = "FAIL"

        print(f"  [{status}] {desc}")
        if not ok:
            print(f"         Input:    '{latex_in}'")
            print(f"         Expected: '{expected}'")
            print(f"         Got:      '{result}'")

    print(f"\n  Resultado: {passed}/{passed + failed} tests pasaron")
    if failed > 0:
        print(f"  ⚠ {failed} tests fallaron")
    else:
        print(f"  Todos los tests pasaron correctamente")

    return failed == 0


if __name__ == "__main__":
    # Correr tests primero
    run_tests()

    # Luego procesar dataset
    print()
    main()
