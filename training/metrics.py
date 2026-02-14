"""
metrics.py — Métricas de evaluación: Exact Match y Validación Simbólica.

Implementa métricas para evaluar la calidad de las respuestas generadas:
1. Exact Match: comparación de strings exactos
2. Validación Simbólica: usa SymPy para verificar equivalencia matemática

Uso:
    from transformer_math_physics_tutor.training.metrics import exact_match_accuracy, symbolic_validation
    em = exact_match_accuracy(predictions, targets, tokenizer)
    is_correct = symbolic_validation("x + 1", "1 + x")
"""

import re
from typing import List, Optional
import numpy as np


def exact_match_accuracy(
    predictions: List,
    targets: List,
    tokenizer=None
) -> float:
    """
    Calcula la accuracy de coincidencia exacta de strings.

    Decodifica las secuencias predichas y las compara con las secuencias
    objetivo. Solo cuenta como correcto si son idénticas (tras limpiar).

    Args:
        predictions: Lista de secuencias predichas (indices o strings).
        targets: Lista de secuencias objetivo (indices o strings).
        tokenizer: Instancia de CharTokenizer para decodificar índices.
                   Si None, asume que predictions y targets ya son strings.

    Returns:
        Proporción de predicciones que coinciden exactamente (0.0 - 1.0).
    """
    if len(predictions) == 0:
        return 0.0

    correct = 0
    total = len(predictions)

    for pred, target in zip(predictions, targets):
        # Decodificar si es necesario
        if tokenizer is not None:
            if hasattr(pred, 'numpy'):
                pred = pred.numpy()
            if hasattr(target, 'numpy'):
                target = target.numpy()

            pred_text = tokenizer.decode(list(pred), skip_special_tokens=True)
            target_text = tokenizer.decode(list(target), skip_special_tokens=True)
        else:
            pred_text = str(pred)
            target_text = str(target)

        # Limpiar espacios para comparación más justa
        pred_clean = pred_text.strip()
        target_clean = target_text.strip()

        if pred_clean == target_clean:
            correct += 1

    accuracy = correct / total
    return accuracy


def extract_final_answer(text: str) -> str:
    """
    Extrae la respuesta final numérica o expresión de un texto de solución.

    Busca patrones comunes como "= 42", "the answer is 42", etc.

    Args:
        text: Texto de la solución completa.

    Returns:
        String con la respuesta final extraída, o el texto completo si no se encuentra patrón.
    """
    # Buscar patrones de respuesta final
    patterns = [
        r"=\s*([^=\n]+?)$",           # Último "= algo"
        r"(?:answer|resultado)\s*(?:is|es|:)\s*(.+?)$",
        r"\\boxed\{(.+?)\}",          # \boxed{respuesta}
        r"(\d+(?:\.\d+)?)\s*$",       # Número al final
    ]

    text = text.strip()

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()

    return text


def symbolic_validation(
    pred_text: str,
    target_text: str
) -> bool:
    """
    Valida si dos expresiones matemáticas son simbólicamente equivalentes.

    Usa SymPy para parsear ambas expresiones y verificar si
    simplify(pred - target) == 0.

    Esto captura equivalencias como:
    - "x + 1" == "1 + x"
    - "2/4" == "1/2"
    - "x^2 - 1" == "(x-1)(x+1)"

    Args:
        pred_text: Texto de la predicción.
        target_text: Texto del objetivo.

    Returns:
        True si las expresiones son simbólicamente equivalentes.
        False si no lo son o si no se pudieron parsear.
    """
    try:
        import sympy
    except ImportError:
        print("SymPy no está instalado. Instálalo con: pip install sympy")
        return False

    try:
        # Extraer respuesta final
        pred_answer = extract_final_answer(pred_text)
        target_answer = extract_final_answer(target_text)

        # Limpiar para SymPy
        pred_clean = pred_answer.replace("^", "**").strip()
        target_clean = target_answer.replace("^", "**").strip()

        # Parsear expresiones
        pred_expr = sympy.sympify(pred_clean)
        target_expr = sympy.sympify(target_clean)

        # Verificar equivalencia: simplify(pred - target) == 0
        diff = sympy.simplify(pred_expr - target_expr)
        return diff == 0

    except (sympy.SympifyError, TypeError, ValueError, SyntaxError, AttributeError):
        # Si no se puede parsear, hacer comparación de string
        return pred_text.strip() == target_text.strip()


def evaluate_batch(
    predictions: List[str],
    targets: List[str],
    tokenizer=None
) -> dict:
    """
    Evalúa un batch de predicciones con múltiples métricas.

    Args:
        predictions: Lista de predicciones (strings o índices).
        targets: Lista de objetivos (strings o índices).
        tokenizer: Tokenizer para decodificar si son índices.

    Returns:
        Diccionario con métricas: exact_match, symbolic_match, total.
    """
    # Decodificar si es necesario
    if tokenizer is not None:
        pred_texts = [
            tokenizer.decode(list(p.numpy() if hasattr(p, 'numpy') else p),
                           skip_special_tokens=True)
            for p in predictions
        ]
        target_texts = [
            tokenizer.decode(list(t.numpy() if hasattr(t, 'numpy') else t),
                           skip_special_tokens=True)
            for t in targets
        ]
    else:
        pred_texts = [str(p) for p in predictions]
        target_texts = [str(t) for t in targets]

    # Exact match
    em_correct = sum(
        1 for p, t in zip(pred_texts, target_texts)
        if p.strip() == t.strip()
    )

    # Symbolic match
    sym_correct = sum(
        1 for p, t in zip(pred_texts, target_texts)
        if symbolic_validation(p, t)
    )

    total = len(predictions)

    return {
        "exact_match": em_correct / max(total, 1),
        "symbolic_match": sym_correct / max(total, 1),
        "exact_match_count": em_correct,
        "symbolic_match_count": sym_correct,
        "total": total
    }


if __name__ == "__main__":
    print("=" * 60)
    print("DEMO: Métricas de Evaluación")
    print("=" * 60)

    # Test exact match
    preds = ["42", "x + 1", "3/4"]
    targets = ["42", "1 + x", "0.75"]

    em = exact_match_accuracy(preds, targets)
    print(f"Exact Match: {em:.2%}")

    # Test symbolic validation
    test_cases = [
        ("42", "42", True),
        ("x + 1", "1 + x", True),
        ("2/4", "1/2", True),
        ("x**2 - 1", "(x-1)*(x+1)", True),
        ("3", "5", False),
    ]

    print("\nValidación Simbólica:")
    for pred, target, expected in test_cases:
        result = symbolic_validation(pred, target)
        status = "OK" if result == expected else "FAIL"
        print(f"  [{status}] '{pred}' == '{target}' → {result}")
