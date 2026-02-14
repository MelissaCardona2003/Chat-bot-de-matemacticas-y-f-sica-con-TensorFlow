"""
evaluate_math_physics.py — Evaluación del modelo Transformer en test set.

Carga el modelo entrenado y evalúa:
1. Token-level accuracy (la misma métrica del entrenamiento)
2. Exact match de la línea Answer: (extrae respuesta predicha vs esperada)
3. Resultados desglosados por dominio (math vs physics)

Produce un reporte claro para el profesor.

Uso:
    python evaluation/evaluate_math_physics.py
"""

import os
import sys
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# GPU/XLA config ANTES de TF
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_cuda_data_dir=/usr/local/cuda-12.8")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf
import numpy as np

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    _original_cast = tf.cast
    def _blackwell_cast(x, dtype, name=None):
        if tf.executing_eagerly():
            with tf.device('/CPU:0'):
                return _original_cast(x, dtype, name=name)
        return _original_cast(x, dtype, name=name)
    tf.cast = _blackwell_cast

from transformer_math_physics_tutor.data.tokenizer import CharTokenizer
from transformer_math_physics_tutor.data.dataset_builder import create_datasets_from_combined
from transformer_math_physics_tutor.models.config import TransformerConfig
from transformer_math_physics_tutor.models.transformer import Transformer
from transformer_math_physics_tutor.training.losses import loss_function, accuracy_function
from transformer_math_physics_tutor.inference.generate import generate_text


def extract_answer_line(text: str) -> str:
    """Extrae el contenido de la línea 'Answer: ...' de una solución."""
    match = re.search(r"Answer:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().rstrip(".")
    return ""


def normalize_number(s: str) -> str:
    """Normaliza un string numérico para comparación más justa."""
    s = s.strip().replace(",", "").replace("$", "").replace(" ", "")
    # Eliminar unidades comunes al final
    s = re.sub(r"\s*(kg|m|s|N|J|W|V|A|Hz|Pa|°[CF]|degrees|kWh|km|cm|Ω|mph|miles|hours|seconds)\.?$",
               "", s, flags=re.IGNORECASE).strip()
    try:
        val = float(s)
        # Si es entero, mostrarlo como entero
        if val == int(val):
            return str(int(val))
        return f"{val:.4f}".rstrip("0").rstrip(".")
    except ValueError:
        return s.lower()


def evaluate_token_accuracy(model, config, dataset, label=""):
    """
    Evalúa token-level accuracy del modelo en un dataset (misma métrica del training).

    Returns:
        (loss, accuracy)
    """
    with tf.device('/CPU:0'):
        val_loss = tf.keras.metrics.Mean()
        val_accuracy = tf.keras.metrics.Mean()

    @tf.function
    def eval_step(inp, dec_inp, dec_target):
        predictions = model((inp, dec_inp), training=False)
        loss = loss_function(dec_target, predictions, config.label_smoothing)
        acc = accuracy_function(dec_target, predictions)
        return loss, acc

    n_batches = 0
    for (inp, dec_inp), dec_target in dataset:
        loss, acc = eval_step(inp, dec_inp, dec_target)
        with tf.device('/CPU:0'):
            val_loss(loss)
            val_accuracy(acc)
        n_batches += 1

    with tf.device('/CPU:0'):
        final_loss = val_loss.result().numpy()
        final_acc = val_accuracy.result().numpy()

    if label:
        print(f"  {label}: Loss={final_loss:.4f}, Acc={final_acc:.4f} ({n_batches} batches)")

    return final_loss, final_acc


def evaluate_exact_match(model, tokenizer, test_problems: List[Dict],
                          max_samples: int = 100) -> Dict:
    """
    Evalúa exact match de la línea Answer: en problemas del test set.

    Para cada problema:
    1. Genera la solución completa con generate_text
    2. Extrae la línea Answer: de predicción y referencia
    3. Compara normalizando números

    Args:
        model: Modelo Transformer.
        tokenizer: CharTokenizer.
        test_problems: Lista de dicts con problem, solution, domain.
        max_samples: Máximo de problemas a evaluar (generación es lenta).

    Returns:
        Dict con métricas desglosadas por dominio.
    """
    if max_samples and len(test_problems) > max_samples:
        # Muestrear uniformemente
        import random
        rng = random.Random(42)
        test_problems = rng.sample(test_problems, max_samples)

    results = {
        "total": 0,
        "correct": 0,
        "by_domain": {},
        "examples": [],
    }

    print(f"\n  Evaluando exact match en {len(test_problems)} problemas...")

    for i, prob in enumerate(test_problems):
        domain = prob.get("domain", "unknown")
        if domain not in results["by_domain"]:
            results["by_domain"][domain] = {"total": 0, "correct": 0}

        # Generar respuesta
        with tf.device('/CPU:0'):
            generated = generate_text(
                model, tokenizer, prob["problem"],
                max_length=300, temperature=0.3, top_k=10,
                repetition_penalty=1.3,
            )

        # Extraer Answer: de referencia y predicción
        ref_answer = extract_answer_line(prob["solution"])
        pred_answer = extract_answer_line(generated)

        # Comparar
        ref_norm = normalize_number(ref_answer)
        pred_norm = normalize_number(pred_answer)
        is_correct = (ref_norm == pred_norm) and ref_norm != ""

        results["total"] += 1
        results["by_domain"][domain]["total"] += 1
        if is_correct:
            results["correct"] += 1
            results["by_domain"][domain]["correct"] += 1

        # Guardar algunos ejemplos
        if len(results["examples"]) < 10 or (is_correct and len([e for e in results["examples"] if e["correct"]]) < 5):
            results["examples"].append({
                "problem": prob["problem"][:120],
                "ref_answer": ref_answer,
                "pred_answer": pred_answer,
                "correct": is_correct,
                "domain": domain,
            })

        if (i + 1) % 20 == 0:
            pct = results["correct"] / results["total"] * 100
            print(f"    Progreso: {i+1}/{len(test_problems)} — "
                  f"EM={pct:.1f}% ({results['correct']}/{results['total']})")

    return results


def main():
    print("=" * 60)
    print("  EVALUACIÓN DEL MODELO TRANSFORMER")
    print("  Math + Physics")
    print("=" * 60)

    # ── Cargar modelo ──────────────────────────────────────
    checkpoint_dir = BASE_DIR / "checkpoints"
    config = TransformerConfig.load(str(checkpoint_dir / "config.json"))
    tokenizer = CharTokenizer(str(checkpoint_dir / "vocab.json"))

    with tf.device('/CPU:0'):
        model = Transformer(config)
        dummy_enc = tf.zeros((1, config.max_encoder_len), dtype=tf.int32)
        dummy_dec = tf.zeros((1, config.max_decoder_len), dtype=tf.int32)
        _ = model((dummy_enc, dummy_dec), training=False)

        weights_path = str(checkpoint_dir / "best_model.weights.h5")
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
            print(f"\n  Pesos cargados desde: best_model.weights.h5")
        else:
            model.load_weights(str(checkpoint_dir / "model_weights.weights.h5"))
            print(f"\n  Pesos cargados desde: model_weights.weights.h5")

    print(f"  Parámetros: {model.count_params():,}")

    # ── Cargar datos ───────────────────────────────────────
    data_file = "combined_math_physics.json"
    data_path = BASE_DIR / "data" / data_file

    if not data_path.exists():
        print(f"\n  ⚠ No se encontró {data_file}")
        print("  Ejecuta primero: python data/build_combined_dataset.py")
        return

    # Crear datasets
    print(f"\n{'='*60}")
    print("  CARGANDO DATASETS")
    print(f"{'='*60}")

    train_ds, val_ds, test_ds, _ = create_datasets_from_combined(
        data_file=data_file,
        tokenizer=tokenizer,
        max_problem_len=config.max_encoder_len,
        max_solution_len=config.max_decoder_len,
        batch_size=32,
        build_vocab=False,
    )

    # ── Token-level accuracy ───────────────────────────────
    print(f"\n{'='*60}")
    print("  1. TOKEN-LEVEL ACCURACY (misma métrica del entrenamiento)")
    print(f"{'='*60}")

    start = time.time()
    val_loss, val_acc = evaluate_token_accuracy(model, config, val_ds, "Validación")
    if test_ds:
        test_loss, test_acc = evaluate_token_accuracy(model, config, test_ds, "Test")
    else:
        test_loss = test_acc = None
    t_token = time.time() - start

    # ── Exact Match ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  2. EXACT MATCH (línea Answer:)")
    print(f"{'='*60}")

    # Cargar problemas del test split
    with open(data_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    test_problems = [p for p in all_data if p.get("split") == "test"]
    print(f"  Problemas de test: {len(test_problems)}")

    # Limitar a 100 para que no tome horas
    start = time.time()
    em_results = evaluate_exact_match(model, tokenizer, test_problems, max_samples=100)
    t_em = time.time() - start

    # ── Reporte ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  REPORTE DE EVALUACIÓN")
    print(f"{'='*60}")

    print(f"\n  Modelo: Transformer ({model.count_params():,} params)")
    print(f"  d_model={config.d_model}, heads={config.num_heads}, "
          f"layers={config.num_layers}, dff={config.dff}")
    print(f"  Dataset: {data_file}")

    print(f"\n  ─── Token-level Accuracy ───")
    print(f"  Validación:  Loss={val_loss:.4f}  Acc={val_acc:.4f} ({val_acc*100:.1f}%)")
    if test_acc is not None:
        print(f"  Test:        Loss={test_loss:.4f}  Acc={test_acc:.4f} ({test_acc*100:.1f}%)")

    print(f"\n  ─── Exact Match (Answer:) ───")
    em_total = em_results["total"]
    em_correct = em_results["correct"]
    em_pct = em_correct / max(em_total, 1) * 100
    print(f"  Global: {em_correct}/{em_total} = {em_pct:.1f}%")

    for domain, stats in sorted(em_results["by_domain"].items()):
        d_total = stats["total"]
        d_correct = stats["correct"]
        d_pct = d_correct / max(d_total, 1) * 100
        print(f"    {domain}: {d_correct}/{d_total} = {d_pct:.1f}%")

    print(f"\n  ─── Ejemplos de predicciones ───")
    for ex in em_results["examples"][:8]:
        status = "✅" if ex["correct"] else "❌"
        print(f"  {status} [{ex['domain']}] {ex['problem'][:80]}...")
        print(f"     Esperado: {ex['ref_answer']}")
        print(f"     Predicho: {ex['pred_answer']}")

    print(f"\n  ─── Tiempos ───")
    print(f"  Token accuracy: {t_token:.1f}s")
    print(f"  Exact match ({em_total} problemas): {t_em:.1f}s "
          f"({t_em/max(em_total,1):.1f}s/problema)")

    # Guardar reporte
    report = {
        "model_params": model.count_params(),
        "config": {
            "d_model": config.d_model, "num_heads": config.num_heads,
            "num_layers": config.num_layers, "dff": config.dff,
        },
        "token_accuracy": {
            "val_loss": float(val_loss), "val_acc": float(val_acc),
            "test_loss": float(test_loss) if test_loss else None,
            "test_acc": float(test_acc) if test_acc else None,
        },
        "exact_match": {
            "total": em_total, "correct": em_correct,
            "pct": round(em_pct, 2),
            "by_domain": em_results["by_domain"],
        },
        "examples": em_results["examples"],
    }

    report_path = BASE_DIR / "checkpoints" / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Reporte guardado: {report_path}")

    print(f"\n{'='*60}")
    print("  EVALUACIÓN COMPLETADA")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
