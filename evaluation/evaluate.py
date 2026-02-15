"""
evaluate_v3_easy.py — Evaluación del modelo Transformer v3_easy.

Evalúa:
1. Token accuracy (val + test) en el subconjunto fácil
2. Exact match de la línea "Answer:" (por dominio)
3. Answer head regression accuracy (MAE, exact match numérico)
4. Entropía de cross-attention (¿rompe el colapso?)
5. Comparación con v2 en el MISMO subconjunto fácil

Uso:
    python evaluation/evaluate_v3_easy.py
"""

import os
import sys
import json
import re
import numpy as np

# XLA/GPU config
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_cuda_data_dir=/usr/local/cuda-12.8")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
parent = os.path.dirname(PROJECT_DIR)
if parent not in sys.path:
    sys.path.insert(0, parent)

import tensorflow as tf

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

from transformer_math_physics_tutor.data.subword_tokenizer import SubwordTokenizer
from transformer_math_physics_tutor.models.config import TransformerConfig
from transformer_math_physics_tutor.models.transformer_v3 import TransformerV3
from transformer_math_physics_tutor.inference.generate import generate_text


def extract_answer(text):
    """Extrae el valor después de 'Answer:' en la solución."""
    match = re.search(r'Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip().rstrip('.')
    return ""


def normalize_answer(answer):
    """Normaliza una respuesta para comparación."""
    answer = answer.strip().lower()
    answer = re.sub(r'[,$]', '', answer)
    answer = re.sub(r'\s+', ' ', answer)
    return answer


def parse_numeric(answer_str):
    """Intenta parsear un string como número."""
    cleaned = answer_str.strip().replace(',', '').replace('$', '').replace('%', '')
    cleaned = re.sub(r'\s*(kg|m|s|km|cm|mm|g|lb|oz|mph|kph|hours?|minutes?|seconds?|days?|years?|feet|inches|meters|liters|gallons|degrees?|°[CF]|kWh?|[JW]|joules?|watts?|newtons?|[Nn]|m/s\^?2?)\.?$', '', cleaned)
    cleaned = cleaned.strip()
    try:
        return float(cleaned)
    except ValueError:
        num_match = re.match(r'^-?[\d.]+', cleaned)
        if num_match:
            try:
                return float(num_match.group())
            except ValueError:
                return None
        return None


def evaluate_cross_attention(model, tokenizer, problems, config):
    """Evalúa si la cross-attention funciona (no es uniforme)."""
    print("\n" + "=" * 60)
    print("DIAGNÓSTICO: Cross-Attention Entropy")
    print("=" * 60)

    def pad_to(tokens, max_len, pad_id):
        if len(tokens) > max_len:
            return tokens[:max_len]
        return tokens + [pad_id] * (max_len - len(tokens))

    results = []

    for prob_text in problems[:10]:
        enc_tokens = tokenizer.encode(prob_text, add_special_tokens=True)
        n_real = len(enc_tokens)
        enc_padded = pad_to(enc_tokens, config.max_encoder_len, tokenizer.pad_token_id)
        enc_input = tf.constant([enc_padded], dtype=tf.int32)
        dec_input = tf.constant([[tokenizer.start_token_id]], dtype=tf.int32)

        with tf.device('/CPU:0'):
            enc_padding_mask = model.create_padding_mask(enc_input)
            dec_padding_mask = model.create_padding_mask(enc_input)
            look_ahead_mask = model.create_look_ahead_mask(tf.shape(dec_input)[1])
            dec_target_padding_mask = model.create_padding_mask(dec_input)
            combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

            enc_output = model.encoder(enc_input, training=False, mask=enc_padding_mask)
            dec_output, attn_weights = model.decoder(
                dec_input, enc_output, training=False,
                look_ahead_mask=combined_mask, padding_mask=dec_padding_mask
            )

        layer_entropies = {}
        for layer_name, weights in attn_weights.items():
            if 'block2' not in layer_name:
                continue
            w = weights.numpy()[0]
            real_w = w[:, :, :n_real]
            real_w = real_w / (real_w.sum(axis=-1, keepdims=True) + 1e-9)
            entropy = -np.sum(real_w * np.log(real_w + 1e-9), axis=-1)
            max_entropy = np.log(n_real)
            frac = np.mean(entropy) / max_entropy
            layer_entropies[layer_name] = float(frac)

        results.append(layer_entropies)

    avg_entropies = {}
    if results:
        for key in results[0]:
            vals = [r[key] for r in results]
            avg = np.mean(vals)
            avg_entropies[key] = float(avg)
            status = "⚠️ UNIFORME" if avg > 0.95 else "✅ SELECTIVA" if avg < 0.80 else "🟡 PARCIAL"
            print(f"  {key}: entropy = {avg:.3f} → {status}")

    return avg_entropies


def main():
    print("=" * 60)
    print("  EVALUACIÓN MODELO v3_easy")
    print("  (Answer Head + Decoder Masking + Easy Subset)")
    print("=" * 60)

    # --- Configuración ---
    checkpoint_dir = os.path.join(PROJECT_DIR, "checkpoints", "v3_easy")
    data_file = os.path.join(PROJECT_DIR, "data", "combined_easy.json")

    # --- Cargar modelo v3 ---
    print("\n1. Cargando modelo v3_easy...")
    config = TransformerConfig.load(os.path.join(checkpoint_dir, "config.json"))

    tok_path = os.path.join(checkpoint_dir, "sp_tokenizer.model")
    if not os.path.exists(tok_path):
        tok_path = os.path.join(PROJECT_DIR, "checkpoints", "v2_subword", "sp_tokenizer.model")
    tokenizer = SubwordTokenizer(tok_path)
    config.vocab_size = tokenizer.vocab_size

    with tf.device('/CPU:0'):
        model = TransformerV3(config, answer_scale=1000.0)
        dummy_enc = tf.zeros((1, config.max_encoder_len), dtype=tf.int32)
        dummy_dec = tf.zeros((1, config.max_decoder_len), dtype=tf.int32)
        _ = model((dummy_enc, dummy_dec), training=False)

        best_path = os.path.join(checkpoint_dir, "best_model.weights.h5")
        fallback = os.path.join(checkpoint_dir, "model_weights.weights.h5")
        weights = best_path if os.path.exists(best_path) else fallback
        model.load_weights(weights)

    print(f"   Modelo cargado: {model.count_params():,} parámetros ({os.path.basename(weights)})")

    # --- Cargar datos ---
    print("\n2. Cargando datos easy test...")
    with open(data_file, 'r') as f:
        all_data = json.load(f)

    test_data = [d for d in all_data if d.get('split') == 'test']
    val_data = [d for d in all_data if d.get('split') == 'val']
    print(f"   {len(test_data)} test, {len(val_data)} val")

    # --- Token accuracy ---
    print("\n3. Token accuracy (val + test)...")
    from transformer_math_physics_tutor.data.dataset_builder import create_datasets_v3_easy
    from transformer_math_physics_tutor.training.losses import loss_function, accuracy_function

    _, val_ds, test_ds, _ = create_datasets_v3_easy(
        tokenizer_model=tok_path,
        max_problem_len=config.max_encoder_len,
        max_solution_len=config.max_decoder_len,
        batch_size=32,
    )

    @tf.function
    def eval_step(inp, dec_inp, dec_target):
        output = model((inp, dec_inp), training=False)
        predictions = output[0]  # TransformerV3 returns (logits, answer_pred)
        loss = loss_function(dec_target, predictions, config.label_smoothing)
        acc = accuracy_function(dec_target, predictions)
        return loss, acc

    val_losses, val_accs = [], []
    for (inp, dec_inp), (dec_target, _) in val_ds:
        loss, acc = eval_step(inp, dec_inp, dec_target)
        val_losses.append(float(loss.numpy()))
        val_accs.append(float(acc.numpy()))

    test_losses, test_accs = [], []
    if test_ds:
        for (inp, dec_inp), (dec_target, _) in test_ds:
            loss, acc = eval_step(inp, dec_inp, dec_target)
            test_losses.append(float(loss.numpy()))
            test_accs.append(float(acc.numpy()))

    val_acc = np.mean(val_accs) if val_accs else 0
    val_loss = np.mean(val_losses) if val_losses else 0
    test_acc = np.mean(test_accs) if test_accs else 0
    test_loss = np.mean(test_losses) if test_losses else 0

    print(f"   Val  — Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    print(f"   Test — Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    # --- Answer head accuracy ---
    print("\n4. Answer head regression accuracy...")

    @tf.function
    def answer_eval_step(inp, dec_inp, answer_true):
        output = model((inp, dec_inp), training=False)
        answer_pred = output[1]  # (batch,)
        return answer_pred

    head_preds = []
    head_trues = []
    if test_ds:
        for (inp, dec_inp), (_, answer_val) in test_ds:
            preds = answer_eval_step(inp, dec_inp, answer_val)
            head_preds.extend((preds.numpy() * 1000.0).tolist())
            head_trues.extend(answer_val.numpy().tolist())

    if head_preds:
        head_preds = np.array(head_preds)
        head_trues = np.array(head_trues)
        head_mae = np.mean(np.abs(head_preds - head_trues))
        # Exact match numérico (con tolerancia de ±0.5 para redondeo)
        head_rounded = np.round(head_preds)
        head_exact = np.sum(np.abs(head_rounded - head_trues) < 0.5)
        head_exact_pct = head_exact / len(head_trues) * 100
        print(f"   Answer head MAE: {head_mae:.1f}")
        print(f"   Answer head exact (±0.5): {head_exact}/{len(head_trues)} = {head_exact_pct:.1f}%")
    else:
        head_mae = 0
        head_exact_pct = 0

    # --- Exact match (generación completa) ---
    print("\n5. Exact match (generación seq2seq)...")

    max_enc = config.max_encoder_len
    max_dec = config.max_decoder_len
    pad_id = tokenizer.pad_token_id

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, max_enc], dtype=tf.int32),
        tf.TensorSpec(shape=[1, max_dec], dtype=tf.int32),
    ])
    def predict_step(enc_input, dec_input):
        output = model((enc_input, dec_input), training=False)
        return output[0]  # Solo logits

    def generate_text_fast(problem_text, max_length=256):
        """Generación autoregresiva con @tf.function (Blackwell compat)."""
        enc_tokens = tokenizer.encode(problem_text, add_special_tokens=True)
        if len(enc_tokens) > max_enc:
            enc_tokens = enc_tokens[:max_enc]
        else:
            enc_tokens = enc_tokens + [pad_id] * (max_enc - len(enc_tokens))
        enc_input = tf.constant([enc_tokens], dtype=tf.int32)

        decoder_tokens = [tokenizer.start_token_id]
        generated_tokens = []

        for step in range(max_length):
            dec_padded = decoder_tokens + [pad_id] * (max_dec - len(decoder_tokens))
            dec_input = tf.constant([dec_padded[:max_dec]], dtype=tf.int32)
            predictions = predict_step(enc_input, dec_input)

            pos = min(len(decoder_tokens) - 1, max_dec - 1)
            logits = predictions[0, pos, :].numpy()
            predicted_id = int(np.argmax(logits))

            if predicted_id == tokenizer.end_token_id:
                break

            generated_tokens.append(predicted_id)
            decoder_tokens.append(predicted_id)

            if len(decoder_tokens) >= max_dec:
                break

            if len(generated_tokens) > 20:
                n = 10
                if len(generated_tokens) >= n * 2:
                    last_ngram = generated_tokens[-n:]
                    for j in range(len(generated_tokens) - n * 2,
                                   max(0, len(generated_tokens) - n * 5) - 1, -1):
                        if generated_tokens[j:j + n] == last_ngram:
                            return tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return tokenizer.decode(generated_tokens, skip_special_tokens=True)

    n_eval = min(100, len(test_data))
    eval_data = test_data[:n_eval]

    correct = 0
    total = 0
    correct_numeric = 0
    total_numeric = 0
    domain_stats = {}
    examples = []

    print(f"   Generando predicciones para {n_eval} problemas...")
    for i, item in enumerate(eval_data):
        if (i + 1) % 20 == 0:
            print(f"   Evaluando {i+1}/{n_eval}...")

        problem = item['problem']
        ref_solution = item['solution']
        domain = item.get('domain', 'unknown')
        true_answer_val = item.get('answer_value', None)

        ref_answer = extract_answer(ref_solution)
        if not ref_answer:
            continue

        pred_solution = generate_text_fast(problem, max_length=config.max_decoder_len)
        pred_answer = extract_answer(pred_solution)

        # Exact match textual
        is_correct = normalize_answer(pred_answer) == normalize_answer(ref_answer)

        # Exact match numérico (más tolerante)
        is_numeric_correct = False
        if true_answer_val is not None:
            pred_num = parse_numeric(pred_answer)
            if pred_num is not None:
                is_numeric_correct = abs(pred_num - true_answer_val) < 0.5
                total_numeric += 1
                if is_numeric_correct:
                    correct_numeric += 1

        if domain not in domain_stats:
            domain_stats[domain] = {'correct': 0, 'total': 0, 'numeric_correct': 0, 'numeric_total': 0}
        domain_stats[domain]['total'] += 1
        total += 1

        if is_correct:
            correct += 1
            domain_stats[domain]['correct'] += 1
        if is_numeric_correct:
            domain_stats[domain]['numeric_correct'] += 1
        if true_answer_val is not None:
            domain_stats[domain]['numeric_total'] += 1

        examples.append({
            'problem': problem[:120],
            'domain': domain,
            'ref_answer': ref_answer,
            'pred_answer': pred_answer,
            'pred_solution': pred_solution[:250],
            'correct': is_correct,
            'numeric_correct': is_numeric_correct,
            'true_value': true_answer_val,
        })

    em_pct = correct / max(total, 1) * 100
    num_em_pct = correct_numeric / max(total_numeric, 1) * 100

    print(f"\n   Exact Match (textual): {correct}/{total} = {em_pct:.1f}%")
    print(f"   Exact Match (numérico ±0.5): {correct_numeric}/{total_numeric} = {num_em_pct:.1f}%")
    for domain, stats in sorted(domain_stats.items()):
        d_pct = stats['correct'] / max(stats['total'], 1) * 100
        n_pct = stats['numeric_correct'] / max(stats['numeric_total'], 1) * 100
        print(f"   {domain}: text={stats['correct']}/{stats['total']} ({d_pct:.1f}%), "
              f"numeric={stats['numeric_correct']}/{stats['numeric_total']} ({n_pct:.1f}%)")

    # --- Cross-attention entropy ---
    print("\n6. Diagnóstico de cross-attention...")
    sample_problems = [d['problem'] for d in eval_data[:10]]
    attn_entropies = evaluate_cross_attention(model, tokenizer, sample_problems, config)

    # --- Guardar reporte ---
    report = {
        'version': 'v3_easy',
        'description': 'Answer Head + Decoder Masking (35%) + Easy Subset',
        'tokenizer': 'SubwordTokenizer (BPE, 4000) — same as v2',
        'token_accuracy': {
            'val_acc': float(val_acc),
            'val_loss': float(val_loss),
            'test_acc': float(test_acc),
            'test_loss': float(test_loss),
        },
        'answer_head': {
            'mae': float(head_mae),
            'exact_pct': float(head_exact_pct),
        },
        'exact_match': {
            'textual_correct': correct,
            'textual_total': total,
            'textual_pct': em_pct,
            'numeric_correct': correct_numeric,
            'numeric_total': total_numeric,
            'numeric_pct': num_em_pct,
            'by_domain': domain_stats,
        },
        'cross_attention_entropy': attn_entropies,
        'examples': examples[:30],
    }

    report_path = os.path.join(checkpoint_dir, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n   Reporte guardado en {report_path}")

    # --- Mostrar ejemplos ---
    print("\n" + "=" * 60)
    print("EJEMPLOS DE PREDICCIONES")
    print("=" * 60)
    for ex in examples[:20]:
        icon = "✅" if ex['correct'] else ("🔢" if ex.get('numeric_correct') else "❌")
        print(f"\n{icon} [{ex['domain']}] {ex['problem'][:90]}")
        print(f"   Esperado: {ex['ref_answer']} (valor: {ex.get('true_value', '?')})")
        print(f"   Predicho: {ex['pred_answer']}")
        if not ex['correct']:
            print(f"   Solución: {ex['pred_solution'][:180]}")

    # --- Resumen ---
    print("\n" + "=" * 60)
    print("RESUMEN v3_easy")
    print("=" * 60)
    print(f"  Token Accuracy — Val: {val_acc:.1%}, Test: {test_acc:.1%}")
    print(f"  Exact Match (textual)  — {correct}/{total} = {em_pct:.1f}%")
    print(f"  Exact Match (numérico) — {correct_numeric}/{total_numeric} = {num_em_pct:.1f}%")
    print(f"  Answer Head MAE — {head_mae:.1f}")
    for k, v in attn_entropies.items():
        print(f"  {k} entropy: {v:.3f}")

    # --- Comparar con v2 ---
    v2_report_path = os.path.join(PROJECT_DIR, 'checkpoints', 'v2_subword', 'evaluation_report.json')
    if os.path.exists(v2_report_path):
        with open(v2_report_path) as f:
            v2 = json.load(f)
        print(f"\n  --- Comparación v2 vs v3_easy ---")
        v2_ta = v2.get('token_accuracy', {})
        v2_em = v2.get('exact_match', {})
        v2_attn = v2.get('cross_attention_entropy', {})
        print(f"  Token Acc — v2: {v2_ta.get('test_acc', 0):.1%} | v3: {test_acc:.1%}")
        print(f"  Exact Match — v2: {v2_em.get('pct', 0):.1f}% | v3: {em_pct:.1f}%")
        for key in v2_attn:
            v2_val = v2_attn[key]
            v3_val = attn_entropies.get(key, 0)
            print(f"  {key} — v2: {v2_val:.3f} | v3: {v3_val:.3f}")


if __name__ == "__main__":
    main()
