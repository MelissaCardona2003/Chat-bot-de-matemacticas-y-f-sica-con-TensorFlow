"""
generate_v4.py — Decodificación autoregresiva para Pointer-Generator (V4).

Genera texto usando el TransformerV4 que produce distribuciones mixtas
(vocab + copy). El generador usa las probabilidades mezcladas directamente
en vez de logits puros, lo que permite copiar tokens del input.

Uso:
    from transformer_math_physics_tutor.inference.generate_v4 import generate_text_v4
    answer = generate_text_v4(model, tokenizer, "Solve: 2x + 3 = 7")
"""

import tensorflow as tf
import numpy as np
from typing import Optional, List

from transformer_math_physics_tutor.data.subword_tokenizer import SubwordTokenizer


def _pad_to_length(tokens: List[int], max_len: int, pad_id: int) -> List[int]:
    """Paddea o trunca una secuencia a max_len."""
    if len(tokens) > max_len:
        return tokens[:max_len - 1] + [tokens[-1]]
    return tokens + [pad_id] * (max_len - len(tokens))


def _detect_ngram_repeat(tokens: List[int], n: int = 8) -> bool:
    """Detecta si los últimos n tokens se repiten en la secuencia anterior."""
    if len(tokens) < n * 2:
        return False
    last_ngram = tokens[-n:]
    for i in range(len(tokens) - n * 2, max(0, len(tokens) - n * 5) - 1, -1):
        if tokens[i:i + n] == last_ngram:
            return True
    return False


def generate_text_v4(
    model: tf.keras.Model,
    tokenizer: SubwordTokenizer,
    input_text: str,
    max_length: int = 300,
    temperature: float = 0.3,
    top_k: int = 10,
    repetition_penalty: float = 1.2,
    copy_boost: float = 1.0,
    verbose: bool = False
) -> str:
    """
    Genera texto usando el TransformerV4 (Pointer-Generator).

    A diferencia de generate_text (V3), esta función:
    - Recibe PROBABILIDADES mezcladas (no logits) del modelo
    - p_gen controla automáticamente cuánto copiar vs generar
    - Opcionalmente puede amplificar la copia con copy_boost

    Args:
        model: TransformerV4 con copy mechanism.
        tokenizer: Instancia de SubwordTokenizer.
        input_text: Texto del problema a resolver.
        max_length: Longitud máxima de la respuesta generada.
        temperature: Factor de temperatura (0.0 = greedy, 0.3 = conservador).
        top_k: Solo considerar los top-k tokens más probables.
        repetition_penalty: Penalización para tokens ya generados.
        copy_boost: Factor para amplificar la porción de copia.
                    1.0 = sin cambio, 2.0 = duplicar peso de copia.
                    Útil en inference para forzar más copia de números.
        verbose: Si True, imprime cada token generado con p_gen info.

    Returns:
        Texto de la respuesta generada.
    """
    # Paso 1: Tokenizar y paddear encoder input
    encoder_input = tokenizer.encode(input_text, add_special_tokens=True)

    if hasattr(model, 'encoder') and hasattr(model.encoder, 'pos_encoding'):
        max_enc_len = model.encoder.pos_encoding.shape[1]
    else:
        max_enc_len = 200

    encoder_input = _pad_to_length(encoder_input, max_enc_len, tokenizer.pad_token_id)
    encoder_input_tf = tf.constant([encoder_input], dtype=tf.int32)

    # Paso 2: Inicializar decoder
    decoder_tokens = [tokenizer.start_token_id]
    generated_tokens = []

    # Paso 3: Generación autoregresiva
    for i in range(max_length):
        decoder_input = tf.constant([decoder_tokens], dtype=tf.int32)

        # Forward pass — TransformerV4 retorna (final_probs, answer_pred, p_gen_mean)
        output = model((encoder_input_tf, decoder_input), training=False)

        if isinstance(output, tuple) and len(output) >= 3:
            final_probs = output[0]   # (1, seq_len, vocab_size)
            p_gen_scalar = output[2]  # p_gen_mean escalar
        elif isinstance(output, tuple) and len(output) == 2:
            final_probs = output[0]
            p_gen_scalar = None
        else:
            final_probs = output
            p_gen_scalar = None

        # Tomar probabilidades del último token
        probs = final_probs[0, -1, :]  # (vocab_size,)
        probs_np = probs.numpy()

        # Aplicar repetition penalty
        if repetition_penalty > 1.0 and generated_tokens:
            seen_tokens = set(generated_tokens)
            for tok_id in seen_tokens:
                if probs_np[tok_id] > 0:
                    probs_np[tok_id] /= repetition_penalty

        # Renormalizar tras penalty
        probs_sum = probs_np.sum()
        if probs_sum > 0:
            probs_np /= probs_sum

        # Seleccionar siguiente token
        if temperature <= 0.0:
            # Greedy
            predicted_id = int(np.argmax(probs_np))
        else:
            # Aplicar temperatura a probabilidades
            # Convertir a logits → dividir por temp → softmax
            log_probs = np.log(np.maximum(probs_np, 1e-12))
            scaled_logits = log_probs / temperature

            # Top-k filtering
            if top_k > 0 and top_k < len(scaled_logits):
                top_k_indices = np.argsort(scaled_logits)[-top_k:]
                mask = np.full(len(scaled_logits), float('-inf'))
                mask[top_k_indices] = scaled_logits[top_k_indices]
                scaled_logits = mask

            # Softmax
            scaled_logits -= np.max(scaled_logits)  # estabilidad
            exp_logits = np.exp(scaled_logits)
            sampling_probs = exp_logits / exp_logits.sum()

            # Sampling
            predicted_id = int(np.random.choice(len(sampling_probs), p=sampling_probs))

        if verbose:
            token_str = tokenizer.decode([predicted_id], skip_special_tokens=False)
            p_gen_info = f", p_gen={float(p_gen_scalar):.3f}" if p_gen_scalar is not None else ""
            # Check si el token está en el input (fue copiado)
            is_copy = predicted_id in encoder_input
            copy_tag = " [COPY]" if is_copy else ""
            print(f"  Step {i+1}: id={predicted_id}, token='{token_str}'"
                  f"{p_gen_info}{copy_tag}")

        # Token de fin
        if predicted_id == tokenizer.end_token_id:
            if verbose:
                print("  → <END> generado.")
            break

        # Detección de repetición
        generated_tokens.append(predicted_id)
        if len(generated_tokens) > 20 and _detect_ngram_repeat(generated_tokens, n=10):
            if verbose:
                print("  → Repetición n-gram detectada.")
            break

        decoder_tokens.append(predicted_id)

    # Decodificar
    result_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return result_text


def generate_text_v4_beam_search(
    model: tf.keras.Model,
    tokenizer: SubwordTokenizer,
    input_text: str,
    max_length: int = 150,
    beam_width: int = 3
) -> str:
    """
    Genera texto usando beam search con TransformerV4.

    Args:
        model: TransformerV4 con copy mechanism.
        tokenizer: SubwordTokenizer.
        input_text: Texto del problema.
        max_length: Longitud máxima.
        beam_width: Ancho del haz.

    Returns:
        Mejor respuesta encontrada.
    """
    # Tokenizar input
    encoder_input = tokenizer.encode(input_text, add_special_tokens=True)

    if hasattr(model, 'encoder') and hasattr(model.encoder, 'pos_encoding'):
        max_enc_len = model.encoder.pos_encoding.shape[1]
    else:
        max_enc_len = 200

    encoder_input = _pad_to_length(encoder_input, max_enc_len, tokenizer.pad_token_id)
    encoder_input_tf = tf.constant([encoder_input], dtype=tf.int32)

    # Beams: (secuencia, score)
    beams = [([tokenizer.start_token_id], 0.0)]
    completed_beams = []

    for step in range(max_length):
        all_candidates = []

        for seq, score in beams:
            if seq[-1] == tokenizer.end_token_id:
                completed_beams.append((seq, score))
                continue

            decoder_input = tf.constant([seq], dtype=tf.int32)

            output = model((encoder_input_tf, decoder_input), training=False)
            if isinstance(output, tuple):
                final_probs = output[0]
            else:
                final_probs = output

            probs = final_probs[0, -1, :]  # (vocab_size,)

            # Log-probabilidades
            log_probs = tf.math.log(tf.maximum(probs, 1e-12)).numpy()

            # Top-k
            top_k_indices = log_probs.argsort()[-beam_width:][::-1]

            for idx in top_k_indices:
                new_seq = seq + [int(idx)]
                new_score = score + log_probs[idx]
                all_candidates.append((new_seq, new_score))

        if not all_candidates:
            break

        alpha = 0.6
        all_candidates.sort(
            key=lambda x: x[1] / (len(x[0]) ** alpha),
            reverse=True
        )
        beams = all_candidates[:beam_width]

        if all(seq[-1] == tokenizer.end_token_id for seq, _ in beams):
            completed_beams.extend(beams)
            break

    completed_beams.extend(beams)

    if completed_beams:
        alpha = 0.6
        best_seq, _ = max(
            completed_beams,
            key=lambda x: x[1] / (len(x[0]) ** alpha)
        )
    else:
        best_seq = [tokenizer.start_token_id]

    result_text = tokenizer.decode(best_seq, skip_special_tokens=True)
    return result_text


if __name__ == "__main__":
    print("=" * 60)
    print("DEMO: Generación V4 (Pointer-Generator)")
    print("=" * 60)
    print("Para usar generate_text_v4, necesitas un TransformerV4 entrenado.")
    print("Ejemplo:")
    print("  from transformer_math_physics_tutor.inference.generate_v4 import generate_text_v4")
    print("  answer = generate_text_v4(model, tokenizer, 'Solve: 2x + 3 = 7')")
