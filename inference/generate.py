"""
generate.py — Decodificación autoregresiva para generación de texto.

Implementa la función de generación de texto que usa el Transformer
entrenado para resolver problemas de forma autoregresiva.

El proceso de generación:
1. Tokeniza el problema de entrada y paddea a la longitud del encoder
2. Inicia el decoder con token <START>
3. En cada paso, predice el siguiente token (greedy, sampling, o top-k)
4. Agrega el token predicho y repite
5. Se detiene al generar <END>, al repetir n-grams, o alcanzar max_length

Uso:
    from transformer_math_physics_tutor.inference.generate import generate_text
    answer = generate_text(model, tokenizer, "Solve: 2x + 3 = 7")
"""

import tensorflow as tf
import numpy as np
from typing import Optional, List

from transformer_math_physics_tutor.data.subword_tokenizer import SubwordTokenizer


def _pad_to_length(tokens: List[int], max_len: int, pad_id: int) -> List[int]:
    """Paddea o trunca una secuencia a max_len."""
    if len(tokens) > max_len:
        return tokens[:max_len - 1] + [tokens[-1]]  # Mantener END
    return tokens + [pad_id] * (max_len - len(tokens))


def _detect_ngram_repeat(tokens: List[int], n: int = 8) -> bool:
    """Detecta si los últimos n tokens se repiten en la secuencia anterior."""
    if len(tokens) < n * 2:
        return False
    last_ngram = tokens[-n:]
    # Buscar este n-gram en tokens anteriores
    for i in range(len(tokens) - n * 2, max(0, len(tokens) - n * 5) - 1, -1):
        if tokens[i:i + n] == last_ngram:
            return True
    return False


def generate_text(
    model: tf.keras.Model,
    tokenizer: SubwordTokenizer,
    input_text: str,
    max_length: int = 300,
    temperature: float = 0.3,
    top_k: int = 10,
    repetition_penalty: float = 1.2,
    verbose: bool = False
) -> str:
    """
    Genera texto usando decodificación autoregresiva con muestreo controlado.

    El modelo recibe el problema tokenizado como encoder input, paddeado
    a la misma longitud usada en entrenamiento para consistencia.

    Args:
        model: Modelo Transformer entrenado.
        tokenizer: Instancia de SubwordTokenizer.
        input_text: Texto del problema a resolver.
        max_length: Longitud máxima de la respuesta generada.
        temperature: Factor de temperatura (0.0 = greedy, 0.3 = conservador).
        top_k: Solo considerar los top-k tokens más probables (0 = todos).
        repetition_penalty: Penalización para tokens ya generados (>1.0 reduce repetición).
        verbose: Si True, imprime cada token generado.

    Returns:
        Texto de la respuesta generada.
    """
    # Paso 1: Tokenizar y paddear el encoder input
    encoder_input = tokenizer.encode(input_text, add_special_tokens=True)

    # Obtener max_encoder_len del modelo
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'pos_encoding'):
        max_enc_len = model.encoder.pos_encoding.shape[1]
    else:
        max_enc_len = 200  # default

    # Paddear a la misma longitud que durante entrenamiento
    encoder_input = _pad_to_length(encoder_input, max_enc_len, tokenizer.pad_token_id)
    encoder_input = tf.constant([encoder_input], dtype=tf.int32)  # (1, max_enc_len)

    # Paso 2: Inicializar decoder
    decoder_tokens = [tokenizer.start_token_id]

    # Paso 3: Generación autoregresiva
    generated_tokens = []

    for i in range(max_length):
        # Preparar decoder input (sin paddear — crece dinámicamente)
        decoder_input = tf.constant([decoder_tokens], dtype=tf.int32)

        # Forward pass
        output = model((encoder_input, decoder_input), training=False)
        # TransformerV3 retorna tupla (logits, answer_pred)
        # Transformer base retorna solo logits
        if isinstance(output, tuple):
            predictions = output[0]
        else:
            predictions = output
        # Tomar logits del último token: (1, vocab_size)
        logits = predictions[0, -1, :]  # (vocab_size,)

        # Aplicar repetition penalty a tokens ya generados
        if repetition_penalty > 1.0 and generated_tokens:
            seen_tokens = set(generated_tokens)
            logits_np = logits.numpy()
            for tok_id in seen_tokens:
                if logits_np[tok_id] > 0:
                    logits_np[tok_id] /= repetition_penalty
                else:
                    logits_np[tok_id] *= repetition_penalty
            logits = tf.constant(logits_np)

        # Seleccionar siguiente token
        if temperature <= 0.0:
            # Greedy
            predicted_id = int(tf.argmax(logits).numpy())
        else:
            # Aplicar temperatura
            scaled_logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                values, indices = tf.math.top_k(scaled_logits, k=min(top_k, len(scaled_logits)))
                # Crear máscara: -inf para todo excepto top-k
                mask = tf.fill(scaled_logits.shape, float('-inf'))
                mask = tf.tensor_scatter_nd_update(
                    mask, tf.expand_dims(indices, 1), values
                )
                scaled_logits = mask

            # Sampling
            predicted_id = int(tf.random.categorical(
                tf.expand_dims(scaled_logits, 0), num_samples=1
            ).numpy()[0][0])

        if verbose:
            token_str = tokenizer.decode([predicted_id], skip_special_tokens=False)
            print(f"  Step {i+1}: id={predicted_id}, char='{token_str}'")

        # Token de fin
        if predicted_id == tokenizer.end_token_id:
            if verbose:
                print("  → <END> generado.")
            break

        # Detección de repetición n-gram
        generated_tokens.append(predicted_id)
        if len(generated_tokens) > 20 and _detect_ngram_repeat(generated_tokens, n=10):
            if verbose:
                print("  → Repetición n-gram detectada.")
            break

        # Expandir decoder
        decoder_tokens.append(predicted_id)

    # Decodificar
    result_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return result_text


def generate_text_beam_search(
    model: tf.keras.Model,
    tokenizer: SubwordTokenizer,
    input_text: str,
    max_length: int = 150,
    beam_width: int = 3
) -> str:
    """
    Genera texto usando beam search (búsqueda en haz).

    Mantiene los top-k (beam_width) candidatos en cada paso,
    lo que generalmente produce mejores resultados que greedy
    pero es más lento.

    Args:
        model: Modelo Transformer entrenado.
        tokenizer: Instancia de SubwordTokenizer.
        input_text: Texto del problema a resolver.
        max_length: Longitud máxima de la respuesta.
        beam_width: Ancho del haz (número de candidatos simultáneos).

    Returns:
        Texto de la mejor respuesta encontrada.
    """
    # Tokenizar input
    encoder_input = tokenizer.encode(input_text, add_special_tokens=True)

    # Truncar si excede la longitud máxima del encoder
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'pos_encoding'):
        max_enc_len = model.encoder.pos_encoding.shape[1]
        if len(encoder_input) > max_enc_len:
            encoder_input = encoder_input[:max_enc_len - 1] + [tokenizer.end_token_id]

    encoder_input = tf.expand_dims(encoder_input, 0)

    # Inicializar beams: cada beam es (secuencia, score_acumulado)
    # Comenzamos con un solo beam que tiene solo <START>
    beams = [([tokenizer.start_token_id], 0.0)]

    completed_beams = []

    for step in range(max_length):
        all_candidates = []

        for seq, score in beams:
            # Si la secuencia ya terminó, pasarla directamente
            if seq[-1] == tokenizer.end_token_id:
                completed_beams.append((seq, score))
                continue

            # Preparar decoder input
            decoder_input = tf.expand_dims(seq, 0)
            decoder_input = tf.cast(decoder_input, tf.int32)

            # Forward pass
            output = model((encoder_input, decoder_input), training=False)
            if isinstance(output, tuple):
                predictions = output[0]
            else:
                predictions = output
            predictions = predictions[:, -1, :]  # (1, vocab_size)

            # Log-probabilidades
            log_probs = tf.nn.log_softmax(predictions).numpy()[0]

            # Tomar top-k tokens
            top_k_indices = log_probs.argsort()[-beam_width:][::-1]

            for idx in top_k_indices:
                new_seq = seq + [int(idx)]
                new_score = score + log_probs[idx]
                all_candidates.append((new_seq, new_score))

        if not all_candidates:
            break

        # Ordenar por score normalizado por longitud y mantener los mejores
        # Length normalization: score / len^alpha previene sesgo a secuencias cortas
        alpha = 0.6  # Factor de normalización por longitud
        all_candidates.sort(
            key=lambda x: x[1] / (len(x[0]) ** alpha),
            reverse=True
        )
        beams = all_candidates[:beam_width]

        # Si todos los beams activos han terminado, parar
        if all(seq[-1] == tokenizer.end_token_id for seq, _ in beams):
            completed_beams.extend(beams)
            break

    # Incluir beams no terminados
    completed_beams.extend(beams)

    # Tomar el mejor beam (con normalización por longitud)
    if completed_beams:
        alpha = 0.6
        best_seq, best_score = max(
            completed_beams,
            key=lambda x: x[1] / (len(x[0]) ** alpha)
        )
    else:
        best_seq = [tokenizer.start_token_id]

    # Decodificar
    result_text = tokenizer.decode(best_seq, skip_special_tokens=True)
    return result_text


if __name__ == "__main__":
    print("=" * 60)
    print("DEMO: Generación de Texto")
    print("=" * 60)
    print("Para usar generate_text, necesitas un modelo entrenado.")
    print("Ejemplo de uso:")
    print("  from transformer_math_physics_tutor.inference.generate import generate_text")
    print("  answer = generate_text(model, tokenizer, 'Solve: 2x + 3 = 7')")
    print("  print(answer)")
