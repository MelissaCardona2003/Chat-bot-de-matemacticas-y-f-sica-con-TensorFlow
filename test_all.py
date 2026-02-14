"""
Test completo de todos los módulos del Transformer Math/Physics Tutor.
"""
import sys
import os
import json
import tempfile

# Asegurar que el directorio padre está en sys.path
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
if _parent not in sys.path:
    sys.path.insert(0, _parent)


# ============================================================
# TEST 1: preprocess_latex
# ============================================================
def test_preprocess_latex():
    print("=" * 60)
    print("TEST 1: data/preprocess_latex.py")
    print("=" * 60)
    from transformer_math_physics_tutor.data.preprocess_latex import clean_latex

    tests = [
        (r'\frac{1}{2} + \frac{3}{4}', '(1/2) + (3/4)'),
        (r'\textbf{hello} world', 'hello world'),
        ('$x^2 + y^2 = z^2$', 'x^2 + y^2 = z^2'),
        (r'\left( a + b \right)', '( a + b )'),
        (r'\sqrt{16}', 'sqrt(16)'),
    ]

    passed = 0
    for latex, expected in tests:
        result = clean_latex(latex)
        status = result.strip() == expected.strip()
        print(f"  {'PASS' if status else 'FAIL'}: clean_latex({repr(latex)})")
        if not status:
            print(f"         Expected: {repr(expected)}")
            print(f"         Got:      {repr(result)}")
        if status:
            passed += 1
    print(f"  Resultado: {passed}/{len(tests)} tests pasados\n")
    return passed == len(tests)


# ============================================================
# TEST 2: tokenizer
# ============================================================
def test_tokenizer():
    print("=" * 60)
    print("TEST 2: data/tokenizer.py")
    print("=" * 60)
    from transformer_math_physics_tutor.data.tokenizer import CharTokenizer

    tokenizer = CharTokenizer()

    texts = [
        "2 + 3 = 5",
        "x^2 + y = 10",
        "(1/2) + (3/4) = (5/4)",
        "solve for x: 2x + 1 = 5",
    ]
    tokenizer.build_vocab(texts)
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  Special tokens: PAD={tokenizer.pad_token_id}, START={tokenizer.start_token_id}, END={tokenizer.end_token_id}, UNK={tokenizer.unk_token_id}")

    test_text = "2 + 3 = 5"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"  Encode '{test_text}': {encoded}")
    print(f"  Decode back: '{decoded}'")

    # Check START and END tokens in encoded sequence
    enc_has_start = encoded[0] == tokenizer.start_token_id
    enc_has_end = encoded[-1] == tokenizer.end_token_id
    dec_pass = decoded == test_text

    print(f"  {'PASS' if enc_has_start else 'FAIL'}: Encoded starts with START")
    print(f"  {'PASS' if enc_has_end else 'FAIL'}: Encoded ends with END")
    print(f"  {'PASS' if dec_pass else 'FAIL'}: Decode matches original")

    # Test save/load
    vocab_path = os.path.join(tempfile.gettempdir(), "test_vocab_temp.json")
    tokenizer.save_vocab(vocab_path)

    tokenizer2 = CharTokenizer()
    tokenizer2.load_vocab(vocab_path)

    encoded2 = tokenizer2.encode(test_text)
    save_load_pass = encoded == encoded2
    print(f"  {'PASS' if save_load_pass else 'FAIL'}: Save/Load vocabulary")

    os.remove(vocab_path)

    all_pass = enc_has_start and enc_has_end and dec_pass and save_load_pass
    print(f"  Resultado: {'4/4' if all_pass else 'FAIL'} tests pasados\n")
    return all_pass


# ============================================================
# TEST 3: TransformerConfig
# ============================================================
def test_config():
    print("=" * 60)
    print("TEST 3: models/config.py")
    print("=" * 60)
    from transformer_math_physics_tutor.models.config import TransformerConfig

    config = TransformerConfig()
    print(f"  d_model={config.d_model}, num_heads={config.num_heads}")
    print(f"  num_layers={config.num_layers}, dff={config.dff}")
    print(f"  dropout={config.dropout_rate}")

    config_path = os.path.join(tempfile.gettempdir(), "test_config_temp.json")
    config.save(config_path)
    config2 = TransformerConfig.load(config_path)

    match = (config.d_model == config2.d_model and
             config.num_heads == config2.num_heads and
             config.num_layers == config2.num_layers)
    print(f"  {'PASS' if match else 'FAIL'}: Save/Load config")

    os.remove(config_path)
    print(f"  Resultado: {'1/1' if match else '0/1'} tests pasados\n")
    return match


# ============================================================
# TEST 4: Positional Encoding
# ============================================================
def test_positional_encoding():
    print("=" * 60)
    print("TEST 4: models/positional_encoding.py")
    print("=" * 60)
    import tensorflow as tf
    from transformer_math_physics_tutor.models.positional_encoding import positional_encoding

    d_model = 256
    max_len = 100

    pe = positional_encoding(max_len, d_model)
    shape_pass = pe.shape == (1, max_len, d_model)
    print(f"  Shape: {pe.shape} (expected (1, {max_len}, {d_model}))")
    print(f"  {'PASS' if shape_pass else 'FAIL'}: Shape correcta")

    range_pass = float(tf.reduce_max(tf.abs(pe))) <= 1.0
    print(f"  {'PASS' if range_pass else 'FAIL'}: Valores en rango [-1, 1]")

    all_pass = shape_pass and range_pass
    print(f"  Resultado: {'2/2' if all_pass else 'FAIL'} tests pasados\n")
    return all_pass


# ============================================================
# TEST 5: MultiHead Attention
# ============================================================
def test_multihead_attention():
    print("=" * 60)
    print("TEST 5: models/multihead_attention.py")
    print("=" * 60)
    import tensorflow as tf
    from transformer_math_physics_tutor.models.multihead_attention import MultiHeadAttention, scaled_dot_product_attention

    q = tf.random.normal((2, 8, 10, 32))
    k = tf.random.normal((2, 8, 10, 32))
    v = tf.random.normal((2, 8, 10, 32))

    output, weights = scaled_dot_product_attention(q, k, v, mask=None)
    sdp_shape = output.shape == (2, 8, 10, 32)
    w_shape = weights.shape == (2, 8, 10, 10)
    print(f"  scaled_dot_product output shape: {output.shape}")
    print(f"  {'PASS' if sdp_shape else 'FAIL'}: Output shape")
    print(f"  {'PASS' if w_shape else 'FAIL'}: Weights shape")

    mha = MultiHeadAttention(d_model=256, num_heads=8)
    x = tf.random.normal((2, 10, 256))
    out, attn_weights = mha(x, x, x, mask=None)
    mha_shape = out.shape == (2, 10, 256)
    print(f"  MultiHeadAttention output shape: {out.shape}")
    print(f"  {'PASS' if mha_shape else 'FAIL'}: MHA output shape")

    all_pass = sdp_shape and w_shape and mha_shape
    print(f"  Resultado: {'3/3' if all_pass else 'FAIL'} tests pasados\n")
    return all_pass


# ============================================================
# TEST 6: Encoder Layer
# ============================================================
def test_encoder_layer():
    print("=" * 60)
    print("TEST 6: models/encoder_layer.py")
    print("=" * 60)
    import tensorflow as tf
    from transformer_math_physics_tutor.models.encoder_layer import EncoderLayer

    enc_layer = EncoderLayer(d_model=256, num_heads=8, dff=1024, rate=0.1)
    x = tf.random.normal((2, 10, 256))
    out = enc_layer(x, training=False, mask=None)

    shape_pass = out.shape == (2, 10, 256)
    print(f"  Output shape: {out.shape}")
    print(f"  {'PASS' if shape_pass else 'FAIL'}: EncoderLayer shape")
    print(f"  Resultado: {'1/1' if shape_pass else '0/1'} tests pasados\n")
    return shape_pass


# ============================================================
# TEST 7: Decoder Layer
# ============================================================
def test_decoder_layer():
    print("=" * 60)
    print("TEST 7: models/decoder_layer.py")
    print("=" * 60)
    import tensorflow as tf
    from transformer_math_physics_tutor.models.decoder_layer import DecoderLayer

    dec_layer = DecoderLayer(d_model=256, num_heads=8, dff=1024, rate=0.1)
    x = tf.random.normal((2, 15, 256))
    enc_output = tf.random.normal((2, 10, 256))

    out, w1, w2 = dec_layer(x, enc_output, training=False,
                            look_ahead_mask=None, padding_mask=None)

    shape_pass = out.shape == (2, 15, 256)
    print(f"  Output shape: {out.shape}")
    print(f"  {'PASS' if shape_pass else 'FAIL'}: DecoderLayer shape")
    print(f"  Self-attn weights shape: {w1.shape}")
    print(f"  Cross-attn weights shape: {w2.shape}")
    print(f"  Resultado: {'1/1' if shape_pass else '0/1'} tests pasados\n")
    return shape_pass


# ============================================================
# TEST 8: Full Transformer Model
# ============================================================
def test_transformer():
    print("=" * 60)
    print("TEST 8: models/transformer.py (Full Transformer)")
    print("=" * 60)
    import tensorflow as tf
    from transformer_math_physics_tutor.models.config import TransformerConfig
    from transformer_math_physics_tutor.models.transformer import Transformer

    config = TransformerConfig(
        vocab_size=50,
        max_encoder_len=20,
        max_decoder_len=20,
        d_model=64,
        num_heads=4,
        num_layers=2,
        dff=128,
        dropout_rate=0.1
    )

    transformer = Transformer(config)

    # Model takes (inp, tar) as a tuple
    inp = tf.constant([[1, 5, 10, 15, 0, 0], [2, 8, 12, 0, 0, 0]])
    tar = tf.constant([[1, 3, 7, 0], [1, 4, 9, 11]])

    # call() returns only final_output (logits)
    predictions = transformer((inp, tar), training=False)

    pred_shape = predictions.shape == (2, 4, 50)
    print(f"  Predictions shape: {predictions.shape} (expected (2, 4, 50))")
    print(f"  {'PASS' if pred_shape else 'FAIL'}: Transformer output shape")

    # Check gradients
    with tf.GradientTape() as tape:
        preds = transformer((inp, tar), training=True)
        loss = tf.reduce_mean(preds)
    grads = tape.gradient(loss, transformer.trainable_variables)
    grad_pass = all(g is not None for g in grads)
    print(f"  Trainable variables: {len(transformer.trainable_variables)}")
    print(f"  {'PASS' if grad_pass else 'FAIL'}: Gradients calculados")

    all_pass = pred_shape and grad_pass
    print(f"  Resultado: {'2/2' if all_pass else 'FAIL'} tests pasados\n")
    return all_pass


# ============================================================
# TEST 9: Learning Rate Scheduler
# ============================================================
def test_scheduler():
    print("=" * 60)
    print("TEST 9: training/scheduler.py")
    print("=" * 60)
    import tensorflow as tf
    from transformer_math_physics_tutor.training.scheduler import CustomSchedule

    schedule = CustomSchedule(d_model=256, warmup_steps=4000)

    lr_1 = float(schedule(tf.constant(1, dtype=tf.float32)))
    lr_4000 = float(schedule(tf.constant(4000, dtype=tf.float32)))
    lr_10000 = float(schedule(tf.constant(10000, dtype=tf.float32)))

    print(f"  LR at step 1:     {lr_1:.8f}")
    print(f"  LR at step 4000:  {lr_4000:.8f}")
    print(f"  LR at step 10000: {lr_10000:.8f}")

    warmup_pass = lr_4000 > lr_1
    decay_pass = lr_4000 > lr_10000

    print(f"  {'PASS' if warmup_pass else 'FAIL'}: LR increases during warmup")
    print(f"  {'PASS' if decay_pass else 'FAIL'}: LR decreases after warmup")

    all_pass = warmup_pass and decay_pass
    print(f"  Resultado: {'2/2' if all_pass else 'FAIL'} tests pasados\n")
    return all_pass


# ============================================================
# TEST 10: Losses and Metrics
# ============================================================
def test_losses_metrics():
    print("=" * 60)
    print("TEST 10: training/losses.py & training/metrics.py")
    print("=" * 60)
    import tensorflow as tf
    from transformer_math_physics_tutor.training.losses import loss_function, accuracy_function
    from transformer_math_physics_tutor.training.metrics import exact_match_accuracy

    vocab_size = 50
    real = tf.constant([[1, 5, 10, 2, 0, 0], [1, 3, 7, 9, 2, 0]])
    pred = tf.random.normal((2, 6, vocab_size))

    loss = loss_function(real, pred)
    acc = accuracy_function(real, pred)

    loss_valid = float(loss) > 0
    acc_valid = 0.0 <= float(acc) <= 1.0

    print(f"  Loss: {float(loss):.4f}")
    print(f"  {'PASS' if loss_valid else 'FAIL'}: Loss > 0")
    print(f"  Accuracy: {float(acc):.4f}")
    print(f"  {'PASS' if acc_valid else 'FAIL'}: Accuracy in [0, 1]")

    pred_strs = ["x = 2", "y = 3", "z = 1"]
    true_strs = ["x = 2", "y = 4", "z = 1"]
    em = exact_match_accuracy(pred_strs, true_strs)
    em_pass = abs(em - 2/3) < 0.01
    print(f"  Exact Match: {em:.4f} (expected ~0.6667)")
    print(f"  {'PASS' if em_pass else 'FAIL'}: Exact match accuracy")

    all_pass = loss_valid and acc_valid and em_pass
    print(f"  Resultado: {'3/3' if all_pass else 'FAIL'} tests pasados\n")
    return all_pass


# ============================================================
# TEST 11: Dataset Builder
# ============================================================
def test_dataset_builder():
    print("=" * 60)
    print("TEST 11: data/dataset_builder.py")
    print("=" * 60)
    import tensorflow as tf
    from transformer_math_physics_tutor.data.tokenizer import CharTokenizer
    from transformer_math_physics_tutor.data.dataset_builder import create_dataset

    # Create a temp JSON file with problems
    problems_data = [
        {"problem": "2 + 3 = ?", "solution": "5"},
        {"problem": "5 * 2 = ?", "solution": "10"},
        {"problem": "10 - 7 = ?", "solution": "3"},
        {"problem": "8 / 4 = ?", "solution": "2"},
    ]
    temp_json = os.path.join(tempfile.gettempdir(), "test_problems.json")
    with open(temp_json, "w", encoding="utf-8") as f:
        json.dump(problems_data, f)

    # Build tokenizer
    tokenizer = CharTokenizer()
    all_texts = [p["problem"] for p in problems_data] + [p["solution"] for p in problems_data]
    tokenizer.build_vocab(all_texts)

    dataset = create_dataset(
        data_file=temp_json,
        tokenizer=tokenizer,
        max_problem_len=20,
        max_solution_len=10,
        batch_size=2
    )

    # Check one batch
    for (enc_inp, dec_inp), dec_target in dataset.take(1):
        shape_ok = len(enc_inp.shape) == 2 and len(dec_inp.shape) == 2 and len(dec_target.shape) == 2
        print(f"  Encoder input shape: {enc_inp.shape}")
        print(f"  Decoder input shape: {dec_inp.shape}")
        print(f"  Decoder target shape: {dec_target.shape}")
        print(f"  {'PASS' if shape_ok else 'FAIL'}: Dataset shapes correctas")

        start_ok = int(dec_inp[0, 0]) == tokenizer.start_token_id
        print(f"  {'PASS' if start_ok else 'FAIL'}: Decoder input starts with START token")

        os.remove(temp_json)
        all_pass = shape_ok and start_ok
        print(f"  Resultado: {'2/2' if all_pass else 'FAIL'} tests pasados\n")
        return all_pass

    os.remove(temp_json)
    print("  FAIL: No batches in dataset\n")
    return False


# ============================================================
# TEST 12: Inference (generate)
# ============================================================
def test_inference():
    print("=" * 60)
    print("TEST 12: inference/generate.py")
    print("=" * 60)
    import tensorflow as tf
    from transformer_math_physics_tutor.models.config import TransformerConfig
    from transformer_math_physics_tutor.models.transformer import Transformer
    from transformer_math_physics_tutor.data.tokenizer import CharTokenizer
    from transformer_math_physics_tutor.inference.generate import generate_text

    tokenizer = CharTokenizer()
    texts = ["2 + 3 = ?", "5", "x + 1 = 3", "2", "0123456789+-*/=?() xyz"]
    tokenizer.build_vocab(texts)

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        max_encoder_len=20,
        max_decoder_len=15,
        d_model=64,
        num_heads=4,
        num_layers=2,
        dff=128,
        dropout_rate=0.1
    )

    transformer = Transformer(config)

    # generate_text(model, tokenizer, input_text, max_length=...)
    result = generate_text(transformer, tokenizer, "2 + 3 = ?", max_length=15)

    gen_pass = isinstance(result, str)
    print(f"  Input: '2 + 3 = ?'")
    print(f"  Generated: '{result}'")
    print(f"  {'PASS' if gen_pass else 'FAIL'}: generate_text produces string output")
    print(f"  Resultado: {'1/1' if gen_pass else '0/1'} tests pasados\n")
    return gen_pass


# ============================================================
# TEST 13: Mini Training Loop
# ============================================================
def test_mini_training():
    print("=" * 60)
    print("TEST 13: Mini Training Loop (3 steps)")
    print("=" * 60)
    import tensorflow as tf
    from transformer_math_physics_tutor.models.config import TransformerConfig
    from transformer_math_physics_tutor.models.transformer import Transformer
    from transformer_math_physics_tutor.data.tokenizer import CharTokenizer
    from transformer_math_physics_tutor.data.dataset_builder import create_dataset
    from transformer_math_physics_tutor.training.scheduler import CustomSchedule
    from transformer_math_physics_tutor.training.losses import loss_function, accuracy_function

    # Create temp data file
    problems_data = [
        {"problem": "2 + 3 = ?", "solution": "5"},
        {"problem": "5 * 2 = ?", "solution": "10"},
        {"problem": "10 - 7 = ?", "solution": "3"},
        {"problem": "8 / 4 = ?", "solution": "2"},
        {"problem": "1 + 1 = ?", "solution": "2"},
        {"problem": "3 * 3 = ?", "solution": "9"},
        {"problem": "6 - 2 = ?", "solution": "4"},
        {"problem": "9 / 3 = ?", "solution": "3"},
    ]
    temp_json = os.path.join(tempfile.gettempdir(), "test_train_problems.json")
    with open(temp_json, "w", encoding="utf-8") as f:
        json.dump(problems_data, f)

    tokenizer = CharTokenizer()
    all_texts = [p["problem"] for p in problems_data] + [p["solution"] for p in problems_data]
    tokenizer.build_vocab(all_texts)

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        max_encoder_len=20,
        max_decoder_len=10,
        d_model=64,
        num_heads=4,
        num_layers=2,
        dff=128,
        dropout_rate=0.1
    )

    model = Transformer(config)

    learning_rate = CustomSchedule(config.d_model, warmup_steps=100)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    dataset = create_dataset(
        data_file=temp_json,
        tokenizer=tokenizer,
        max_problem_len=20,
        max_solution_len=10,
        batch_size=4
    )

    losses = []
    for step, ((enc_inp, dec_inp), dec_target) in enumerate(dataset.take(3)):
        with tf.GradientTape() as tape:
            predictions = model((enc_inp, dec_inp), training=True)
            loss = loss_function(dec_target, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        acc = accuracy_function(dec_target, predictions)
        losses.append(float(loss))
        print(f"  Step {step+1}: loss={float(loss):.4f}, accuracy={float(acc):.4f}")

    os.remove(temp_json)

    train_pass = len(losses) > 0 and all(l > 0 for l in losses)
    print(f"  {'PASS' if train_pass else 'FAIL'}: Training loop completo sin errores")
    print(f"  Resultado: {'1/1' if train_pass else '0/1'} tests pasados\n")
    return train_pass


# ============================================================
# TEST 14: Physics Problems JSON
# ============================================================
def test_physics_json():
    print("=" * 60)
    print("TEST 14: data/physics_problems.json")
    print("=" * 60)

    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "physics_problems.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    count = len(data)
    has_fields = all("problem" in item and "solution" in item for item in data)

    print(f"  Number of problems: {count}")
    print(f"  {'PASS' if count >= 10 else 'FAIL'}: At least 10 problems")
    print(f"  {'PASS' if has_fields else 'FAIL'}: All items have 'problem' and 'solution' fields")

    if count > 0:
        print(f"  Sample: {data[0]['problem'][:60]}...")

    all_pass = count >= 10 and has_fields
    print(f"  Resultado: {'2/2' if all_pass else 'FAIL'} tests pasados\n")
    return all_pass


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" TRANSFORMER MATH/PHYSICS TUTOR - TEST SUITE")
    print("=" * 60 + "\n")

    results = {}

    results["preprocess_latex"] = test_preprocess_latex()
    results["tokenizer"] = test_tokenizer()
    results["config"] = test_config()
    results["positional_encoding"] = test_positional_encoding()
    results["multihead_attention"] = test_multihead_attention()
    results["encoder_layer"] = test_encoder_layer()
    results["decoder_layer"] = test_decoder_layer()
    results["transformer"] = test_transformer()
    results["scheduler"] = test_scheduler()
    results["losses_metrics"] = test_losses_metrics()
    results["dataset_builder"] = test_dataset_builder()
    results["inference"] = test_inference()
    results["mini_training"] = test_mini_training()
    results["physics_json"] = test_physics_json()

    print("\n" + "=" * 60)
    print(" RESUMEN FINAL")
    print("=" * 60)

    total = len(results)
    passed = sum(1 for v in results.values() if v)

    for name, status in results.items():
        icon = "PASS" if status else "FAIL"
        print(f"  [{icon}] {name}")

    print(f"\n  Total: {passed}/{total} modulos pasaron todas las pruebas")

    if passed == total:
        print("\n  TODOS LOS TESTS PASARON! El proyecto funciona correctamente.")
    else:
        print(f"\n  {total - passed} modulo(s) necesitan correccion.")

    sys.exit(0 if passed == total else 1)
