"""
rebuild_tokenizer.py — Reconstruye el vocabulario del CharTokenizer.

Lee TODOS los textos de problem + solution del dataset combinado
y construye un nuevo vocab que cubra todos los caracteres necesarios.

Uso:
    python data/rebuild_tokenizer.py
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"


def main():
    print("=" * 60)
    print("  RECONSTRUCCIÓN DEL TOKENIZER")
    print("=" * 60)

    # Buscar el dataset combinado, o los individuales
    data_files = [
        DATA_DIR / "combined_math_physics.json",
        DATA_DIR / "math_combined.json",
        DATA_DIR / "physics_combined.json",
        DATA_DIR / "gsm8k_train_clean.json",
        DATA_DIR / "gsm8k_test_clean.json",
        DATA_DIR / "math_clean.json",
        DATA_DIR / "physics_problems.json",
    ]

    all_texts = []
    loaded = set()

    for path in data_files:
        if path.exists() and path.name not in loaded:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for d in data:
                if "problem" in d:
                    all_texts.append(d["problem"])
                if "solution" in d:
                    all_texts.append(d["solution"])
            print(f"  ✓ {path.name}: {len(data)} entradas")
            loaded.add(path.name)

    if not all_texts:
        print("⚠ No se encontraron datasets.")
        return

    print(f"\n  Textos totales: {len(all_texts)}")

    # Construir vocabulario
    from transformer_math_physics_tutor.data.tokenizer import CharTokenizer

    tokenizer = CharTokenizer()
    tokenizer.build_vocab(all_texts)

    # Guardar
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    vocab_path = CHECKPOINT_DIR / "vocab.json"
    tokenizer.save_vocab(str(vocab_path))

    # Estadísticas
    print(f"\n  Vocab size: {tokenizer.vocab_size} tokens")
    print(f"  Caracteres (sin especiales): {tokenizer.vocab_size - 4}")

    # Mostrar algunos caracteres
    chars = sorted(set(c for c in tokenizer.char_to_idx.keys()
                       if c not in tokenizer.special_tokens))
    print(f"  Muestra de caracteres: {''.join(chars[:50])}")

    print(f"\n{'=' * 60}")
    print("  TOKENIZER RECONSTRUIDO")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
