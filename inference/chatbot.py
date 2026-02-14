"""
chatbot.py — Chatbot tutor interactivo de matemáticas y física.

Implementa un loop interactivo donde el usuario puede hacer
preguntas de matemáticas y física, y el modelo responde.

Uso:
    from transformer_math_physics_tutor.inference.chatbot import MathPhysicsTutor
    tutor = MathPhysicsTutor(model, tokenizer)
    tutor.chat_loop()
"""

import os
import json
import time
from pathlib import Path
from typing import Optional, List, Dict

import tensorflow as tf
from transformer_math_physics_tutor.data.tokenizer import CharTokenizer
from transformer_math_physics_tutor.inference.generate import generate_text, generate_text_beam_search
from transformer_math_physics_tutor.models.config import TransformerConfig
from transformer_math_physics_tutor.models.transformer import Transformer


BASE_DIR = Path(__file__).resolve().parent.parent


class MathPhysicsTutor:
    """
    Chatbot tutor de matemáticas y física.

    Proporciona una interfaz interactiva donde el usuario puede
    escribir problemas y recibir soluciones generadas por el Transformer.

    Attributes:
        model: Modelo Transformer entrenado.
        tokenizer: Instancia de CharTokenizer.
        history: Lista de tuplas (pregunta, respuesta) del historial.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        tokenizer: CharTokenizer,
        use_beam_search: bool = False,
        beam_width: int = 3,
        max_length: int = 300,
        temperature: float = 0.3
    ):
        """
        Inicializa el tutor.

        Args:
            model: Modelo Transformer entrenado.
            tokenizer: Instancia de CharTokenizer con vocabulario cargado.
            use_beam_search: Si True, usa beam search en lugar de greedy.
            beam_width: Ancho del haz para beam search.
            max_length: Longitud máxima de respuestas generadas.
            temperature: Temperatura para generación greedy.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.use_beam_search = use_beam_search
        self.beam_width = beam_width
        self.max_length = max_length
        self.temperature = temperature
        self.history: List[Dict[str, str]] = []

    def ask(self, problem: str) -> str:
        """
        Envía un problema al modelo y obtiene la respuesta.

        Args:
            problem: Texto del problema a resolver.

        Returns:
            Texto de la solución generada.
        """
        start_time = time.time()

        if self.use_beam_search:
            answer = generate_text_beam_search(
                self.model,
                self.tokenizer,
                problem,
                max_length=self.max_length,
                beam_width=self.beam_width
            )
        else:
            answer = generate_text(
                self.model,
                self.tokenizer,
                problem,
                max_length=self.max_length,
                temperature=self.temperature
            )

        elapsed = time.time() - start_time

        # Guardar en historial
        self.history.append({
            "problem": problem,
            "answer": answer,
            "time": round(elapsed, 2)
        })

        return answer

    def chat_loop(self) -> None:
        """
        Loop interactivo de chat con el tutor.

        El usuario escribe problemas y recibe soluciones.
        Comandos especiales:
            /exit o /quit: Salir del chat
            /history: Mostrar historial
            /clear: Limpiar historial
            /help: Mostrar ayuda
            /save: Guardar historial en JSON
        """
        self._print_welcome()

        while True:
            try:
                user_input = input("\n🧑 Tú: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n¡Hasta luego!")
                break

            if not user_input:
                continue

            # Comandos especiales
            if user_input.startswith("/"):
                if self._handle_command(user_input):
                    continue
                else:
                    break  # /exit

            # Generar respuesta
            print("\n🤖 Tutor: Pensando...", end="\r")
            answer = self.ask(user_input)
            last_entry = self.history[-1]
            print(f"🤖 Tutor: {answer}")
            print(f"   (Tiempo: {last_entry['time']}s)")

    def _handle_command(self, command: str) -> bool:
        """
        Maneja comandos especiales.

        Args:
            command: Comando del usuario (empieza con /).

        Returns:
            True si se debe continuar el loop, False para salir.
        """
        cmd = command.lower().strip()

        if cmd in ["/exit", "/quit", "/salir"]:
            print("\n¡Hasta luego! Sigue practicando matemáticas y física. 📚")
            return False

        elif cmd in ["/history", "/historial"]:
            self._show_history()

        elif cmd in ["/clear", "/limpiar"]:
            self.history.clear()
            print("Historial limpiado.")

        elif cmd in ["/help", "/ayuda"]:
            self._print_help()

        elif cmd in ["/save", "/guardar"]:
            self._save_history()

        else:
            print(f"Comando desconocido: {cmd}. Escribe /help para ver comandos.")

        return True

    def _print_welcome(self) -> None:
        """Imprime el mensaje de bienvenida."""
        print("=" * 60)
        print("  🎓 TUTOR DE MATEMÁTICAS Y FÍSICA")
        print("  Transformer Encoder-Decoder")
        print("=" * 60)
        print()
        print("¡Hola! Soy tu tutor de matemáticas y física.")
        print("Escribe un problema y te ayudaré a resolverlo.")
        print()
        print("Ejemplos de problemas:")
        print("  • Solve for x: 2x + 3 = 7")
        print("  • What is 3/4 + 1/2?")
        print("  • Find the velocity if F=10N and m=2kg")
        print()
        print("Comandos: /help, /history, /save, /exit")
        print("-" * 60)

    def _print_help(self) -> None:
        """Imprime la ayuda de comandos."""
        print("\n--- Comandos disponibles ---")
        print("  /help     - Mostrar esta ayuda")
        print("  /history  - Ver historial de preguntas")
        print("  /save     - Guardar historial en JSON")
        print("  /clear    - Limpiar historial")
        print("  /exit     - Salir del tutor")

    def _show_history(self) -> None:
        """Muestra el historial de conversación."""
        if not self.history:
            print("No hay historial todavía.")
            return

        print(f"\n--- Historial ({len(self.history)} entradas) ---")
        for i, entry in enumerate(self.history, 1):
            print(f"\n  [{i}] Problema: {entry['problem']}")
            print(f"      Respuesta: {entry['answer']}")
            print(f"      Tiempo: {entry['time']}s")

    def _save_history(self) -> None:
        """Guarda el historial en un archivo JSON."""
        output_path = BASE_DIR / "chat_history.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        print(f"Historial guardado en {output_path}")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: Optional[str] = None,
        config_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        **kwargs
    ) -> "MathPhysicsTutor":
        """
        Crea un tutor desde un checkpoint guardado.

        Args:
            checkpoint_dir: Directorio con los checkpoints del modelo.
            config_path: Ruta al archivo de configuración JSON.
            vocab_path: Ruta al archivo de vocabulario JSON.
            **kwargs: Argumentos adicionales para MathPhysicsTutor.

        Returns:
            Instancia de MathPhysicsTutor lista para usar.
        """
        if checkpoint_dir is None:
            checkpoint_dir = str(BASE_DIR / "checkpoints")
        if config_path is None:
            config_path = str(BASE_DIR / "checkpoints" / "config.json")
        if vocab_path is None:
            vocab_path = str(BASE_DIR / "checkpoints" / "vocab.json")

        # Cargar configuración
        config = TransformerConfig.load(config_path)

        # Cargar tokenizer
        tokenizer = CharTokenizer(vocab_path)
        config.vocab_size = tokenizer.vocab_size

        # Crear modelo y construirlo con forward pass dummy
        model = Transformer(config)
        dummy_enc = tf.zeros((1, config.max_encoder_len), dtype=tf.int32)
        dummy_dec = tf.zeros((1, config.max_decoder_len), dtype=tf.int32)
        _ = model((dummy_enc, dummy_dec), training=False)

        # Cargar pesos (preferir best_model si existe)
        best_path = os.path.join(checkpoint_dir, "best_model.weights.h5")
        fallback_path = os.path.join(checkpoint_dir, "model_weights.weights.h5")
        weights_path = best_path if os.path.exists(best_path) else fallback_path

        if os.path.exists(weights_path):
            model.load_weights(weights_path)
            print(f"✅ Modelo cargado desde {weights_path}")
        else:
            print("⚠️ No se encontraron pesos. El modelo no está entrenado.")

        return cls(model, tokenizer, **kwargs)


if __name__ == "__main__":
    print("=" * 60)
    print("CHATBOT: Tutor de Matemáticas y Física")
    print("=" * 60)
    print()
    print("Para iniciar el chatbot necesitas un modelo entrenado.")
    print()
    print("Uso:")
    print("  # Desde checkpoint:")
    print("  tutor = MathPhysicsTutor.from_checkpoint()")
    print("  tutor.chat_loop()")
    print()
    print("  # Con modelo existente:")
    print("  tutor = MathPhysicsTutor(model, tokenizer)")
    print("  answer = tutor.ask('Solve: 2x + 3 = 7')")
