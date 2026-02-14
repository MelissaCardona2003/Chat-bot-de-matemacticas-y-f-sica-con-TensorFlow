"""
download_dataset.py — Descarga MATH dataset de hendrycks/math.

Descarga el dataset MATH completo desde el repositorio de Hendrycks et al.,
lo extrae y organiza en carpetas por tipo (train/test) y tema.

El dataset MATH contiene problemas matemáticos formales con soluciones
paso a paso, organizados por tema y nivel de dificultad (1-5).

Uso:
    python data/download_dataset.py
"""

import os
import sys
import json
import time
import tarfile
import shutil
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, List, Dict


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MATH_DIR = DATA_DIR / "MATH"

# URL principal del dataset
MATH_URL = "https://people.eecs.berkeley.edu/~hendrycks/MATH.tar"

# URLs alternativas
MATH_URLS = [
    "https://people.eecs.berkeley.edu/~hendrycks/MATH.tar",
    "https://zenodo.org/record/7934955/files/MATH.tar",
]


def download_file(
    url: str,
    dest_path: Path,
    max_retries: int = 3,
    timeout: int = 60
) -> bool:
    """
    Descarga un archivo desde una URL con reintentos y barra de progreso.

    Args:
        url: URL del archivo a descargar.
        dest_path: Ruta local donde guardar el archivo.
        max_retries: Número máximo de reintentos en caso de fallo.
        timeout: Timeout en segundos para la conexión.

    Returns:
        True si la descarga fue exitosa, False en caso contrario.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Intento {attempt}/{max_retries}: {url}")

            # Crear request con User-Agent
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Python/transformer_tutor)"}
            )

            with urllib.request.urlopen(req, timeout=timeout) as response:
                total_size = response.headers.get("Content-Length")
                total_size = int(total_size) if total_size else None

                downloaded = 0
                chunk_size = 8192 * 16  # 128KB chunks

                with open(dest_path, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Barra de progreso simple
                        if total_size:
                            pct = downloaded / total_size * 100
                            mb_down = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            print(
                                f"\r  Descargando: {mb_down:.1f}/{mb_total:.1f} MB "
                                f"({pct:.1f}%)",
                                end="", flush=True
                            )
                        else:
                            mb_down = downloaded / (1024 * 1024)
                            print(
                                f"\r  Descargando: {mb_down:.1f} MB",
                                end="", flush=True
                            )

                print()  # Nueva línea después de la barra
                print(f"  Descarga completada: {dest_path}")
                return True

        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            print(f"  Error en intento {attempt}: {e}")
            if dest_path.exists():
                dest_path.unlink()
            if attempt < max_retries:
                wait = attempt * 5
                print(f"  Esperando {wait}s antes de reintentar...")
                time.sleep(wait)

    return False


def extract_tar(tar_path: Path, extract_dir: Path) -> bool:
    """
    Extrae un archivo .tar en el directorio especificado.

    Args:
        tar_path: Ruta al archivo .tar.
        extract_dir: Directorio donde extraer.

    Returns:
        True si la extracción fue exitosa.
    """
    try:
        print(f"  Extrayendo {tar_path.name}...")
        extract_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(tar_path, "r:*") as tar:
            # Filtrar para seguridad (evitar path traversal)
            members = tar.getmembers()
            safe_members = [
                m for m in members
                if not m.name.startswith("/") and ".." not in m.name
            ]
            tar.extractall(path=extract_dir, members=safe_members)

        print(f"  Extraído en {extract_dir}")
        return True

    except (tarfile.TarError, OSError) as e:
        print(f"  Error al extraer: {e}")
        return False


def collect_problems(math_dir: Path) -> Dict[str, List[Dict]]:
    """
    Lee todos los archivos JSON individuales del MATH dataset y los organiza.

    El dataset MATH tiene la estructura:
        MATH/train/algebra/1.json, 2.json, ...
        MATH/train/prealgebra/1.json, 2.json, ...
        MATH/test/algebra/1.json, ...

    Cada JSON tiene: {"problem": str, "level": str, "type": str, "solution": str}

    Args:
        math_dir: Directorio raíz del MATH dataset extraído.

    Returns:
        Diccionario con claves "train" y "test", cada uno con lista de problemas.
    """
    results = {"train": [], "test": []}

    for split in ["train", "test"]:
        split_dir = math_dir / split
        if not split_dir.exists():
            # Intentar sin subdirectorio MATH/
            for candidate in math_dir.iterdir():
                if candidate.is_dir() and (candidate / split).exists():
                    split_dir = candidate / split
                    break

        if not split_dir.exists():
            print(f"  ADVERTENCIA: No se encontró directorio {split}")
            continue

        count = 0
        for topic_dir in sorted(split_dir.iterdir()):
            if not topic_dir.is_dir():
                continue

            topic = topic_dir.name

            for json_file in sorted(topic_dir.glob("*.json")):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Extraer nivel como entero
                    level_str = data.get("level", "Level 1")
                    try:
                        level = int(level_str.replace("Level ", ""))
                    except (ValueError, AttributeError):
                        level = 0

                    problem = {
                        "problem": data.get("problem", ""),
                        "solution": data.get("solution", ""),
                        "topic": data.get("type", topic).lower().replace(" ", "_"),
                        "level": level,
                    }
                    results[split].append(problem)
                    count += 1

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"  Error leyendo {json_file}: {e}")
                    continue

        print(f"  {split}: {count} problemas cargados")

    return results


def save_collected(problems: Dict[str, List[Dict]], output_dir: Path) -> None:
    """
    Guarda los problemas recolectados en archivos JSON consolidados.

    Args:
        problems: Diccionario con splits "train"/"test".
        output_dir: Directorio donde guardar.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, data in problems.items():
        if data:
            output_path = output_dir / f"math_{split}_raw.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  Guardado {output_path} ({len(data)} problemas)")


def main():
    """Pipeline completo de descarga del MATH dataset."""
    print("=" * 60)
    print("  DESCARGA DEL MATH DATASET")
    print("  (Hendrycks et al., 2021)")
    print("=" * 60)

    # Verificar si ya existe
    if MATH_DIR.exists() and any(MATH_DIR.rglob("*.json")):
        json_count = len(list(MATH_DIR.rglob("*.json")))
        print(f"\nEl dataset MATH ya existe en {MATH_DIR}")
        print(f"  Archivos JSON encontrados: {json_count}")
        print("  Usa --force para volver a descargar.")

        if "--force" not in sys.argv:
            # Aún así recolectar y consolidar
            print("\nRecolectando problemas...")
            problems = collect_problems(MATH_DIR)
            save_collected(problems, DATA_DIR)
            return

    # Descargar
    tar_path = DATA_DIR / "MATH.tar"
    downloaded = False

    print("\nDescargando MATH dataset...")
    for url in MATH_URLS:
        if download_file(url, tar_path):
            downloaded = True
            break

    if not downloaded:
        print("\n" + "!" * 60)
        print("  NO SE PUDO DESCARGAR EL MATH DATASET")
        print("!" * 60)
        print()
        print("Posibles soluciones:")
        print("  1. Verificar conexión a internet")
        print("  2. Descargar manualmente desde:")
        print(f"     {MATH_URL}")
        print(f"     y colocar el .tar en: {tar_path}")
        print("  3. Usar el dataset existente: math_training_data.json")
        print("     (215 problemas simples ya incluidos)")
        print()
        print("El proyecto puede funcionar sin el dataset MATH completo.")
        return

    # Extraer
    print("\nExtrayendo dataset...")
    if not extract_tar(tar_path, MATH_DIR):
        print("Error al extraer. Intenta manualmente.")
        return

    # Recolectar y consolidar
    print("\nRecolectando problemas...")
    problems = collect_problems(MATH_DIR)
    save_collected(problems, DATA_DIR)

    # Limpiar archivo tar (opcional, ahorrar espacio)
    if tar_path.exists():
        size_mb = tar_path.stat().st_size / (1024 * 1024)
        print(f"\nArchivo tar ({size_mb:.1f} MB) disponible en: {tar_path}")
        print("  Puedes eliminarlo manualmente para ahorrar espacio.")

    # Resumen
    total = sum(len(v) for v in problems.values())
    print(f"\n{'=' * 60}")
    print(f"  DESCARGA COMPLETADA")
    print(f"  Total de problemas: {total}")
    for split, data in problems.items():
        if data:
            topics = set(p["topic"] for p in data)
            print(f"  {split}: {len(data)} problemas, {len(topics)} temas")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
