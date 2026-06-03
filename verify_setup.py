"""
verify_setup.py — Comprueba que el entorno está correctamente configurado.
Ejecutar con: python verify_setup.py
"""

import sys

def verificar_python():
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro} — ", end="")
    if version.major == 3 and version.minor >= 10:
        print("OK")
    else:
        print("ADVERTENCIA: se recomienda Python 3.10+")

def verificar_paquete(nombre_import, nombre_display=None):
    if nombre_display is None:
        nombre_display = nombre_import
    try:
        modulo = __import__(nombre_import)
        version = getattr(modulo, "__version__", "version desconocida")
        print(f"{nombre_display} {version} — OK")
    except ImportError:
        print(f"{nombre_display} — ERROR: no instalado")

def verificar_torch():
    try:
        import torch
        print(f"torch {torch.__version__} — OK")
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Dispositivo disponible: {dispositivo}")
        # Prueba basica: crear un tensor y hacer una operacion
        x = torch.tensor([1.0, 2.0, 3.0])
        assert x.sum().item() == 6.0
        print(f"  Operacion de prueba (suma de tensor): OK")
    except ImportError:
        print("torch — ERROR: no instalado")

def verificar_dotenv():
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("python-dotenv — OK (.env cargado)")
    except ImportError:
        print("python-dotenv — ERROR: no instalado")

def verificar_estructura():
    import os
    carpetas = ["data/raw", "data/processed", "data/generated", "src", "checkpoints", "logs"]
    print("\nEstructura de carpetas:")
    for carpeta in carpetas:
        existe = os.path.isdir(carpeta)
        estado = "OK" if existe else "FALTA"
        print(f"  {carpeta}/ — {estado}")

if __name__ == "__main__":
    print("=" * 45)
    print("  Verificacion del entorno PokeGPT")
    print("=" * 45)
    verificar_python()
    verificar_torch()
    verificar_paquete("numpy")
    verificar_dotenv()
    verificar_estructura()
    print("=" * 45)
