# setup.ps1 — Crea el entorno virtual e instala las dependencias del proyecto.
# Ejecucion: .\setup.ps1  (desde la raiz del proyecto)

$PYTHON = "C:\Users\Jaime\AppData\Local\Programs\Python\Python312\python.exe"
$VENV   = ".venv"

Write-Host "==> Creando entorno virtual en $VENV ..."
& $PYTHON -m venv $VENV

Write-Host "==> Activando entorno virtual ..."
& "$VENV\Scripts\Activate.ps1"

Write-Host "==> Actualizando pip ..."
& "$VENV\Scripts\python.exe" -m pip install --upgrade pip

Write-Host "==> Instalando dependencias desde requirements.txt ..."
& "$VENV\Scripts\pip.exe" install -r requirements.txt

Write-Host ""
Write-Host "Listo. Para activar el entorno en el futuro ejecuta:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
