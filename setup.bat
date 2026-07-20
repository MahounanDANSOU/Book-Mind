@echo off
REM ============================================================
REM  Installation complete de BookMind (a lancer une seule fois)
REM  Prerequis : Python 3.11+ et Ollama installes (voir README)
REM ============================================================
cd /d %~dp0

echo [1/5] Creation de l'environnement virtuel Python...
python -m venv venv
if errorlevel 1 (
    echo ERREUR : Python introuvable. Installez Python 3.11+ depuis https://www.python.org/downloads/
    echo et cochez "Add python.exe to PATH" pendant l'installation.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat

echo [2/5] Installation des dependances Python...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo ERREUR : l'installation des dependances a echoue. Verifiez votre connexion internet.
    pause
    exit /b 1
)

echo [3/5] Creation des dossiers de donnees...
if not exist data\raw mkdir data\raw
if not exist data\db mkdir data\db
if not exist data\eval.jsonl type nul > data\eval.jsonl

echo [4/5] Telechargement du modele d'embeddings bge-m3 via Ollama (~1.2 Go, une seule fois)...
ollama pull bge-m3
if errorlevel 1 (
    echo ATTENTION : Ollama n'est pas installe ou pas dans le PATH.
    echo Installez-le depuis https://ollama.com/download puis relancez :
    echo     ollama pull bge-m3
)

echo [5/5] Telechargement du modele LLM via Ollama (~5.2 Go, une seule fois)...
ollama pull qwen3:8b
if errorlevel 1 (
    echo ATTENTION : Ollama n'est pas installe ou pas dans le PATH.
    echo Installez-le depuis https://ollama.com/download puis relancez :
    echo     ollama pull qwen3:8b
)

echo.
echo ============================================================
echo  Installation du composant OCR (PaddleOCR - optionnel)
echo  Recommande pour les PDF scannes ou texte piege dans des images
echo  GPU Nvidia detecte : installation optimisee
echo ============================================================
call setup_ocr.bat

echo.
echo ============================================================
echo  Installation terminee !
echo  1. Placez vos documents (.pdf, .txt, .md) dans data\raw\
echo  2. Lancez start_app.bat
echo ============================================================
pause
