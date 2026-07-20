@echo off
cd /d %~dp0
call venv\Scripts\activate.bat
set HF_HUB_OFFLINE=1

echo Verification des nouveaux documents dans data\raw...
python ingest.py --si-nouveau
if errorlevel 1 (
    echo ATTENTION : l'ingestion a echoue, l'app demarre avec l'ancien index.
    pause
)

echo Verification d'Ollama...
curl -s -o nul http://localhost:11434
if not errorlevel 1 (
    echo Ollama deja lance.
    goto lancerapp
)

echo Ollama n'est pas lance, demarrage...
start "" /min ollama serve
set /a _essais=0

:waitollama
timeout /t 1 /nobreak >nul
curl -s -o nul http://localhost:11434
if not errorlevel 1 (
    echo Ollama est pret.
    goto lancerapp
)
set /a _essais+=1
if %_essais% lss 30 goto waitollama

echo ATTENTION : Ollama ne repond pas apres 30 secondes. Verifiez qu'il est
echo bien installe (https://ollama.com/download). L'app demarre quand meme :
echo la recherche de documents fonctionnera, mais pas la generation de reponses.

:lancerapp
streamlit run app.py
