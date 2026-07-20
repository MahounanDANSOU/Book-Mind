@echo off
REM ============================================================
REM  Installation OCR (PaddleOCR) pour BookMind
REM  Moteur 100%% local — utilise pour les PDF scannes / texte
REM  piege dans des images. Version CPU par defaut.
REM  Pour un GPU Nvidia : voir https://www.paddlepaddle.org.cn/en/install
REM  (remplacer paddlepaddle par la version GPU adaptee a votre CUDA)
REM ============================================================
cd /d %~dp0

echo Activation de l'environnement virtuel...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERREUR : impossible d'activer le venv. Lancez setup.bat d'abord.
    pause
    exit /b 1
)

echo Installation de PaddleOCR et de son moteur paddlepaddle...
pip install paddleocr paddlepaddle
if errorlevel 1 (
    echo ERREUR : installation de PaddleOCR impossible. Verifiez votre connexion internet.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  OCR (PaddleOCR) installe et pret !
echo  - Le modele de reconnaissance (~100 Mo) sera telecharge et
echo    mis en cache a la premiere page OCRisee.
echo  - Langue reglable dans config.yaml (ocr_langue, defaut: fr)
echo  Verifiez que config.yaml a bien : ocr_fallback: true
echo ============================================================
pause
