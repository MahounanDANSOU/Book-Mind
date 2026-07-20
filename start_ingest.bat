@echo off
cd /d %~dp0
call venv\Scripts\activate.bat
set HF_HUB_OFFLINE=1
python ingest.py
pause