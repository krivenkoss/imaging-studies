@echo off
cd /d "%~dp0"
set "JUPYTER_RUNTIME_DIR=%~dp0.jupyter\runtime"
set "JUPYTER_DATA_DIR=%~dp0.jupyter\data"
call c:\Pythons\venvs\xai_p31315\Scripts\activate.bat
call jupyter.exe lab
