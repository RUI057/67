@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title 手語辨識專案
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

rem ==============================================================
rem  Windows 專用：雙擊就會開啟操作選單（收集資料 / 訓練 / 測試）。
rem  第一次會自動建立專用環境、安裝套件（約 5 到 20 分鐘），之後幾秒就開好。
rem  設定流程都在 setup_env.py；這裡只負責找到版本相容的 Python。
rem ==============================================================

set "PY="

rem 1. 已經設定好的專用環境：直接用
if exist "%PUBLIC%\cyut_venv\Scripts\python.exe" set PY="%PUBLIC%\cyut_venv\Scripts\python.exe"
if defined PY %PY% -c "import sys" >nul 2>&1 || set "PY="
if defined PY goto run

rem 2. 找相容的 Python：3.9 到 3.12（3.13 以上 mediapipe 還不支援）
for %%V in (3.11 3.12 3.10 3.9) do if not defined PY py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
if defined PY goto run
for %%C in (python python3) do if not defined PY %%C -c "import sys; sys.exit(0 if (3,9) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>&1 && set "PY=%%C"
if defined PY goto run
goto nopython

:run
%PY% setup_env.py
if errorlevel 1 pause
goto end

:nopython
echo.
echo  [X] 找不到可用的 Python（需要 3.9 到 3.12 版）
echo.
echo  目前電腦上裝的 Python：
py -0 2>nul || echo    （沒有偵測到）
echo.
echo  注意：Python 3.13 以上太新，mediapipe 還不支援，請安裝 3.11。
echo.
where winget >nul 2>&1 || goto manual
choice /C YN /M "要現在自動安裝 Python 3.11 嗎"
if errorlevel 2 goto manual
winget install -e --id Python.Python.3.11
echo.
echo  安裝完成後，請關閉這個視窗，再雙擊一次「開始.bat」。
pause
goto end

:manual
echo  請下載 Python 3.11（頁面往下捲，選 Windows installer 64-bit）：
echo    https://www.python.org/downloads/release/python-3119/
echo  安裝時記得勾選「Add python.exe to PATH」，裝完再雙擊一次「開始.bat」。
start "" "https://www.python.org/downloads/release/python-3119/"
pause
goto end

:end
endlocal
