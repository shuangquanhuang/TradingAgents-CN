@echo off
setlocal

set "ROOT=%~dp0"

if not exist "%ROOT%venv\Scripts\activate.bat" (
    echo [ERROR] Cannot find venv\Scripts\activate.bat
    pause
    exit /b 1
)

if not exist "%ROOT%frontend\package.json" (
    echo [ERROR] Cannot find frontend\package.json
    pause
    exit /b 1
)

echo Starting backend...
start "TradingAgents-CN Backend" cmd /k "cd /d %ROOT% && call venv\Scripts\activate.bat && python -m app"

echo Starting frontend...
start "TradingAgents-CN Frontend" cmd /k "cd /d %ROOT%frontend && npm run dev"

echo Backend and frontend startup commands have been launched.
endlocal
