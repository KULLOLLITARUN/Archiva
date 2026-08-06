@echo off
setlocal

echo ==============================================
echo           Archiva - Startup
echo ==============================================
echo.

if not exist ".env" (
    echo [X] .env file not found. Copy .env.example and fill in your keys.
    exit /b 1
)

:: ── Kill any existing backend / frontend processes ────────────────────────────
echo - Stopping any existing servers...
taskkill /F /IM uvicorn.exe >nul 2>&1 && echo   [+] Killed old uvicorn (backend)
taskkill /F /FI "WINDOWTITLE eq Archiva Backend" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq Archiva Frontend" >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000 " ^| findstr "LISTENING" 2^>nul') do (
    taskkill /PID %%a /F >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":3000 " ^| findstr "LISTENING" 2^>nul') do (
    taskkill /PID %%a /F >nul 2>&1
)
ping -n 2 127.0.0.1 >nul

:: ── Activate virtual environment ─────────────────────────────────────────────
if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
) else (
    echo [X] No venv found. Run: python -m venv venv ^&^& pip install -r requirements.txt
    exit /b 1
)

:: ── Python dependencies (only if not already satisfied) ──────────────────────
echo - Checking Python dependencies...
pip install -q -r requirements.txt

:: ── Document store (skip if pickle already exists) ───────────────────────────
echo.
if exist "store_state.pkl" (
    echo - Store found — skipping re-ingestion.  ^(Delete store_state.pkl to force a rebuild.^)
) else (
    echo - No store found. Running document ingestion...
    python load_docs.py
    if errorlevel 1 (
        echo.
        echo [X] Document ingestion failed and no usable store was created.
        echo     Add supported files to test_docs\ and run start.bat again.
        exit /b 1
    )
)

:: ── Frontend dependencies (only if node_modules missing) ─────────────────────
cd frontend
if not exist "node_modules" (
    echo - Installing frontend dependencies...
    call npm install --silent
) else (
    echo - Frontend node_modules found — skipping npm install.
)
cd ..

echo.
echo - Starting FastAPI backend on http://localhost:8000 ...
start "Archiva Backend" cmd /k "if exist venv\Scripts\activate.bat (call venv\Scripts\activate.bat) & uvicorn main:app --host 0.0.0.0 --port 8000"

:: ── Wait for backend to be ready (up to 60 seconds) ──────────────────────────
echo - Waiting for backend to start...
set /a tries=0
:wait_loop
    set /a tries+=1
    if %tries% gtr 30 (
        echo   [!] Backend took too long — starting frontend anyway.
        goto start_frontend
    )
    netstat -ano | findstr ":8000 " | findstr "LISTENING" >nul 2>&1
    if errorlevel 1 (
        ping -n 2 127.0.0.1 >nul
        goto wait_loop
    )
echo   [OK] Backend is ready!

:start_frontend
echo.
echo - Starting React frontend (Vite) on http://localhost:3000 ...
cd frontend
start "Archiva Frontend" cmd /k "npm run dev"
cd ..

echo.
echo ==============================================
echo [OK] Archiva is starting up!
echo   Backend  - http://localhost:8000
echo   Frontend - http://localhost:3000
echo.
echo   Close the two newly opened console windows to stop the servers.
echo ==============================================
