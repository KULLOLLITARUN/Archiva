#!/usr/bin/env bash
# Archiva — convenience startup script
# Usage: bash start.sh

set -e

echo ""
echo "============================================"
echo "             Archiva - Startup              "
echo "============================================"
echo ""

# Check .env
if [ ! -f ".env" ]; then
  echo "[X] .env file not found. Copy .env.example and fill in your keys."
  exit 1
fi

source .env

if [ -z "$GROQ_API_KEY" ] || [ "$GROQ_API_KEY" = "your_groq_api_key_here" ]; then
  echo "[X] GROQ_API_KEY is not set in .env"
  exit 1
fi

if [ -z "$DATABASE_URL" ]; then
  echo "[X] DATABASE_URL is not set in .env — see README's Database Setup section."
  exit 1
fi

# Check Python deps
echo "-> Checking Python dependencies..."
pip install -q -r requirements.txt
echo "   [OK] Dependencies ready"

# Ingest any files in test_docs/ not already in Postgres. Safe to run every
# time — load_store_from_postgres() + content-hash dedup mean this is a
# no-op past the first run, no store_state.pkl file to check for anymore.
echo ""
echo "-> Syncing test_docs/ with Postgres..."
python load_docs.py

# Start backend in background
echo ""
echo "-> Starting FastAPI backend on http://localhost:8000 ..."
uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "   [OK] Backend PID: $BACKEND_PID"

# Start frontend
echo ""
echo "-> Starting React frontend..."
cd frontend
npm install --silent
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "============================================"
echo "[OK] Archiva is running"
echo "   Backend  -> http://localhost:8000"
echo "   Frontend -> http://localhost:3000"
echo "   Press Ctrl+C to stop both servers"
echo "============================================"
echo ""

# Cleanup on Ctrl+C
trap "echo ''; echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT
wait
