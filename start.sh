#!/usr/bin/env bash
# DocChat — convenience startup script (BM25 mode, no embedding server required)
# Usage: bash start.sh

set -e

echo ""
echo "============================================"
echo "        DocChat - Startup (BM25 mode)       "
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

# Check Python deps
echo "-> Checking Python dependencies..."
pip install -q -r requirements.txt
echo "   [OK] Dependencies ready"

# Ingest docs
if [ ! -f "store_state.pkl" ]; then
  echo ""
  echo "-> No store found. Running document ingestion..."
  python load_docs.py
else
  echo "   [OK] Store already exists (delete store_state.pkl to re-ingest)"
fi

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
echo "[OK] DocChat is running"
echo "   Backend  -> http://localhost:8000"
echo "   Frontend -> http://localhost:5173"
echo "   Press Ctrl+C to stop both servers"
echo "============================================"
echo ""

# Cleanup on Ctrl+C
trap "echo ''; echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT
wait
