#!/usr/bin/env bash
# Levanta The Intelligence System con una sola instrucción:  ./run.sh
#
#   ./run.sh          construye y levanta todo, y espera a que la API responda
#   ./run.sh down     detiene los contenedores (los datos quedan en los volúmenes)
#   ./run.sh logs     sigue los logs de api y worker
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-up}" in
  down) exec docker compose down ;;
  logs) exec docker compose logs -f api worker ;;
  up) ;;
  *) echo "uso: ./run.sh [up|down|logs]" >&2; exit 1 ;;
esac

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Se creó .env a partir de .env.example: completá ANTHROPIC_API_KEY y volvé a correr ./run.sh" >&2
  exit 1
fi
if grep -q '^ANTHROPIC_API_KEY=sk-ant-\.\.\.$' .env; then
  echo "Falta ANTHROPIC_API_KEY en .env" >&2
  exit 1
fi

docker compose up --build -d

echo -n "Esperando a la API"
for _ in $(seq 1 60); do
  if curl -fs http://localhost:8000/health >/dev/null 2>&1; then
    echo " lista."
    echo
    echo "  API (Swagger):  http://localhost:8000/docs"
    echo "  Phoenix:        http://localhost:6006"
    echo
    echo "La primera vez el worker descarga el modelo de embeddings: seguilo con ./run.sh logs"
    exit 0
  fi
  echo -n "."
  sleep 2
done
echo
echo "La API no respondió a tiempo. Revisá: docker compose logs api" >&2
exit 1
