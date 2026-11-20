#!/usr/bin/env bash
# Flood test: same load against 1, 2 and 4 API containers behind nginx.
# Usage: bash locust/run_load_tests.sh [users] [duration]
set -euo pipefail
cd "$(dirname "$0")/.."
USERS=${1:-100}; DURATION=${2:-60s}; SPAWN=20
LOCUST=${LOCUST:-.venv/bin/locust}
mkdir -p locust/results
for N in 1 2 4; do
  echo "=== $N container(s) ==="
  docker compose up -d --build --scale api=$N >/dev/null
  docker compose restart lb >/dev/null   # re-resolve the scaled replicas
  for i in $(seq 1 90); do
    ok=$(curl -s localhost:8080/health | grep -c '"model_ready":true' || true)
    [ "$ok" = "1" ] && [ "$(docker compose ps api --format '{{.Health}}' | grep -c healthy)" = "$N" ] && break
    sleep 2
  done
  $LOCUST -f locust/locustfile.py --host http://localhost:8080 --headless \
          -u "$USERS" -r $SPAWN -t "$DURATION" --csv "locust/results/c$N" --only-summary 2>&1 | tail -4 || true
done
docker compose down >/dev/null
.venv/bin/python locust/summarise.py
