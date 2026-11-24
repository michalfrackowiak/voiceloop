# Infrastructure (P0-01)

Local development stack, CI, and future pipelines/dashboards for VoiceLoop.

## Layout

- `../docker-compose.yml` — Postgres 16, MinIO, and MLflow (local development only).
- `../.env.example` — non-secret defaults consumed by Compose and typed settings.
- `../.github/workflows/ci.yml` — lint, type check, and test pipeline.
- `../.pre-commit-config.yaml` — local hooks using the uv-managed toolchain.
- `pipelines/` — reserved for P2-10 (cron + CLI orchestration).
- `dashboards/` — reserved for P2-11 (Prometheus/Grafana definitions).

## Local stack

From the repository root:

```powershell
Copy-Item .env.example .env   # optional; defaults in docker-compose.yml also work
docker compose up -d
docker compose ps             # postgres, minio, and mlflow should be healthy
docker compose down           # add -v to delete named volumes
```

Endpoints with default ports:

| Service | URL | Notes |
|---|---|---|
| Postgres | `localhost:5432` | Database `voiceloop`, user `voiceloop` |
| MinIO S3 API | `http://localhost:9000` | Bucket `voiceloop-artifacts` |
| MinIO console | `http://localhost:9001` | Same dev credentials as S3 API |
| MLflow | `http://localhost:5000` | Postgres backend, S3 artifact store, `/health` |

The MLflow server waits for Postgres and MinIO health checks before starting.
Its artifact root is `s3://voiceloop-artifacts/mlflow`, served through MLflow;
the image's built-in Postgres and S3 clients are used. All passwords in
`.env.example` are development placeholders — replace them for any shared or
non-local deployment.

## Quality gates

```powershell
uv sync --extra dev
uv run ruff check .
uv run mypy .
uv run pytest -q
uv run pre-commit install          # optional; local hooks
uv run pre-commit run --all-files
```

CI (`.github/workflows/ci.yml`) runs the same three checks on Python 3.11 for
pushes to `main` and all pull requests. `uv sync --extra dev` installs the
project plus dev tools; heavy ML/runtime dependencies stay optional extras so
CI needs no GPU, network services, or credentials.

Docker is not required for CI. To sanity-check the Compose YAML without Docker:

```powershell
.\.venv\Scripts\python.exe -c "import yaml; print(list(yaml.safe_load(open('docker-compose.yml', encoding='utf-8'))['services']))"
```
