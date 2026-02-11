# Compliance & Privacy

Data handling, security, and operational controls for regulated deployments.

## Data

- **Tenant isolation**: All data filtered by `tenant_id`; no cross-tenant access. Documents and AI audit stored in DB.
- **LLM calls**: Text is sent to configured backend (Ollama, vLLM, etc.). Local backends keep data on your infra; `openai_compatible` with external URL sends data to that provider.
- **Retention**: Documents kept until deleted. `ai_call_audit` stores payloads for traceability; grows indefinitely unless purged.

### AI audit retention

- **Policy**: Set `AI_AUDIT_RETENTION_DAYS` (e.g. 90) to enable automatic purge of old audit records.
- **Purge**: Run `python -m app.audit` from the API directory (e.g. via cron) to delete records older than the configured days.
- **Example cron** (daily at 02:00): `0 2 * * * cd /app && python -m app.audit`

## Audit

Every AI call is logged to `ai_call_audit` (`tenant_id`, `flow_name`, `success`, `request_payload`, `response_payload`). App-level only; DB/access audits are infra-specific.

## Security

- API key required in prod via `API_KEY`; `X-API-Key` header; `X-Tenant-ID` for tenant.
- Security headers: X-Content-Type-Options, X-Frame-Options, HSTS (prod).
- CORS: restrict via `CORS_ALLOWED_ORIGINS` in prod.
- TLS and secrets: use env/secrets manager; enforce HTTPS in production.

## Privacy

Tenant-scoped data; implement data subject rights (access, deletion) at app or gateway. Only required fields sent to LLM. Responses include `source` and metadata for transparency.

## Ops

- Metrics: `/metrics` (requires X-API-Key when API_KEY set); health: `/api/v1/health` (db, redis, llm status).
- Redis (`REDIS_URL`): per-tenant rate limit (default 120/min).

## Checklist

| Area            | Status     |
|-----------------|------------|
| Tenant isolation| Done       |
| Audit trail     | Done       |
| API auth        | Required in prod |
| Rate limiting   | Redis      |
| Monitoring      | Done       |
| Retention       | Per tenant |
| Third-party LLM | Operator   |
