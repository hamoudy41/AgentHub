from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_health(client):
    r = await client.get("/api/v1/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "environment" in data
    assert "timestamp" in data
    assert "db_ok" in data
    assert "redis_ok" in data
    assert "llm_ok" in data
    assert data["llm_ok"] is True  # conftest sets LLM_PROVIDER + LLM_BASE_URL


@pytest.mark.asyncio
async def test_tenant_default_when_no_header(client):
    r = await client.post(
        "/api/v1/documents",
        json={"id": "d1", "title": "T", "text": "body"},
    )
    assert r.status_code == 201
    assert r.json()["id"] == "d1"


@pytest.mark.asyncio
async def test_documents_create_and_get(client, tenant_headers):
    r = await client.post(
        "/api/v1/documents",
        headers=tenant_headers,
        json={"id": "doc-1", "title": "Title", "text": "Content"},
    )
    assert r.status_code == 201
    data = r.json()
    assert data["id"] == "doc-1"
    assert data["title"] == "Title"
    assert data["text"] == "Content"
    assert "created_at" in data

    r2 = await client.get("/api/v1/documents/doc-1", headers=tenant_headers)
    assert r2.status_code == 200
    assert r2.json()["text"] == "Content"


@pytest.mark.asyncio
async def test_documents_get_404_wrong_tenant(client, tenant_headers):
    await client.post(
        "/api/v1/documents",
        headers=tenant_headers,
        json={"id": "doc-2", "title": "T", "text": "X"},
    )
    r = await client.get(
        "/api/v1/documents/doc-2",
        headers={"X-Tenant-ID": "other-tenant"},
    )
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_documents_get_404_missing(client, tenant_headers):
    r = await client.get(
        "/api/v1/documents/nonexistent",
        headers=tenant_headers,
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_documents_create_duplicate_returns_409(client, tenant_headers):
    await client.post(
        "/api/v1/documents",
        headers=tenant_headers,
        json={"id": "dup", "title": "First", "text": "Content"},
    )
    r = await client.post(
        "/api/v1/documents",
        headers=tenant_headers,
        json={"id": "dup", "title": "Second", "text": "Other"},
    )
    assert r.status_code == 409
    assert "already exists" in r.json()["detail"]


@pytest.mark.asyncio
async def test_documents_upload(client, tenant_headers):
    files = {"file": ("doc.txt", b"Hello world", "text/plain")}
    data = {}
    r = await client.post(
        "/api/v1/documents/upload",
        headers=tenant_headers,
        files=files,
        data=data,
    )
    assert r.status_code == 201
    out = r.json()
    assert out["id"] == "doc"
    assert out["title"] == "doc.txt"
    assert out["text"] == "Hello world"
    assert "created_at" in out


@pytest.mark.asyncio
async def test_documents_upload_formats(client, tenant_headers):
    """Upload succeeds for .txt, .md, .json, .csv, .xml, .html formats."""
    formats = [
        ("readme.md", b"# Title\nMarkdown content", "text/markdown"),
        ("data.json", b'{"key": "value"}', "application/json"),
        ("sheet.csv", b"a,b,c\n1,2,3", "text/csv"),
        ("page.xml", b"<root><item>text</item></root>", "application/xml"),
        ("index.html", b"<html><body>Hello</body></html>", "text/html"),
    ]
    for filename, content, mime in formats:
        stem = filename.rsplit(".", 1)[0]
        files = {"file": (filename, content, mime)}
        r = await client.post(
            "/api/v1/documents/upload",
            headers=tenant_headers,
            files=files,
        )
        assert r.status_code == 201, f"Failed for {filename}"
        out = r.json()
        assert out["id"] == stem
        assert out["title"] == filename
        assert out["text"] == content.decode("utf-8")
        assert "created_at" in out


@pytest.mark.asyncio
async def test_documents_upload_with_custom_id_and_title(client, tenant_headers):
    files = {"file": ("any.txt", b"Content", "text/plain")}
    data = {"document_id": "custom-id", "title": "Custom Title"}
    r = await client.post(
        "/api/v1/documents/upload",
        headers=tenant_headers,
        files=files,
        data=data,
    )
    assert r.status_code == 201
    out = r.json()
    assert out["id"] == "custom-id"
    assert out["title"] == "Custom Title"
    assert out["text"] == "Content"


@pytest.mark.asyncio
async def test_documents_upload_file_too_large_returns_413(client, tenant_headers):
    # 5 MB + 1 byte
    oversized = b"x" * (5 * 1024 * 1024 + 1)
    files = {"file": ("big.txt", oversized, "text/plain")}
    r = await client.post(
        "/api/v1/documents/upload",
        headers=tenant_headers,
        files=files,
    )
    assert r.status_code == 413
    assert "too large" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_documents_upload_invalid_encoding_returns_400(client, tenant_headers):
    # Latin-1 / binary that is not valid UTF-8
    invalid_utf8 = b"\xff\xfe\x00\x01"
    files = {"file": ("bad.txt", invalid_utf8, "text/plain")}
    r = await client.post(
        "/api/v1/documents/upload",
        headers=tenant_headers,
        files=files,
    )
    assert r.status_code == 400
    assert "utf-8" in r.json()["detail"].lower() or "decode" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_documents_upload_filename_too_long_returns_400(client, tenant_headers):
    """Filename stem exceeding 64 chars yields 400, not 500."""
    long_stem = "a" * 65
    files = {"file": (f"{long_stem}.txt", b"Content", "text/plain")}
    r = await client.post(
        "/api/v1/documents/upload",
        headers=tenant_headers,
        files=files,
    )
    assert r.status_code == 400
    assert "64" in r.json()["detail"]
    assert "document_id" in r.json()["detail"].lower() or "filename" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_documents_upload_explicit_document_id_too_long_returns_400(client, tenant_headers):
    """Explicit document_id exceeding 64 chars yields 400."""
    files = {"file": ("short.txt", b"Content", "text/plain")}
    data = {"document_id": "x" * 65}
    r = await client.post(
        "/api/v1/documents/upload",
        headers=tenant_headers,
        files=files,
        data=data,
    )
    assert r.status_code == 400
    assert "64" in r.json()["detail"]


@pytest.mark.asyncio
async def test_documents_upload_missing_file_returns_422(client, tenant_headers):
    r = await client.post(
        "/api/v1/documents/upload",
        headers=tenant_headers,
        data={"document_id": "x", "title": "T"},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_documents_upload_duplicate_returns_409(client, tenant_headers):
    """Explicit document_id conflicts with existing upload."""
    files = {"file": ("a.txt", b"First", "text/plain")}
    await client.post("/api/v1/documents/upload", headers=tenant_headers, files=files)
    r = await client.post(
        "/api/v1/documents/upload",
        headers=tenant_headers,
        files={"file": ("b.txt", b"Second", "text/plain")},
        data={"document_id": "a"},
    )
    assert r.status_code == 409
    assert "already exists" in r.json()["detail"]


@pytest.mark.asyncio
async def test_documents_upload_duplicate_same_filename_returns_409(client, tenant_headers):
    """Same filename twice: ID derived from stem conflicts on second upload."""
    files = {"file": ("report.txt", b"First report", "text/plain")}
    await client.post("/api/v1/documents/upload", headers=tenant_headers, files=files)
    r = await client.post(
        "/api/v1/documents/upload",
        headers=tenant_headers,
        files={"file": ("report.txt", b"Second report", "text/plain")},
    )
    assert r.status_code == 409
    assert "already exists" in r.json()["detail"]
    assert "report" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_documents_upload_duplicate_vs_json_create_returns_409(client, tenant_headers):
    """Upload with document_id that matches doc created via POST /documents."""
    await client.post(
        "/api/v1/documents",
        headers=tenant_headers,
        json={"id": "conflict", "title": "From JSON", "text": "Existing"},
    )
    r = await client.post(
        "/api/v1/documents/upload",
        headers=tenant_headers,
        files={"file": ("other.txt", b"New content", "text/plain")},
        data={"document_id": "conflict"},
    )
    assert r.status_code == 409
    assert "already exists" in r.json()["detail"]


@pytest.mark.asyncio
async def test_notary_summarize_with_document_id(client, tenant_headers):
    await client.post(
        "/api/v1/documents",
        headers=tenant_headers,
        json={"id": "ndoc", "title": "N", "text": "Notary document body here."},
    )
    r = await client.post(
        "/api/v1/ai/notary/summarize",
        headers=tenant_headers,
        json={"document_id": "ndoc", "text": "ignored", "language": "en"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["document_id"] == "ndoc"
    assert data["source"] in {"llm", "fallback"}
    assert "summary" in data


@pytest.mark.asyncio
async def test_classify(client, tenant_headers):
    r = await client.post(
        "/api/v1/ai/classify",
        headers=tenant_headers,
        json={
            "text": "This is an invoice for 100 euros.",
            "candidate_labels": ["invoice", "letter", "contract"],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["label"] in ["invoice", "letter", "contract"]
    assert data["source"] in {"llm", "fallback"}
    assert "model" in data


@pytest.mark.asyncio
async def test_ask(client, tenant_headers):
    r = await client.post(
        "/api/v1/ai/ask",
        headers=tenant_headers,
        json={"question": "What is the total?", "context": "The total amount is 50 EUR."},
    )
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert data["source"] in {"llm", "fallback"}


@pytest.mark.asyncio
async def test_validation_error_returns_422(client, tenant_headers):
    r = await client.post(
        "/api/v1/ai/notary/summarize",
        headers=tenant_headers,
        json={},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_classify_400_when_llm_not_configured(client, tenant_headers):
    with patch("app.services_ai_flows.llm_client.is_configured", return_value=False):
        r = await client.post(
            "/api/v1/ai/classify",
            headers=tenant_headers,
            json={"text": "Hello", "candidate_labels": ["a", "b"]},
        )
    assert r.status_code == 400
    assert "LLM not configured" in r.json()["detail"]


@pytest.mark.asyncio
async def test_response_includes_security_headers(client):
    """Responses include X-Content-Type-Options, X-Frame-Options, Referrer-Policy."""
    r = await client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.headers.get("X-Content-Type-Options") == "nosniff"
    assert r.headers.get("X-Frame-Options") == "DENY"
    assert "strict-origin" in (r.headers.get("Referrer-Policy") or "")


@pytest.mark.asyncio
async def test_response_includes_request_id(client):
    """Response includes X-Request-ID, from request or generated."""
    r = await client.get("/api/v1/health")
    assert r.status_code == 200
    assert "X-Request-ID" in r.headers
    assert len(r.headers["X-Request-ID"]) >= 16


@pytest.mark.asyncio
async def test_request_id_propagates_from_header(client):
    """When client sends X-Request-ID, same value is returned."""
    req_id = "test-request-id-12345"
    r = await client.get("/api/v1/health", headers={"X-Request-ID": req_id})
    assert r.status_code == 200
    assert r.headers.get("X-Request-ID") == req_id


@pytest.mark.asyncio
async def test_documents_create_rejects_text_over_max_length(client, tenant_headers):
    """Document text exceeding 500k chars returns 422."""
    long_text = "x" * 500_001
    r = await client.post(
        "/api/v1/documents",
        headers=tenant_headers,
        json={"id": "d1", "title": "T", "text": long_text},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_ask_rejects_question_over_max_length(client, tenant_headers):
    """Ask question exceeding 2000 chars returns 422."""
    long_q = "x" * 2001
    r = await client.post(
        "/api/v1/ai/ask",
        headers=tenant_headers,
        json={"question": long_q, "context": "Some context."},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_classify_rejects_too_many_candidate_labels(client, tenant_headers):
    """Classify with more than 10 candidate_labels returns 422."""
    labels = [f"label_{i}" for i in range(11)]
    r = await client.post(
        "/api/v1/ai/classify",
        headers=tenant_headers,
        json={"text": "Hello", "candidate_labels": labels},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_cors_allows_origin(client):
    """CORS middleware adds Access-Control-Allow-Origin for requests with Origin header."""
    r = await client.get(
        "/api/v1/health",
        headers={"Origin": "https://example.com"},
    )
    assert r.status_code == 200
    assert "access-control-allow-origin" in [h.lower() for h in r.headers]


@pytest.mark.asyncio
async def test_metrics_requires_api_key_when_configured(client_with_api_key):
    """When API_KEY is set, /metrics returns 401 without X-API-Key."""
    r = await client_with_api_key.get("/metrics")
    assert r.status_code == 401
    assert (
        "api key" in r.json().get("detail", "").lower()
        or "key" in r.json().get("detail", "").lower()
    )


@pytest.mark.asyncio
async def test_metrics_succeeds_with_valid_api_key(client_with_api_key):
    """When API_KEY is set, /metrics returns 200 with valid X-API-Key."""
    r = await client_with_api_key.get("/metrics", headers={"X-API-Key": "test-secret-key"})
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_health_remains_public_when_api_key_configured(client_with_api_key):
    """Health endpoint does not require API key."""
    r = await client_with_api_key.get("/api/v1/health")
    assert r.status_code == 200
