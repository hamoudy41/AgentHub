from __future__ import annotations

from app.core.errors import AuthorizationError, ConfigurationError, ServiceUnavailableError


def test_core_error_subclass_constructors_cover_remaining_paths():
    cfg = ConfigurationError("cfg")
    auth = AuthorizationError("auth")
    svc = ServiceUnavailableError("svc", service_name="redis")

    assert cfg.error_code == "CONFIGURATION_ERROR"
    assert auth.error_code == "AUTHORIZATION_ERROR"
    assert svc.details["service"] == "redis"
