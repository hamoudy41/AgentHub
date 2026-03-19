"""Backward-compatible application entrypoint.

The HTTP layer now lives under `app.http`, but `app.api` stays as the stable import
surface for the application factory used by tests and deployment entrypoints.
"""

from .http.app import app, create_app

__all__ = ["app", "create_app"]
