"""Stable import surface for `create_app` / `app` (implementation lives in `app.http`)."""

from .http.app import app, create_app

__all__ = ["app", "create_app"]
