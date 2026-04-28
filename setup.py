"""Compatibility shim for legacy build frontends.

Project metadata, dependencies, package discovery and console scripts live in
pyproject.toml. The maintained development workflow is uv based; keep this file
minimal so setup.py does not become a second source of truth.
"""

from setuptools import setup


setup()
