"""Mark ``tests`` as a regular package.

Without this file, ``tests/`` is an implicit PEP 420 namespace package. In the
shared agent-utilities environment, several installed distributions
(third-party ``caio`` and ``linkpreview``, plus formerly some of our own
fleet packages — see INFRA-4) also ship a top-level ``tests/`` directory. When
any of those ships a real ``tests/__init__.py`` (``linkpreview`` does, and
cannot be fixed upstream), Python's import system treats that installed
distribution's ``tests`` as a *regular* package — and a regular package found
anywhere on ``sys.path`` wins outright over an implicit namespace package,
regardless of search order. That shadowed this repository's own
``tests.fixtures`` submodule with ``ModuleNotFoundError: No module named
'tests.fixtures'`` whenever pytest ran through the shared environment.

Giving this repository's own ``tests/`` an ``__init__.py`` makes it a regular
package too. Combined with pytest's default ``prepend`` import mode — which
inserts this package's parent (the repository root) at the front of
``sys.path`` before collection — our own ``tests`` package is now the first
regular-package match Python's path-based finder returns, so it resolves
before any shadowing package installed elsewhere on ``sys.path`` is ever
reached. See ``plans/repository-manager-development-program/BUG-LEDGER.md``
(INFRA-4) for the reproduction and the evaluation of alternative mitigations
(``pythonpath``/``rootdir`` alone and ``--import-mode=importlib`` +
``consider_namespace_packages`` alone were both tested and do **not** resolve
the collision; only this does).
"""
