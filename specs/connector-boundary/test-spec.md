# RM-CONNECTOR-001 test contract

| Requirement | Verification |
|---|---|
| 01 | Static import census plus test that every old-package import has a recorded disposition. |
| 02 | Fake SDK client exercises tool discovery and transport success; absent SDK returns typed failure. |
| 03 | Temporary Git tests for lane allocation, guarded commit, queue submission, and concept reservation import only RM governance. |
| 04 | Golden MCP/CLI schemas and response/error fixtures before and after migration. |
| 05 | Build/install package in a clean environment and run startup smoke plus retired-symbol search. |

Negative tests include an import cycle, removed private symbol, missing dependency, stale lane fence, and red validation gate. Run full repository pre-commit and the CCCC staged complexity check (no new cyclomatic >10/cognitive >15 function), plus Ruff/mypy/pytest. jscpd and Dupehound are not ordinary RM hooks; report any supplemental run or tool gap. KISS requires one implementation per responsibility and no transitional dual-write/fallback once cutover lands.
