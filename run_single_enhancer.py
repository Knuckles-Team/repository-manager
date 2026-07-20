#!/usr/bin/env python3
import importlib.util
import json
import os
import time
from pathlib import Path

from agent_utilities.security.persistence_privacy import sanitize_for_persistence

# Paths
SCRIPTS_DIR_VALUE = os.getenv("CODE_ENHANCER_SCRIPTS_DIR")
if not SCRIPTS_DIR_VALUE:
    raise SystemExit("CODE_ENHANCER_SCRIPTS_DIR must be configured")
SCRIPTS_DIR = Path(SCRIPTS_DIR_VALUE).expanduser().resolve()
PROJECT_DIR = Path(__file__).parent.resolve()
SPECIFY_DIR = PROJECT_DIR / ".specify"


def run_enhancer():
    print("🚀 Starting Code Enhancer on project:", PROJECT_DIR.name)

    analyzers = [
        ("analyze_project", "analyze_project"),
        ("audit_dependencies", "audit_dependencies"),
        ("analyze_codebase", "analyze_codebase"),
        ("analyze_security", "analyze_security"),
        ("analyze_tests", "analyze_tests"),
        ("audit_documentation", "audit_documentation"),
        ("analyze_architecture", "analyze_architecture"),
        ("trace_concepts", "trace_concepts"),
        ("run_linters", "run_linters"),
        ("run_precommit", "run_precommit"),
        ("run_tests", "run_tests"),
        ("analyze_directory_density", "analyze_directory_density"),
        ("analyze_ui", "analyze_ui"),
        ("analyze_version_sync", "analyze_version_sync"),
        ("audit_changelog", "audit_changelog"),
        ("grade_pytest", "grade_pytest"),
        ("scan_env_vars", "scan_env_vars"),
    ]

    results = []

    for module_name, func_name in analyzers:
        print(f"\n🔍 Running analyzer: {module_name}...")
        start_time = time.monotonic()
        try:
            # Dynamically load the analyzer script
            script_path = SCRIPTS_DIR / f"{module_name}.py"
            spec = importlib.util.spec_from_file_location(module_name, str(script_path))
            if spec is None or spec.loader is None:
                print(f"❌ Could not load spec for {module_name}")
                continue

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            func = getattr(module, func_name)

            # Execute with project dir
            result = func(str(PROJECT_DIR))
            elapsed = time.monotonic() - start_time
            print(
                f"✅ Finished {module_name} in {elapsed:.2f}s (Score: {result.get('score', 'N/A')}, Grade: {result.get('grade', 'N/A')})"
            )

            if result.get("score", 0) != -1:
                results.append(result)
        except Exception as e:
            elapsed = time.monotonic() - start_time
            print(f"Operation failed: {type(e).__name__}")
            results.append(
                {
                    "domain": module_name.replace("_", " ").title(),
                    "score": 0,
                    "grade": "F",
                    "findings": [f"Analysis error: {type(e).__name__[:200]}"],
                    "justifications": [],
                }
            )

    print("\n📝 Compiling final results...")
    safe_results, _privacy_report = sanitize_for_persistence(results)

    # Ensure specify dir exists
    SPECIFY_DIR.mkdir(parents=True, exist_ok=True)

    # Save raw json results
    results_json_path = SPECIFY_DIR / "results.json"
    results_json_path.write_text(json.dumps(safe_results, indent=2), encoding="utf-8")
    print("💾 Sanitized results saved successfully")

    # Load and run generate_report
    try:
        report_spec = importlib.util.spec_from_file_location(
            "generate_report", str(SCRIPTS_DIR / "generate_report.py")
        )
        if report_spec is None or report_spec.loader is None:
            raise Exception("Could not load generate_report spec")
        report_module = importlib.util.module_from_spec(report_spec)
        report_spec.loader.exec_module(report_module)

        report_md_path = SPECIFY_DIR / "reports" / "code_enhancement_report.md"
        report_md_path.parent.mkdir(parents=True, exist_ok=True)

        report_module.generate_report(
            safe_results,
            project_name=PROJECT_DIR.name,
            output_path=str(report_md_path),
        )
        print("📄 Prettified report written successfully")
    except Exception as e:
        print(f"Operation failed: {type(e).__name__}")

    # Load and run generate_sdd_handoff
    try:
        sdd_spec = importlib.util.spec_from_file_location(
            "generate_sdd_handoff", str(SCRIPTS_DIR / "generate_sdd_handoff.py")
        )
        if sdd_spec is None or sdd_spec.loader is None:
            raise Exception("Could not load generate_sdd_handoff spec")
        sdd_module = importlib.util.module_from_spec(sdd_spec)
        sdd_spec.loader.exec_module(sdd_module)

        sdd_module.generate_sdd_handoff(
            results, project_name=PROJECT_DIR.name, output_dir=str(PROJECT_DIR)
        )
        print("🎯 SDD handoff successfully written to: .specify/specs/")
    except Exception as e:
        print(f"Operation failed: {type(e).__name__}")


if __name__ == "__main__":
    run_enhancer()
