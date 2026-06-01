from repository_manager.models import (
    ProjectResult,
    ValidationCategory,
    _build_summary_md,
    _clean_and_truncate_error_lines,
    _extract_error_lines,
    _extract_pytest_failures,
)


def test_extract_pytest_failures_with_long_traceback():
    # Construct a mock pytest output with a very long traceback (more than 15 lines)
    mock_output = (
        "============================= test session starts =============================\n"
        "collected 2 items\n"
        "\n"
        "=================================== FAILURES ===================================\n"
        "___ test_foo_failure ___\n"
        "def test_foo_failure():\n"
        "    a = 1\n"
        "    b = 2\n"
        "    c = 3\n"
        "    d = 4\n"
        "    e = 5\n"
        "    f = 6\n"
        "    g = 7\n"
        "    h = 8\n"
        "    i = 9\n"
        "    j = 10\n"
        "    k = 11\n"
        "    l = 12\n"
        "    m = 13\n"
        "    n = 14\n"
        "    o = 15\n"
        "    p = 16\n"
        "    assert a == p\n"
        "E   AssertionError: assert 1 == 16\n"
        "\n"
        "=========================== short test summary info ============================\n"
        "FAILED test_foo_failure - AssertionError: assert 1 == 16\n"
        "======================== 1 failed, 1 passed in 0.12s =========================\n"
    )

    result = _extract_pytest_failures(mock_output)

    # Check that we extracted failures and short test summary
    assert "- **pytest Failures**" in result
    assert "  **test_foo_failure**" in result
    assert "- **pytest Short Summary**" in result
    assert "  FAILED test_foo_failure - AssertionError: assert 1 == 16" in result
    assert "  **Result:** 1 failed, 1 passed in 0.12s" in result

    # The traceback block has 1 header + 20 lines = 21 lines.
    # It should be truncated to exactly 15 lines plus the info line.
    failures_index = result.index("  **test_foo_failure**")
    # Verify that the traceback is truncated to 15 lines
    # We find the next section boundary or end of traceback
    traceback_lines = []
    for line in result[failures_index + 1 :]:
        if line.startswith("- **") or line.startswith("  **Result:"):
            break
        traceback_lines.append(line)

    assert len(traceback_lines) == 15  # 14 lines of traceback + 1 info line
    assert "more lines of traceback" in traceback_lines[-1]


def test_extract_error_lines_delegation():
    # If is_pytest is True, it should use _extract_pytest_failures
    pytest_out = "=== FAILURES ===\n___ test_err ___\nE AssertionError\n=== short test summary ===\nFAILED test_err"
    res1 = _extract_error_lines(pytest_out, is_pytest=True)
    assert "- **pytest Failures**" in res1
    assert "  **test_err**" in res1

    # If is_pytest is False, it should use normal pre-commit hook parsing
    pc_out = "Hook...........Failed\n- hook_name\nexit code: 1"
    res2 = _extract_error_lines(pc_out, is_pytest=False)
    assert "- **Hook** — Failed" in res2


def test_clean_and_truncate_error_lines_pytest():
    # Test that _clean_and_truncate_error_lines correctly handles pytest block boundaries
    error_lines = [
        "- **pytest Failures**",
        "  **test_a**",
        "  line 1",
        "  line 2",
        "- **pytest Short Summary**",
        "  FAILED test_a",
    ]
    formatted = _clean_and_truncate_error_lines(error_lines, max_lines_per_block=1)
    # Both "- **pytest Failures**" and "- **pytest Short Summary**" should trigger a block boundary
    # and get truncated to max_lines_per_block=1 if their blocks are longer
    assert "- **pytest Failures**" in formatted
    assert "  **test_a**" in formatted
    assert "  > ℹ️ ... and 2 more lines of output." in formatted
    assert "- **pytest Short Summary**" in formatted
    assert "  FAILED test_a" in formatted


def test_build_summary_md_with_pytest():
    # Mock data for categories and failures
    categories = [
        ValidationCategory(
            name="Pytest Suite",
            total=1,
            failure_count=1,
            failures=[
                ProjectResult(
                    project="test-project",
                    message="pytest test run failed",
                    output=(
                        "=== FAILURES ===\n"
                        "___ test_failed ___\n"
                        "E AssertionError: 1 == 2\n"
                        "=== short test summary ===\n"
                        "FAILED test_failed\n"
                        "=== 1 failed in 0.05s ===\n"
                    ),
                    command="pytest --cov=test_project tests/",
                )
            ],
        )
    ]

    summary = _build_summary_md(
        timestamp="2026-05-26 12:00:00",
        total=1,
        success_count=0,
        failure_count=1,
        categories=categories,
        failed_projects=["test-project"],
    )

    summary_str = "\n".join(summary)
    # Check that it extracted the pytest failures correctly instead of falling back to raw output
    assert "### ❌ test-project" in summary_str
    assert "**Pytest Suite**" in summary_str
    assert "- **pytest Failures**" in summary_str
    assert "  **test_failed**" in summary_str
    assert "  **Result:** 1 failed in 0.05s" in summary_str
    # Verify it is not showing raw start session info
    assert "test session starts" not in summary_str
