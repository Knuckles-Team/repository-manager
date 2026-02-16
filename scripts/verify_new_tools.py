
import os
import sys
import logging
from repository_manager.repository_manager import Git

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verification")

print(f"Git class loaded from: {sys.modules['repository_manager.repository_manager'].__file__}")

def verify_tools():
    workspace = os.path.abspath(os.path.dirname(__file__))
    git = Git(workspace=workspace, is_mcp_server=False)

    print(f"Workspace: {workspace}")

    # 1. Create a dummy file for testing
    test_file = "test_verification.txt"
    test_path = os.path.join(workspace, test_file)
    with open(test_path, "w") as f:
        f.write("Hello World\nThis is a test file for verification.\nLine 3: Search me.\nLine 4: Replace me.\n")
    print(f"Created test file: {test_path}")

    try:
        # 2. Test search_codebase with absolute path
        print("\n--- Testing search_codebase (absolute path) ---")
        search_result = git.search_codebase(query="Search me", path=workspace)
        if search_result.status == "success" and "Search me" in search_result.data:
            print("PASS: search_codebase (absolute) found the string.")
        else:
            print(f"FAIL: search_codebase (absolute) failed. Result: {search_result}")

        # 2b. Test search_codebase with recursive default (CWD)
        # We need to change CWD to workspace for this test to work naturally if we don't pass path
        cwd_original = os.getcwd()
        os.chdir(workspace)
        print("\n--- Testing search_codebase (CWD default) ---")
        try:
            search_result_cwd = git.search_codebase(query="Search me", path=None)
            if search_result_cwd.status == "success" and "Search me" in search_result_cwd.data:
                print("PASS: search_codebase (CWD) found the string.")
            else:
                print(f"FAIL: search_codebase (CWD) failed. Result: {search_result_cwd}")
        finally:
            os.chdir(cwd_original)

        # 3. Test find_files
        print("\n--- Testing find_files ---")
        find_result = git.find_files(name_pattern="test_verification.txt", path=workspace)
        if find_result.status == "success" and "test_verification.txt" in find_result.data:
            print("PASS: find_files found the file.")
        else:
            print(f"FAIL: find_files failed. Result: {find_result}")

        # 4. Test read_file with relative path
        print("\n--- Testing read_file (relative) ---")
        # Relative to workspace since we initialized Git with workspace
        # But wait, read_file in Git class now takes absolute or relative to CWD
        # So we should pass absolute path to be safe, or relative to where script is running
        read_result = git.read_file(path=test_path, start_line=1, end_line=2)
        if read_result.status == "success" and "Hello World" in read_result.data:
            print("PASS: read_file read correctly.")
        else:
            print(f"FAIL: read_file failed. Result: {read_result}")

        # 5. Test replace_in_file
        print("\n--- Testing replace_in_file ---")
        replace_result = git.replace_in_file(path=test_path, target_content="Line 4: Replace me.", replacement_content="Line 4: Replaced successfully.")
        if replace_result.status == "success":
            print("PASS: replace_in_file reported success.")
            # Verify content
            with open(test_path, "r") as f:
                content = f.read()
            if "Line 4: Replaced successfully." in content:
                print("PASS: Content verification succeeded.")
            else:
                print(f"FAIL: Content verification failed. Content: {content}")
        else:
            print(f"FAIL: replace_in_file failed. Result: {replace_result}")

    finally:
        # Cleanup
        if os.path.exists(test_path):
            os.remove(test_path)
            print(f"\nCleaned up test file: {test_path}")

if __name__ == "__main__":
    verify_tools()
