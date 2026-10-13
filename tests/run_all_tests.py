"""
Master Test Suite Runner for Zero-Trust Local Code RAG Engine.
Executes all unit, integration, authentication, and offline security test suites.
"""
import os
import sys
import subprocess

test_files = [
    "tests/test_strict_verification.py",
    "tests/test_auth.py",
    "tests/test_offline_embedding.py",
    "tests/test_chunker.py",
    "tests/test_chunker_verification_integration.py",
    "tests/test_reindex_stale_vectors.py",
    "tests/test_git_host_allowlist.py"
]

def run_suite():
    os.environ["RAG_AUTH_TOKEN"] = "test-runner-token-2026"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    passed = 0
    failed = 0
    results = []

    print("=" * 80)
    print(" ZERO-TRUST LOCAL CODE RAG: EXECUTING FULL MASTER TEST SUITE")
    print("=" * 80)

    for test_rel in test_files:
        test_path = os.path.join(base_dir, test_rel.replace("/", os.sep))
        print(f"\n--> RUNNING: {test_rel} ...")
        proc = subprocess.run(
            [sys.executable, "-u", test_path],
            cwd=base_dir,
            capture_output=True,
            text=True
        )
        if proc.returncode == 0:
            passed += 1
            results.append((test_rel, "PASSED", proc.stdout.strip()))
            print(f"    [PASSED] {test_rel}")
            for line in proc.stdout.strip().splitlines():
                if line.startswith("PASS:") or "PASSED" in line:
                    print(f"      {line}")
        else:
            failed += 1
            results.append((test_rel, "FAILED", proc.stderr.strip() or proc.stdout.strip()))
            print(f"    [FAILED] {test_rel}")
            print(proc.stderr.strip() or proc.stdout.strip())

    print("\n" + "=" * 80)
    print(f" SUMMARY: Total Test Suites: {len(test_files)} | Passed: {passed} | Failed: {failed}")
    print("=" * 80)
    
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    run_suite()
