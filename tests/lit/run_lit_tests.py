#!/usr/bin/env python3
"""
Lightweight golden test runner matching FileCheck semantics for *.dlc tests.
"""

import os
import re
import subprocess
import sys

def run_test(test_path: str, dlc_opt_path: str) -> bool:
    with open(test_path, "r") as f:
        lines = f.readlines()

    run_cmd = None
    check_patterns = []

    for line in lines:
        line_str = line.strip()
        if line_str.startswith("// RUN:"):
            cmd_template = line_str[7:].strip()
            # Substitute %dlc-opt and %s
            run_cmd = cmd_template.replace("%dlc-opt", dlc_opt_path).replace("%s", test_path)
            # Remove pipe to FileCheck if present
            if "|" in run_cmd:
                run_cmd = run_cmd.split("|")[0].strip()
        elif line_str.startswith("// CHECK:"):
            check_patterns.append(line_str[9:].strip())

    if not run_cmd:
        print(f"[SKIP] No RUN command found in {test_path}")
        return True

    res = subprocess.run(run_cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[FAIL] {test_path} command failed with code {res.returncode}")
        print("Stderr:\n", res.stderr)
        return False

    output = res.stdout
    cur_pos = 0
    for pattern in check_patterns:
        # Simplify whitespace matching
        regex_pattern = re.escape(pattern)
        regex_pattern = re.sub(r"\\\s+", r"\\s+", regex_pattern)
        match = re.search(regex_pattern, output[cur_pos:])
        if not match:
            print(f"[FAIL] {test_path}")
            print(f"Expected CHECK pattern: {pattern}")
            print(f"Remaining output:\n{output[cur_pos:]}")
            return False
        cur_pos += match.end()

    print(f"[PASS] {os.path.basename(test_path)}")
    return True

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    dlc_opt_path = os.path.abspath(os.path.join(root_dir, "../../build/dlc-opt"))

    if not os.path.exists(dlc_opt_path):
        print(f"Error: dlc-opt binary not found at {dlc_opt_path}. Please build the project first.")
        sys.exit(1)

    dlc_files = [
        os.path.join(root_dir, f)
        for f in os.listdir(root_dir)
        if f.endswith(".dlc")
    ]

    all_passed = True
    print(f"Running {len(dlc_files)} lit golden IR test(s)...")
    for f in sorted(dlc_files):
        if not run_test(f, dlc_opt_path):
            all_passed = False

    if all_passed:
        print("All Lit Golden IR Tests Passed Successfully!")
        sys.exit(0)
    else:
        print("Some Lit Tests Failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
