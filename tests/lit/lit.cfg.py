import os
import shutil
import subprocess

import lit.formats

config.name = "MiniDLC"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".dlc"]
config.test_source_root = os.path.dirname(__file__)

repo_root = os.path.abspath(os.path.join(config.test_source_root, "..", ".."))

# dlc-opt: --param dlc_opt=/path/to/dlc-opt, else <repo>/build/dlc-opt
dlc_opt = lit_config.params.get("dlc_opt", os.path.join(repo_root, "build", "dlc-opt"))
if not os.path.exists(dlc_opt):
    lit_config.fatal(f"dlc-opt not found at {dlc_opt}; build first or pass --param dlc_opt=...")
config.substitutions.append(("%dlc-opt", dlc_opt))


def find_filecheck():
    explicit = lit_config.params.get("filecheck") or os.environ.get("FILECHECK")
    if explicit:
        return explicit
    on_path = shutil.which("FileCheck")
    if on_path:
        return on_path
    try:
        bindir = subprocess.check_output(["llvm-config", "--bindir"], text=True).strip()
        candidate = os.path.join(bindir, "FileCheck")
        if os.path.exists(candidate):
            return candidate
    except (OSError, subprocess.CalledProcessError):
        pass
    return None


filecheck = find_filecheck()
if not filecheck:
    lit_config.fatal("LLVM FileCheck not found; install LLVM tools or pass --param filecheck=...")
config.environment["PATH"] = os.pathsep.join([os.path.dirname(filecheck), config.environment.get("PATH", "")])
