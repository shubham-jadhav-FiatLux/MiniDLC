import os
import lit.formats

config.name = "MiniDLC-Lit"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".dlc"]
config.test_source_root = os.path.dirname(__file__)

# Locate dlc-opt binary
build_dir = os.path.abspath(os.path.join(config.test_source_root, "../../build"))
dlc_opt_path = os.path.join(build_dir, "dlc-opt")

config.substitutions.append(("%dlc-opt", dlc_opt_path))
