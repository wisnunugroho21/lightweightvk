#!/usr/bin/env python3

import os
import shutil
import sys
from pathlib import Path

base_dir = os.getcwd()
lib_dir = os.path.join(base_dir, "src", "glslang")

# https://github.com/KhronosGroup/glslang/pull/3514
patch_path = os.path.join(base_dir, "patches", "glslang_crash_arm64_v8a.patch")

os.chdir(lib_dir)

os.system(
    'git apply {}'.format(patch_path)
)
