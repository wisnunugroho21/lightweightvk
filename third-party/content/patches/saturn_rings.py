#!/usr/bin/env python3

import os
import shutil
from pathlib import Path

base_dir=os.getcwd()
lib_dir=os.path.join(base_dir, "src", "solarsystem")

shutil.move(os.path.join(lib_dir, "d6xt63j-169c5f5a-9033-417c-a48a-496ec8d05f77.jpg"), os.path.join(lib_dir, "1k_saturn_rings.jpg"))
