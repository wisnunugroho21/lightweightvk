#!/usr/bin/env python3

import os
import shutil
from pathlib import Path

base_dir=os.getcwd()
lib_dir=os.path.join(base_dir, "src", "solarsystem")

shutil.move(os.path.join(lib_dir, "ThinkstockPhotos-517462556.web_.jpg"), os.path.join(lib_dir, "768_sun_corona.jpg"))
