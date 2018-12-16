# life is short, you need use python to create something!
# author    TuringEmmy
# time      12/7/18 7:02 PM
# project   DeepLearingStudy

import os
import re
import shutil

if not os.path.exists("/mnt/hgfs/WorkSpace/data/poem_generator/dataset/tang"):
    os.mkdir("tang")
if not os.path.exists("/mnt/hgfs/WorkSpace/data/poem_generator/dataset/song"):
    os.mkdir("song")

for file in os.listdir("."):
    if os.path.isfile(os.path.join("./", file)) and re.match('(.*)(\.)(json)', file) != None:
        shutil.copyfile(os.path.join("./", file), os.path.join("./", file.split(".")[1], file))
