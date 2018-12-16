# life is short, you need use python to create something!
# author    TuringEmmy
# time      12/7/18 7:02 PM
# project   DeepLearingStudy

import json
import os
import re

typeName = "poetrySong"

titleField = "title"
contentField = "paragraphs"
authorField = "author"

jsonPath = "./" + typeName
saveFile = open(os.path.join(jsonPath, typeName + ".txt"), "w")

for file in os.listdir(jsonPath):
    if os.path.isfile(os.path.join(jsonPath,file)) and re.match('(.*)(\.)(json)', file) != None:
        print("processing file: %s" % file)
        poems = json.load(open(os.path.join(jsonPath,file), "r"))
        for singlePoem in poems:
            content = "".join(singlePoem[contentField])
            title = singlePoem[titleField]
            author = singlePoem[authorField]
            saveFile.write(title + "::" + author + "::" + content + "\n")

saveFile.close()