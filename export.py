# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2023/7/27 10:35
# @Author : liumin
# @File : export.py

import shutil
from ultralytics import YOLO

for name in ['stringing_vi_cell', 'stringing_vi_hdpy','stringing_vi_hdqs','stringing_vi_dhd', 'stringing_vi_jb', 'stringing_vi_yw', 'stringing_vi_qj']:
    weight_path = 'runs/detect/' + name + '/weights/best.pt'
    new_weight_path = weight_path.replace('best.pt', name+'.pt')
    shutil.copyfile(weight_path, new_weight_path)
    # Load a model
    model = YOLO(new_weight_path)  # load an official model

    # Export the model
    model.export(format='openvino', ch=1)


