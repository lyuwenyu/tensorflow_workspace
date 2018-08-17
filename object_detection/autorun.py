import os
import re
import time
import multiprocessing

f = re.compile('\d.*\d')

while True:
	
	res = os.popen('nvidia-smi --query-gpu=memory.free --format=csv').readlines()
	mem = int(f.findall(res[-1])[0])

	if mem > 7000:
		print('large than 7000.', mem)
		os.system('python train.py --logtostderr --train_dir=./outputs/ssd_mobil_v4/train --pipeline_config_path=./config/ssd_mobilenet_v1_coco_fpn.config')
		break
	else:
		pass
		print(mem)
		
	time.sleep(600)
