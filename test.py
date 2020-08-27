from datetime import datetime
import os
from shutil import copy2

mydir = os.path.join(os.getcwd(), datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(mydir)

src = os.path.join(os.getcwd(),'reload.py')
dst = os.path.join(os.getcwd(),mydir)
copy2(src,dst)

os.chdir(dst)
os.system('/usr/bin/python3 reload.py')

