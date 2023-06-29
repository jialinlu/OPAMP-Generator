import re
import os
import numpy as np

param = open('./param', 'r')
param_lines = param.readlines()

temp = open('./ocnScript_temp.ocn','r')
temp_lines = temp.readlines()

ocn_script = open('./oceanScript_opamp.ocn','w')

for i in range(7):
    ocn_script.writelines(temp_lines[i])

for i in range(len(param_lines)):
    stripped = param_lines[i]
    match    = re.search(r'^\S+\s+(\S+)\s+\S+\s+(\S+)$',stripped)
    ocn_script.writelines('desVar( '+'"'+match.groups()[0]+'" '+match.groups()[1]+' )\n')

for i in range(7,33):
    ocn_script.writelines(temp_lines[i])

param.close()
temp.close()
ocn_script.close()
