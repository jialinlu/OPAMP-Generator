import os
import re
import autograd.numpy as np

total_gain = 0
param = open('./param', 'r')
lines  = param.readlines()
for line in lines:
    stripped = line.strip()
    match = re.search(r'\.param\s(g\S+)\s+\S+\s+(\S+)$', stripped)
    if match:
        total_gain = total_gain + abs(float(match.groups()[1]))

res = open('./result.po', 'w')
sim = os.path.exists('./sim.out')

if sim== True:
    print('simulation sucessed!')
    out       = open('./sim.out','r')
    lines     = out.readlines()
    stripped1 = lines[2].strip()
    match1    = re.search(r'^(\S+)\s+(\S+)\s+(\S+)$', stripped1)
    if match1:
        gain      = float(match1.groups()[0])
        pm        = float(match1.groups()[1])
        gbw       = float(match1.groups()[2])
        out.close()
        power     = total_gain * 1.2 * 1e3 / 20
        ## run1
        ## prb_constr0     gain      >= 85  dB    ##
        ## prb_constr1     pm        >= 55        ##
        ## prb_constr2     gbw       >= 0.7 MHz   ##
        ## prb_constr3     power     <= 250 uW    ##
        ## goal            fom_s     maximize     ##
        
        ## run2
        ## prb_constr0     gain      >= 90  dB    ##
        ## prb_constr1     pm        >= 50        ##
        ## prb_constr2     gbw       >= 0.6 MHz   ##
        ## prb_constr3     power     <= 270 uW    ##
        ## goal            fom_s     maximize     ##

        prb_constr0 = -1*gain+85
        prb_constr1 = -1*pm+55
        prb_constr2 = -1*(gbw/(1e6))+0.7
        prb_constr3 = power - 250
        fom_s       = (gbw/1e6)*(10*1e3) / (power*1e-3)
        goal        = -fom_s
        res.writelines(str(goal)+' '+str(prb_constr0)+' '+str(prb_constr1)+' '+str(prb_constr2)+' '+str(prb_constr3)+'\n')
        res.close()
    else:
        print('simulation error!')
        goal        = -0.1 #(np.random.rand()+1)*100
        prb_constr0 = 0.1 #(np.random.rand()+1)*-10
        prb_constr1 = 0.1 #(np.random.rand()+1)*-10
        prb_constr2 = 0.1 #(np.random.rand()+1)*0.2
        prb_constr3 = 50 #(np.random.rand()+1)*50
        res.writelines(str(goal)+' '+str(prb_constr0)+' '+str(prb_constr1)+' '+str(prb_constr2)+' '+str(prb_constr3)+'\n')
        res.close()
else:
    print('simulation failed!')
    goal        = -0.1 #(np.random.rand()+1)*100
    prb_constr0 = 0.1 #(np.random.rand()+1)*-10
    prb_constr1 = 0.1 #(np.random.rand()+1)*-10
    prb_constr2 = 0.1 #(np.random.rand()+1)*0.2
    prb_constr3 = 50 #(np.random.rand()+1)*50
    res.writelines(str(goal)+' '+str(prb_constr0)+' '+str(prb_constr1)+' '+str(prb_constr2)+' '+str(prb_constr3)+'\n')
    res.close()
