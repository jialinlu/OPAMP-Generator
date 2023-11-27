## OPAMP_Generator

OPAMP Generator consists of three modules: a topology optimization module, a topology to transistor-level circuit mapping module and an automatic layout generator. 
This repo contains the topology optimizaiton algorithm and transistor-level circuit mapping module. The ALIGN (https://github.com/ALIGN-analoglayout/ALIGN-public) and MAGLICAL (https://github.com/magical-eda/MAGICAL) tool can be adapted to generate layouts based on transistor-level circuits. 

### Demo of the generator

https://github.com/jialinlu/work_release/assets/43021674/baaba8f4-3ffb-4228-8020-19292d345b36

### Topology Optimization (topo_opt directory)

1. Before running the code, please make sure that the Python (recommended 3.6.13) environment has "torch" (recommended 1.9.1) and "pygraphviz" (recommended 1.6) libraries installed, as well as other common scientific computing libraries. The environment configuration can be found in the environment.yml file located in the root of this repository.

2. Use the provided dataset /topo_opt/dataset_withoutY_1w.txt or generate a new training dataset by
```bash
python -W ignore get_dataset.py
```

3. The main training program of VGAE model is train.py, you can execute the training by
```bash
source run_train.sh
```
or directly execute the command: 
```bash
python -W ignore train.py --model DVAE_test2 --data_file dataset_ withoutY_1w --batch_size 16 --save-appendix _1w --lr 1e-5 --gpu 1
```

4. The main program for performing topology optimization is optimization_bfgs.py, which can be executed with
```bash
source run_opt.sh
```
or directly with the command: 
```bash
python -W ignore optimization_bfgs.py --iteration 30 --save- appendix _Bfgs_exp1_bound15 --nz 10 --which_gp sklearn --load_model_path _nz10_1w_demo --load_model_name 400 --emb_bound 15 --bfgs_time 200 -- samping_ratio 0.0001 --gpu 3
```
Here, we use HEBO as the sizing algorithm under a fixed topology.

5. The training results of the VGAE model will be saved in the results folder, under which there is already a trained demo; the results of the topology optimization will be saved in the opt_results and tmp_circuits folders, and the circuit demo can be found in /schematic_mapping/behv_circuits.

### Automatic Mapping of Transistor-level Circuits (schematic_mapping directory)

1. Make sure that the libraries "hebo" and "pymoo" are installed in the Python (recommended 3.6.13) environment before running the code, as well as other common scientific computing libraries;

2. This program is suitable for tsmc18 process, the lookup table needed for gmid method is given under tsmc18rf_1p6m_lut, but the corresponding transistor model file (rf018.scs) should be copied to this directory before running;

3. The main program is main.py, and the running command
```bash
python -W ignore main.py --TITLE --MAP --SIM --OUT --REOPT1 --intrinsic_gain_times 1
```

4. The behavioral-level circuit models to be mapped need to be placed in the behv_circuits directory, and the mapped transistor-level circuits will be generated under mapped_circuits. The three circuits shown in the paper have been placed in the corresponding directory.
Note: In the netlist of behavioral-level models and transistor-level circuits, the device sizes are all attempts of the last iteration of the BO program, so a direct simulation of them may not correspond to the optimal results shown in the paper.

### Generated OPAMPs for Benchmarks (testbench)

The behv_circuits folder in testbench directory contains 32 behavior-level three-stage op-amp circuits searched by the automatic topology optimization program, which can be automatically transformed into transistor-level circuit netlists by the schematic_mapping program and optimized for sizing.

After testing, all 32 behavioral-level circuits in the .18 process can achieve the following circuit specifications after transistor-level mapping:

       prb_constr0 gain >= 85 dB    

       prb_constr1 pm >= 55        

       prb_constr2 gbw >= 0.7 MHz   

       prb_constr3 power <= 250 uW    

       goal fom_s maximize     

## Notes:

1. Topology optimization and transistor-level circuit mapping both need to call Cadence Spectre simulator simulation, please ensure that the corresponding software is configured in the environment variable, the recommended ic615 version;

2. VGAE neural network training requires GPU and cuda environment, CUDA 11.0 is recommended;

3. The transistor mapping code in testbench is the same as under the schematic_mapping path, but the automatic calculation of R and C dimensions in the circuit is not yet partially supported. If you need to dock the automatic generation process of the circuit layout, you need to determine the dimensions of R and C manually according to the process.

## References
Please cite the following papers if this repo is used in your project. 

1. J. Lu, L. Lei, F. Yang, C. Yan and X. Zeng, "Automated Compensation Scheme Design for Operational Amplifier via Bayesian Optimization," 2021 58th ACM/IEEE Design Automation Conference (DAC), San Francisco, CA, USA, 2021, pp. 517-522, doi: 10.1109/DAC18074.2021.9586306.

2. J. Lu, L. Lei, F. Yang, L. Shang and X. Zeng, "Topology Optimization of Operational Amplifier in Continuous Space via Graph Embedding," 2022 Design, Automation & Test in Europe Conference & Exhibition (DATE), Antwerp, Belgium, 2022, pp. 142-147, doi: 10.23919/DATE54114.2022.9774676.

3. J. Lu, L. Lei, J. Huang, F. Yang, L. Shang and X. Zeng, "Automatic Op-Amp Generation from Specification to Layout", IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, vol. 42, no. 12, pp. 4378-4390, Dec. 2023, doi: 10.1109/TCAD.2023.3296374.
