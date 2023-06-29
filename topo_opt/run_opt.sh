
source /apps/EDAs/ic615.cshrc

### use bfgs method
python -W ignore optimization_bfgs.py --iteration 30 --save-appendix _Bfgs_exp1_bound15 --nz 10 --which_gp sklearn --load_model_path _nz10_1w_demo --load_model_name 400 --emb_bound 15 --bfgs_time 200 --samping_ratio 0.0001 --gpu 3

