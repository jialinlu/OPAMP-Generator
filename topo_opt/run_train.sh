# training
python -W ignore train.py --model DVAE_test2 --data_file dataset_withoutY_1w --batch_size 16 --save-appendix _1w --lr 1e-5 --gpu 1

# testing
# python -W ignore train.py --only-test --model DVAE_test2 --data_file dataset_withoutY_1w --trainSet_size 9000 --save-appendix _4test --load_model_path _nz10_1w --load_model_name 200 | tee test.log

