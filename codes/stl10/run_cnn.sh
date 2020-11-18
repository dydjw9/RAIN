# finetuning
#python main_cnn.py --exp_name embed  --isTrain True --resume False --weight_decay 5e-4 --gamma 0.2 --device_ids 0_1_2 --end_epoch 250 --lr 0.0001 --milestones 150_175_225 

#test
python main_cnn.py --isTrain False --device_ids 0_1 --dir_model ../../checkpoints/stl10/model/model.pth  --logging_file STL10.txt --robustness_evaluation_number 5000

#EOT model 
python main_cnn.py --isTrain False --device_ids 0_1 --dir_model ../../checkpoints/stl10/model/EOT_model.pth  --logging_file STL10.txt --robustness_evaluation_number 5000


