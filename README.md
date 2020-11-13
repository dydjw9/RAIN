
## RAIN: A Simple Approach for Robust and Accurate Image Classication Networks

This is the codes for the paper RAIN: A Simple Approach for Robust and Accurate Image Classication Networks. 



### Requirement
 * numpy                     1.17.2+
 * python                    3.6.9+
 * pytorch                   1.4.0+
 * torchvision               0.5.0+
 * eagerpy                   0.27.0+
 * foolbox                   3.0.0+
 * pillow                    6.2.0+
 * advertorch                0.2.2+

### Folder Structure
The folder structure is as the following shows. 
The pretrained model and validation datas can be found in the [Pretrained Model](https://drive.google.com/file/d/1WaCI29lInIEs4OgDl9mSSP8US__U7hvh/view?usp=sharing).

```
RAIN
│   README.md 
│
└───codes
│   │   all sources codes 
│   │   
│   │
│   └───stl10
│       │   codes for stl10 dataset
│   
│   └───imagenet
│       │   codes for imagenet dataset
│   
│   └───ad_attack
│       │   codes for FGSM,PGD,ZOO attack
│  
│   └───black_box_attack
│       │   codes for NES attack
└───datas
│   └───fullimagenet
│       │   imagenet validation set
│   └───stl10
│       │   stl10 validation set
└───checkpoints
│   └───fullimagenet
│       │   imagenet pretrained model
│   └───stl10
│       │   stl10 pretrained model
│   └───EDSR
│       │   EDSR pretrained model
└───results 

```

### Usage
The codes for stl10 and ImageNet follows the same organization, You may use the shell script "run_cnn.sh" in the subfolder stl10 and imagenet to start finetuning or test. 
  

To Finetune

	python main_cnn.py --exp_name embed_model  --isTrain True  --resume False --weight_decay 5e-4 --gamma 0.2 --device_ids 0_1

The running time for finetuning takes about 1 hours in STL10, and about 10 hours in ImageNet. The GPU we used are two RTX 2080TI. 

To Test

	python main_cnn.py --isTrain False --device_ids 0_1 --dir_model ../../checkpoints/fullimgnet/model/model.pth   --logging_file ImgNet.txt --robustness_evaluation_number 5000
	
Since the robustness evaluation under iterative attacks methods takes from 30mins(FGSM) to 12 hours(ZOO), you may change the "robustness_evaluation_number" to a smaller number for faster results. Because our proposed method is randomization-based method, smaller testing number may make the results a bit different from our reported results. 

In the file "main_cnn.py" line 219, you may change the codes to evaluate the robustness under specific attack methods. For example, 

	acc6 = 0 if False else CW_attack(NUM,model,val_loader_128,1,iters = 40,isnorm=False)
	
The default setting is to test all white-box attacks. 

For the EAD attack in ImageNet dataset, it may consumes enourmous GPU memroy, you may change the batch size in line 221 of the file "main_cnn.py" to save memory. The default number is 32. 

  

