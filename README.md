# Enhancing Image Brightness without Paired Data: A Fast No-Reference Deep Learning Method (EIB-FNDL)

This work presents an improvement on the ZeroDCE++ model by simplifying the network architecture, resulting in faster performance. Our code is based on the original ZeroDCE++ implementation available on GitHub (link: "https://github.com/Li-Chongyi/Zero-DCE_extension"), which we modified to develop our proposed method. Additionally, we have cited the ZeroDCE++ paper in our work.



✌If you use this code, please cite our paper. Please hit the star at the top-right corner. Thanks!



# Pytorch
We used Pytorch for the implementation of our code

## Requirements
1. Python 3.9 
2. extorch 2.3.1+cu121
3. opencv 4.10.0
4. torchvision 0.18.1+cu121
5. cuda 12.1
6. PIL 10.2.0

Only a basic environment setup is needed to implement our method.


### Folder structure
Download all files from our repo
The following shows the basic folder structure.
```

├── data
│   ├── test_data 
│   └── train_data 
├── lowlight_test.py # testing code
├── lowlight_train.py # training code
├── model.py # the proposed network
├── dataloader.py
├── Metric
│   ├── akh_brisque.py
│   ├── BRISQUE.py
│   ├── eff.py
│   ├── ref.py
│   ├── SPAQ.py
├── model_parameter
│   ├── Epoch99.pth #  A pre-trained model (Epoch99.pth)
├── snapshots_epochs
```


The metric file is adapted from "https://github.com/ShenZheng2000/LLIE_Survey." We have added `akh_brisque.py` to provide an alternative if BRISQUE does not function properly. Minor adjustments were made to other files to enable our system to perform specific assessments.


### Test: 
Locate the result path in the code and modify it to save results to a specific path on your system.

#### Test dataset:
* NPE, LIME, MEF, DICM, VV [[link](https://drive.google.com/drive/folders/1lp6m5JE3kf3M66Dicbx5wSnvhxt90V4T)]

* SICE [[link](https://github.com/csjcai/SICE)]

* LOL [[link](https://daooshee.github.io/BMVC2018website/)]

* LOL-V2 [[link](https://github.com/flyywh/CVPR-2020-Semi-Low-Light)]

* SICE_Grad and SICE_Mix [[Download](https://drive.google.com/file/d/1gii4AEyyPp_kagfa7TyugnNPvUhkX84x/view?usp=sharing)]

cd EIB-FNDL
```
python lowlight_test.py 
```

### Train: 

#### Train dataset:
link: "https://github.com/Li-Chongyi/Zero-DCE_extension/tree/main/Zero-DCE%2B%2B/data"


cd EIB-FNDL
Locate the `train_data` path in the code and modify it to match the location of your files.


```
python lowlight_train.py 
```

##  License
The code is made available for academic research purpose only. Under Attribution-NonCommercial 4.0 International License.




## Contact
If you have any questions, please contact Alireza Khajehvandi at a.khajehvandi.pro@gmail.com.
