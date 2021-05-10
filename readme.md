# 11785 Group Project

## Code Set 1: CartoonGAN.py
In this code set, we modified the CartoonGAN model to incorporate ResNet, LeakyReLU, and more normalization functions.

### Install dependencies

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### To Run the Code on Google Colab

```
from google.colab import drive
drive.mount('/content/gdrive')

!git clone --single-branch --branch main https://<your_token>@github.com/hans-lizihan/11785-project.git
%cd /content/11785-project
!python CartoonGAN.py 
--batch_size 16 \
--test_image_path dataset/test \
--model_save_path=/content/gdrive/MyDrive/11785/project/checkpoints_one_punch/ \
--animation_image_dir dataset/TgtDataSet/One-Punch\ Man \
--edge_smoothed_image_dir dataset/TgtDataSet/One-Punch\ Man_smooth
```

## Code Set 2: CartoonGAN_PreProcess.ipynb
In this code set, we modified the CartoonGAN model to pre-process the source image before it is feed into the network. In the pre-processing function, the image is slightly blurred, and the edges are intensified to make the image more similar to cartoon images even before feeding the image into the network. The idea is that pre-processing the image makes the network easier to convert the data into cartoon style by reducing its work.

### Install dependencies

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### To Run the Code on Google Colab
Import the ipynb file into the colab. Then run it.

## Code Set 3: CartoonGAN_Experiment.ipynb
In this code set, we modified the CartoonGAN model to pre-process the source image before it is feed into the network. We are doing pre-processing for the same reason as above, but the pre-processing is done in another way. We first train a GAN model transforming cartoon images into real-world pictures. Then use the (trained) generator to transform target data for CartoonGan training.

### Install dependencies

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### To Run the Code on Google Colab
Import the ipynb file into the colab. Then run it.


## Reference
Our Project referred to the following repository as the baseline CartoonGAN implementation.
https://github.com/znxlwm/pytorch-CartoonGAN