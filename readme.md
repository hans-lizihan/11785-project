# 11785 Group Project

## Dependencies

Install dependencies

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

to use it on google colab,

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
