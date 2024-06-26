
![frame0000](https://github.com/shani1610/hoi-data-processing/assets/56839113/a9afe38a-8dea-4fa1-934d-39c67410f719)

# HOI Data Processing 
human-object interaction video dataset processing
including object detection and object removal using inpaiting

# Installation

for visual AI lab usage
```
ssh israelov@133.15.42.60
```
(for first time need to do other kubectl commands)
```
kubectl exec -n israelov -it israelov-hoi-data-processing -- /bin/bash
```
(for first time only) I am using python 3.8.19, CUDA 11.2

```
conda create -n hoi-env python=3.8 
conda activate hoi-env
```
requirement file contains the requirments both for groundingDINO and ProPainter:
```pip install -r requirements.txt```

the corresponding PyTorch for me is 
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```
please pay attention to the CUDA version, we need the Pytorch corresponding to `nvcc --version` and not the driver's one `nvidia smi`

### ProPainter:

the inpainting is being done using (ProPainter)[https://github.com/sczhou/ProPainter], for object removal ProPainter requires the frames and the masks of the object to be removed. 

please clone the repository and replace the inference_propainter.py with inference_propainter_my.py 

### GroundingDino:

some of the code was adpted from roboflow notebook: Automated Dataset Annotation and Evaluation with Grounding DINO

```
git clone https://github.com/IDEA-Research/GroundingDINO.git
```
we use latest Grounding DINO model API that is not official yet so do this checkout
```
cd {HOME}/GroundingDINO
git checkout feature/more_compact_inference_api
```

in terminal run:
```
cd {HOME}
mkdir {HOME}/weights
cd {HOME}/weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

please do: 
```
cd GroundingDINO
pip install -q -e .
```

then some more 

```pip uninstall charset-normalizer```

```conda install -c conda-forge charset-normalizer```

```pip uninstall transformers```

```pip install transformers==4.28.0```

### Yolov9


# Usage

## extract files behave

I am using (BEHAVE dataset)[https://virtualhumans.mpi-inf.mpg.de/behave/
], full body human-object interaction dataset with multi-view RGBD frames and corresponding 3D SMPL and object fits along with the annotated contacts between them. please download Date01 sequences and extract it under a folder ./data/preprocessed_behave/

this script is extracting the RGB image from camera 0 and the object masks for each sequence (raw videos are available as well).
```
python ./sample/extract_files_behave.py
```
or
```
python ./sample/extract_files_behave.py --root_directory "./data/preprocessed_behave" --output_color "./data/images_for_gdino/" --output_inpaint "./data/images_for_ppainter/"
```

please be patient because it might take a while for the images to appear. 

## inpainting

```
cd ProPainter/
```

```
python inference_propainter.py -i "./../data/images_for_ppainter/Date01_Sub01_backpack_back/Date01_Sub01_backpack_back" -m "./../data/images_for_ppainter/Date01_Sub01_backpack_back/Date01_Sub01_backpack_back_mask_resized" -o "./../data/hoi_dataset/Date01_Sub01_backpack_back" --name_of_subject "Date01_Sub01_backpack_back" --save_frames_behave
```

```
python inference_propainter.py -i "./../data/images_for_ppainter/Date01_Sub01_basketball/Date01_Sub01_basketball" -m "./../data/images_for_ppainter/Date01_Sub01_basketball/Date01_Sub01_basketball_mask_resized" -o "./../data/hoi_dataset/" --name_of_subject "Date01_Sub01_basketball" --save_frames_behave
```

```
python inference_propainter.py -i "./../data/images_for_ppainter/Date01_Sub01_boxlarge_hand/Date01_Sub01_boxlarge_hand" -m "./../data/images_for_ppainter/Date01_Sub01_boxlarge_hand/Date01_Sub01_boxlarge_hand_mask_resized" -o "./../data/hoi_dataset/" --name_of_subject "Date01_Sub01_boxlarge_hand" --save_frames_behave
```


maybe can be written without the --name of subject so might consider checking it. 

## object detection:

from the hoi-data-processing folder run:

```
python create_dataset.py --frames_dir "./data/images_for_gdino/Date01_Sub01_backpack_back_color/" --object_name "backpack" --name_of_subject "Date01_Sub01_backpack_back" --annotation_output_folder "./data/hoi_dataset/Date01_Sub01_backpack_back/labels/"
```

```
python create_dataset.py --frames_dir "./data/images_for_gdino/Date01_Sub01_basketball_color/" --object_name "basketball" --name_of_subject "Date01_Sub01_basketball" --annotation_output_folder "./data/hoi_dataset/Date01_Sub01_basketball/labels/"
```

```
python create_dataset.py --frames_dir "./data/images_for_gdino/Date01_Sub01_boxlarge_hand_color/" --object_name "box" --name_of_subject "Date01_Sub01_boxlarge_hand" --annotation_output_folder "./data/hoi_dataset/Date01_Sub01_boxlarge_hand/labels/"
```

to check corresondenceL
```
python check_img_lab_corr.py --img_path "./data/hoi_dataset/Date01_Sub01_backpack_back/images/Date01_Sub01_backpack_back0022.jpg" --lab_path "./data/hoi_dataset/Date01_Sub01_backpack_back/labels/Date01_Sub01_backpack_back0022.txt"
```

### Data preperation for Yolo
```
python ./sample/dataset_for_yolo.py ./data/hoi_dataset ./yolov9/hoi_dataset
```
this will create this hirerchy: 
```bash
└── yolov9/
        └── hoi-dataset/
            ├── test/
            │   ├── images/
            │   │   ├── Date01_Sub01_backpack_back0008.jpg
            │   │   └── ...
            │   └── labels/
            │       ├── Date01_Sub01_backpack_back0008.txt
            │       └── ...
            ├── train/
            │   ├── images
            │   └── labels
            ├── valid/
            │   ├── images
            │   └── labels
            └── data.yaml
```            
the data.yaml contains this info:
```
names:
  0: backpack
  1: basketball
  2: box
nc: 3
test: hoi_dataset/train/images
train: hoi_dataset/train/images
val: hoi_dataset/valid/images
```

```
bash scripts/get_behave.sh
```

# Training 

## train-yolov9-object-detection

adapted from roboflow's notebook [train-yolov9-object-detection-on-custom-dataset.ipynb](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov9-object-detection-on-custom-dataset.ipynb#scrollTo=pixgo4qnjdoU)



## Installation

HOME is what you get by running pwd
was tested on Python 3.9.18, CUDA 11.8 (check using nvcc --version)
```
conda create -n yolo9env python=3.9
conda activate yolo9env 

```

from pytorch website: https://pytorch.org/get-started/locally/

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

```
git clone https://github.com/SkalskiP/yolov9.git
cd yolov9
pip install -r requirements.txt -q
```

from HOME:
```
mkdir weights
wget -P ./weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt
```

## Training 

```
cd yolov9

python train_dual.py \
--batch 16 --epochs 25 --img 640 --device 0 --min-items 0 --close-mosaic 15 \
--data hoi_dataset/data.yaml \
--weights ./../weights/yolov9-e.pt \
--cfg models/detect/yolov9-e.yaml \
--hyp hyp.scratch-high.yaml
```

errors and solutions: 

Unable to find a valid cuDNN algorithm to run convolution
run with --batch 8 --epochs 16 

AttributeError: 'FreeTypeFont' object has no attribute 'getsize'
need to downgrade Pillow; pip install Pillow==9.5.0


some useful terminal stuff:
```
rm -r data/Date01_Sub01_backpack_back_color
```
```
pip install --root-user-action=ignore thop>=0.1.1
```
```
file ./hoi_dataset/test/images/Date01_Sub01_basketball0000.jpg
```
it gives us the size in our case: 432x240
current versions that work with ProPainter and GroundingDINO:
```
ackage                  Version     Editable project location
------------------------ ----------- ------------------------------------------------
addict                   2.4.0
av                       12.0.0
certifi                  2024.2.2
charset-normalizer       3.3.2
contourpy                1.1.1
cycler                   0.12.1
einops                   0.7.0
filelock                 3.13.1
fonttools                4.50.0
fsspec                   2024.3.1
future                   1.0.0
groundingdino            0.1.0       /home/israelov/hoi-data-processing/GroundingDINO
huggingface-hub          0.21.4
idna                     3.4
imageio                  2.34.0
imageio-ffmpeg           0.4.9
importlib_metadata       7.1.0
importlib_resources      6.4.0
Jinja2                   3.1.3
kiwisolver               1.4.5
lazy_loader              0.3
MarkupSafe               2.1.5
matplotlib               3.7.5
mkl-fft                  1.3.8
mkl-random               1.2.4
mkl-service              2.4.0
mpmath                   1.3.0
networkx                 3.1
numpy                    1.24.3
nvidia-cublas-cu12       12.1.3.1
nvidia-cuda-cupti-cu12   12.1.105
nvidia-cuda-nvrtc-cu12   12.1.105
nvidia-cuda-runtime-cu12 12.1.105
nvidia-cudnn-cu12        8.9.2.26
nvidia-cufft-cu12        11.0.2.54
nvidia-curand-cu12       10.3.2.106
nvidia-cusolver-cu12     11.4.5.107
nvidia-cusparse-cu12     12.1.0.106
nvidia-nccl-cu12         2.19.3
nvidia-nvjitlink-cu12    12.4.99
nvidia-nvtx-cu12         12.1.105
opencv-python            4.9.0.80
packaging                24.0
pillow                   10.2.0
pip                      24.0
platformdirs             4.2.0
pycocotools              2.0.7
pyparsing                3.1.2
python-dateutil          2.9.0.post0
PyWavelets               1.4.1
PyYAML                   6.0.1
regex                    2023.12.25
requests                 2.31.0
safetensors              0.4.2
scikit-image             0.21.0
scipy                    1.10.1
setuptools               68.2.2
six                      1.16.0
supervision              0.4.0
sympy                    1.12
tifffile                 2023.7.10
timm                     0.9.16
tokenizers               0.13.3
tomli                    2.0.1
torch                    1.12.1
torchaudio               0.12.1
torchvision              0.13.1
tqdm                     4.66.2
transformers             4.28.0
triton                   2.2.0
typing_extensions        4.10.0
urllib3                  2.1.0
wheel                    0.41.2
yapf                     0.40.2
zipp                     3.18.1
```

## Intercap dataset 

register and download the data from [here](https://intercap.is.tue.mpg.de/)
the dataset inludes interactions with soccer ball, suitcase, toolbox, tennis racquet, chair (move, pick up, sit), umbrella, cup, skateboard, bottle. 

the dataset extract into the data folder of the repository. 

then run to get the videos and object masks (there is a default input path, output path and resize factor but these can be changed). 

```
python ./sample/intercap4propainter.py
```

then run the script ```./scripts/ppainter_inference.sh```

if you make any changes run before that: ```chmod +x ./scripts/ppainter_inference.sh```


# Acknowledgements

For the tree i used:
(https://tree.nathanfriend.io/)[https://tree.nathanfriend.io/]


