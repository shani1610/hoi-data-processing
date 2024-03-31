# HOI Data Processing 
human-object interaction video dataset processing
including object detection and object removal using inpaiting

python 3.8.19
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -q -e .

please pay attention to the CUDA version, we need the Pytorch corresponding to `nvcc --version` (my is CUDA 11.2) and not the driver's one `nvidia smi`

## Dataset annotation 

This is annotation for the dataset

the code was adpted from roboflow notebook: Automated Dataset Annotation and Evaluation with Grounding DINO

## OLD installation
HOME is what you get by running pwd
was tested on Python 3.9.18, CUDA 11.8 (check using nvcc --version)
```
conda create -n gdinoenv python=3.9
conda activate gdinoenv 

```

from pytorch website: https://pytorch.org/get-started/locally/

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

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
# Visual AI usage

ssh israelov@133.15.42.60

(for first time need to do other kubectl commands)
kubectl exec -n israelov -it israelov-hoi-data-processing -- /bin/bash

(for first time only)
conda create -n hoi-env python=3.8 

conda activate hoi-env

cd hoi-data-processing/

python ./sample/extract_files_behave.py

or

python ./sample/extract_files_behave.py --root_directory "./data/preprocessed_behave" --output_color "./data/images_for_gdino/" --output_inpaint "./data/images_for_ppainter/"


wait because it might take a while for the images to appear. 

now the inpainting part:
cd ProPainter/

python inference_propainter.py -i "./../data/images_for_ppainter/Date01_Sub01_backpack_back/Date01_Sub01_backpack_back_color_resized" -m "./../data/images_for_ppainter/Date01_Sub01_backpack_back/Date01_Sub01_backpack_back_mask_resized" -o "./../data/hoi_dataset/Date01_Sub01_backpack_back" --save_frames

to save the frames in different way:

python inference_propainter.py -i "./../data/images_for_ppainter/Date01_Sub01_backpack_back/Date01_Sub01_backpack_back_color_resized" -m "./../data/images_for_ppainter/Date01_Sub01_backpack_back/Date01_Sub01_backpack_back_mask_resized" -o "./../data/hoi_dataset/Date01_Sub01_backpack_back" --name_of_subject "Date01_Sub01_backpack_back" --save_frames_behave

maybe can be written without the --name of subject so might consider checking it. 

for object detection:

from the hoi-data-processing folder

```
python create_dataset.py --frames_dir "./data/images_for_gdino/Date01_Sub01_backpack_back_color/" --object_name "backpack" --name_of_subject "Date01_Sub01_backpack_back" --annotation_output_folder "./data/hoi_dataset/Date01_Sub01_backpack_back/labels/"
```
to check corresondenceL
python check_img_lab_corr.py --img_path "./data/hoi_dataset/Date01_Sub01_backpack_back/images/Date01_Sub01_backpack_back0022.jpg" --lab_path "./data/hoi_dataset/Date01_Sub01_backpack_back/labels/Date01_Sub01_backpack_back0022.txt"

had to do some package stuff before saving the annotations:

pip uninstall charset-normalizer

conda install -c conda-forge charset-normalizer

pip uninstall transformers

pip install transformers==4.28.0


some terminal stuff:
rm -r data/Date01_Sub01_backpack_back_color
