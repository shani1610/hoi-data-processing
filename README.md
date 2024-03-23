# HOI Data Processing 
human-object interaction video dataset processing
including object detection and object removal using inpaiting

## Dataset annotation 

This is annotation for the dataset

the code was adpted from roboflow notebook: Automated Dataset Annotation and Evaluation with Grounding DINO

## installation
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
