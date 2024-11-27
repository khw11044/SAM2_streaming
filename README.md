# segment-anything-2 real-time
Run Segment Anything Model 2 on a **live video stream**

## SAM2 live video stream 

## SAM2 streaming 

## real-time SAM2


## News
- 27/11/2024 : 최초 SAM2 실시간 세그멘테이션 코드 성공 


## Demos
<div align=center>
<p align="center">
<img src="./assets/blackswan.gif" width="880">
</p>

</div>


## Getting Started

### Installation

```bash
conda create -n seg python=3.11 -y

conda activate seg 

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

```


```bash
pip install -e .
```
### Download Checkpoint

Then, we need to download a model checkpoint.

```bash
cd checkpoints
./download_ckpts.sh
```

Then SAM-2-online can be used in a few lines as follows for image and video and **camera** prediction.

### Demo streaming 

```bash

python demo_stream.py


python demo.py

```

demo_stream.py 는 바운딩 박스 마우스로 지정하고 엔터를 누르면 segmentation 시작 



## References:

- SAM2 Repository: https://github.com/facebookresearch/segment-anything-2

- https://github.com/Gy920/segment-anything-2-real-time/tree/main