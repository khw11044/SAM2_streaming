# segment-anything-2 real-time WebCam Streaming
Run Segment Anything Model 2 on a **live video stream**

[논문](https://arxiv.org/abs/2408.00714)

[깃헙링크](https://arxiv.org/abs/2408.00714)


## News
- 27/11/2024 : 최초 SAM2 실시간 세그멘테이션 코드 성공 
- sam2, sam2.1 둘다 가능

## Demo

![segmentation](output_gif/segmentation.gif)

</div>


## Getting Started

### Installation

반드시 파이썬 버전은 3.11 이상이어야 합니다

```bash
conda create -n sam2 python=3.11 -y

conda activate sam2 

pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```


```bash
pip install -e .

```

**위에 명령어 후 아래와 같은 오류 발생시**

```bash

        File "/tmp/pip-build-env-v31jxhmj/overlay/lib/python3.11/site-packages/torch/__init__.py", line 367, in <module>
          from torch._C import *  # noqa: F403
          ^^^^^^^^^^^^^^^^^^^^^^
      ImportError: /tmp/pip-build-env-v31jxhmj/overlay/lib/python3.11/site-packages/torch/lib/../../nvidia/cusparse/lib/libcusparse.so.12: undefined symbol: __nvJitLinkComplete_12_4, version libnvJitLink.so.12
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: subprocess-exited-with-error

× Getting requirements to build editable did not run successfully.
│ exit code: 1
╰─> See above for output.

note: This error originates from a subprocess, and is likely not a problem with pip.

```

다음과 같은 명령어 진행 

```bash
python setup.py build_ext --inplace
```

다음으로 필요 라이브러리 설치 

```bash

pip install -r requirements.txt

```


### Download Checkpoint

SAM2 모델 다운 

```bash
cd checkpoints/sam2

./download_ckpts.sh

cd ../..
```

SAM2.1 모델 다운 

```bash
cd checkpoints/sam2.1

./download_ckpts.sh

cd ../..
```

자 이제 SAM2를 실시간 스트리밍 화면에 적용할 준비가 됐습니다.

### Demo streaming 

1. mp4 파일에서 테스트 해보기 

```python
python demo.py
```

2. webcam에서 실시간 sam2 적용하기 - 마우스로 바운딩 박스를 그려서 sam2할 객체 지정 

```python 
python demo_webcam_box.py
```

![2](https://github.com/user-attachments/assets/0d0ef6b6-6037-4269-ab89-50a4628dccd1)


3. webcam에서 실시간 sam2 적용하기 - 마우스 point 클릭으로 sam2할 객체 지정 

```python 
python demo_webcam_point.py
```

![3_1](https://github.com/user-attachments/assets/5ce081cc-74a7-4765-a63e-461b164537c4)

![3_2](https://github.com/user-attachments/assets/2fab19e2-4e42-442e-84cc-8cf4799f2386)


4. webcam에서 실시간 sam2 적용하기 - 첫 프레임에 사람 tracking 

```python 
python demo_webcam_yolo.py
```

![4](https://github.com/user-attachments/assets/93f5477e-1a0c-48c6-807d-33bdeed06ad6)


## References:

- SAM2 Repository: https://github.com/facebookresearch/segment-anything-2

- https://github.com/Gy920/segment-anything-2-real-time/tree/main