import os
import cv2
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_camera_predictor
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()        # use bfloat16 for the entire notebook

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ----------- argparse 추가 -----------
parser = argparse.ArgumentParser()
parser.add_argument("--model_version", type=str, default="sam2", help="모델 버전 (e.g., sam2, sam2.1)")
args = parser.parse_args()
# ------------------------------------


# 모델 설정
model_version=args.model_version
sam2_checkpoint = f"./checkpoints/{model_version}/{model_version}_hiera_small.pt"
model_cfg = f"{model_version}/{model_version}_hiera_s.yaml"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)


# 전역 변수 초기화
drawing = False
ix, iy, fx, fy = -1, -1, -1, -1
bbox = None
enter_pressed = False

# 마우스 콜백 함수
def draw_rectangle(event, x, y, flags, param):
    global drawing, ix, iy, fx, fy, bbox, enter_pressed
    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 버튼 누름
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:  # 마우스 이동
        if drawing:
            fx, fy = x, y
    elif event == cv2.EVENT_LBUTTONUP:  # 마우스 왼쪽 버튼 뗌
        drawing = False
        fx, fy = x, y
        bbox = (ix, iy, fx, fy)  # 바운딩 박스 저장
        enter_pressed = True  # 드래그 종료 시 바로 다음 단계로 진행

# 카메라 열기
cap = cv2.VideoCapture(0)
cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera", draw_rectangle)


if_init = False
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    width, height = frame.shape[:2][::-1]
    if not ret:
        print("Failed to grab frame")
        break

    # 바운딩 박스 선택
    if not enter_pressed:
        temp_frame = frame.copy()
        if drawing and ix >= 0 and iy >= 0:  # 드래그 중인 경우
            cv2.rectangle(frame, (ix, iy), (fx, fy), (255, 0, 0), 2)        
        cv2.putText(frame, "Select an object by drawing a box", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):  # ESC 또는 Q로 종료
            break
    else:
        
        if not if_init:
            if_init = True
            predictor.load_first_frame(frame)
            using_box = True 
            ann_frame_idx = 0
            ann_obj_id = (1)
            labels = np.array([1], dtype=np.int32)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            bbox = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float32)
            _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                    frame_idx=ann_frame_idx, obj_id=ann_obj_id, bbox=bbox
                )
        else:
            out_obj_ids, out_mask_logits = predictor.track(frame)

            all_mask = np.zeros((height, width, 1), dtype=np.uint8)
            # print(all_mask.shape)
            for i in range(0, len(out_obj_ids)):
                out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).byte().cuda()
                all_mask = out_mask.cpu().numpy() * 255

            all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
            frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)
            
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
