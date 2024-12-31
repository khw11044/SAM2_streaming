import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from IPython import display
from sam2.build_sam import build_sam2_camera_predictor


from ultralytics import YOLO  
yolomodel = YOLO("./checkpoints/yolo/yolo11n.pt")

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    

def get_bbox(frame):
    results = yolomodel.track(source=frame, classes=[0], conf=0.5, show=False, stream=True, verbose=False)
    largest_box = None  # 가장 큰 바운딩 박스를 저장할 변수
    largest_area = 0  # 가장 큰 바운딩 박스의 면적

    # 탐지 결과 처리
    for result in results:
        boxes = result.boxes  # 탐지된 객체의 박스 정보
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
            area = (x2 - x1) * (y2 - y1)  # 바운딩 박스 면적 계산
            
            # 가장 큰 바운딩 박스 갱신
            if area > largest_area:
                largest_area = area
                largest_box = (x1, y1, x2, y2, box.conf[0], int(box.cls[0]))  # 좌표, 신뢰도, 클래스 저장

    return largest_box

model_version='sam2'
sam2_checkpoint = f"./checkpoints/{model_version}/{model_version}_hiera_small.pt"
model_cfg = f"{model_version}/{model_version}_hiera_s.yaml"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)


# 전역 변수 초기화
if_init = False
largest_bbox=None
bbox_show = True
seg_show = False

# 카메라 열기
cap = cv2.VideoCapture(0)


while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    if not ret:
        print("Failed to grab frame")
        break
    
    if not largest_bbox and not if_init:
        largest_bbox = get_bbox(frame)
    
    if largest_bbox and not if_init:
        x1, y1, x2, y2, conf, cls = largest_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        predictor.load_first_frame(frame)
        bbox = np.array([[largest_bbox[0], largest_bbox[1]],
                        [largest_bbox[2], largest_bbox[3]]], dtype=np.float32)
        
        _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(frame_idx=0, obj_id=1, bbox=bbox)
        if_init = True
        
    elif if_init:
        out_obj_ids, out_mask_logits = predictor.track(frame)
        # all_mask = np.zeros((height, width, 1), dtype=np.uint8)
        all_mask = torch.zeros((height, width), dtype=torch.uint8, device=device)
        for i in range(len(out_obj_ids)):
            out_mask = (out_mask_logits[i] > 0.0).byte()
            all_mask = torch.bitwise_or(all_mask, out_mask.squeeze(0))

        if bbox_show:
            combined_mask = (all_mask > 0).byte().cpu().numpy().astype(np.uint8)
            coords = np.argwhere(combined_mask)
            if coords.size != 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)        
                
        # 마스크 적용
        if seg_show and all_mask is not None:
            all_mask = all_mask.cpu().numpy() * 255
            all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
            frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)
        
    
    # OpenCV로 이미지 표시
    cv2.imshow("YOLO Object Detection", frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
