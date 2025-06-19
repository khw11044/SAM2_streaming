import torch
import cv2
import argparse
import numpy as np
from sam2.build_sam import build_sam2_camera_predictor

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ----------- argparse 추가 -----------
parser = argparse.ArgumentParser()
parser.add_argument("--model_version", type=str, default="sam2", help="모델 버전 (e.g., sam2, sam2.1)")
args = parser.parse_args()
# ------------------------------------

# 모델 및 예측기 초기화
model_version=args.model_version
sam2_checkpoint = f"./checkpoints/{model_version}/{model_version}_hiera_small.pt"
model_cfg = f"{model_version}/{model_version}_hiera_s.yaml"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

# 전역 변수
point = None
point_selected = False
if_init = False
random_color = True

# 마우스 콜백 함수
def collect_point(event, x, y, flags, param):
    global point, point_selected
    if not point_selected and event == cv2.EVENT_LBUTTONDOWN:
        point = [x, y]
        point_selected = True

# 카메라 열기
cap = cv2.VideoCapture(0)
cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera", collect_point)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    temp_frame = frame.copy()

    if not point_selected:
        # 점 선택 메시지 표시
        cv2.putText(temp_frame, "Select an object by clicking a point", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Camera", temp_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # Segmentation 수행
        if not if_init:
            if_init = True
            predictor.load_first_frame(frame)

            ann_frame_idx = 0
            ann_obj_id = (1,)  # 단일 객체 ID
            labels = np.array([1], dtype=np.int32)
            points = np.array([point], dtype=np.float32)

            # `add_new_prompt` 호출
            _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
            )
        else:
            # Tracking
            out_obj_ids, out_mask_logits = predictor.track(frame)

        # Mask 시각화
        all_mask = np.zeros_like(frame, dtype=np.uint8)  # 전체 마스크 초기화 (RGB로 변경)
        
        if random_color:
            # 랜덤 색상 마스크
            color = tuple(np.random.randint(0, 256, size=3))
            out_mask = (out_mask_logits[0] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            colored_mask = np.zeros_like(frame, dtype=np.uint8)
            for c in range(3):  # 각 채널에 색상 적용
                colored_mask[:, :, c] = out_mask[:, :, 0] * color[c]
        else:
            # 단색 흰색 마스크
            out_mask = (out_mask_logits[0] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
            colored_mask = cv2.cvtColor(out_mask, cv2.COLOR_GRAY2RGB)

        # 전체 마스크에 추가
        all_mask = cv2.addWeighted(all_mask, 1, colored_mask, 0.5, 0)
        frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()