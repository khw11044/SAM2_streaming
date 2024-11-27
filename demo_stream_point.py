import torch
import numpy as np
import cv2
from sam2.build_sam import build_sam2_camera_predictor

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# 모델 및 예측기 초기화
sam2_checkpoint = "./checkpoints/sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

# 전역 변수
points = []
num_points = -1
points_entered = False
prompt_text = "Select number of points"
error_text = "Press only number"
if_init = False


# 마우스 콜백 함수
def collect_points(event, x, y, flags, param):
    global points, num_points, points_entered
    if points_entered and event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        if len(points) == num_points:
            points_entered = False  # 점 입력이 완료되면 플래그 해제


# 카메라 열기
cap = cv2.VideoCapture(0)
cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera", collect_points)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]
    temp_frame = frame.copy()

    # 점 수 입력
    if num_points == -1:
        cv2.putText(temp_frame, prompt_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Camera", temp_frame)
        key = cv2.waitKey(1)
        if key in range(48, 58):  # 0-9 숫자 입력
            num_points = key - 48
            points_entered = True
            points = []
        elif key == 27 or key == ord('q'):  # ESC 또는 Q로 종료
            break
    elif points_entered:
        # 점 선택 중
        for point in points:
            cv2.circle(temp_frame, tuple(point), 5, (255, 0, 0), -1)
        cv2.putText(temp_frame, f"Select {num_points} points", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Camera", temp_frame)
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):  # ESC 또는 Q로 종료
            break
    else:
        # Segmentation 수행
        if not if_init:
            if_init = True
            predictor.load_first_frame(frame)

            ann_frame_idx = 0
            ann_obj_id = tuple(range(1, num_points + 1))
            labels = np.arange(1, num_points + 1, dtype=np.int32)
            points = np.array(points, dtype=np.float32)

            # `add_new_prompt` 호출
            _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
            )
        else:
            # Tracking
            out_obj_ids, out_mask_logits = predictor.track(frame)

        # Mask 시각화
        all_mask = np.zeros((height, width, 1), dtype=np.uint8)
        for i in range(len(out_obj_ids)):
            out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
            all_mask = cv2.bitwise_or(all_mask, out_mask)

        all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
        frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
