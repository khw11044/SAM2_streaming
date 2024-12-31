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
model_version='sam2'
sam2_checkpoint = f"./checkpoints/{model_version}/{model_version}_hiera_small.pt"
model_cfg = f"{model_version}/{model_version}_hiera_s.yaml"
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
            print('out_obj_ids:', out_obj_ids)
            print('out_mask_logits:', out_mask_logits)

        # Mask 시각화
        all_mask = np.zeros_like(frame, dtype=np.uint8)  # 전체 마스크 초기화 (RGB로 변경)
        random_colors = [tuple(np.random.randint(0, 256, size=3)) for _ in range(len(out_obj_ids))]  # 랜덤 색상 생성

        for i, color in enumerate(random_colors):
            # 각 마스크 생성
            out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

            # 마스크를 RGB 형태로 확장
            colored_mask = np.zeros_like(frame, dtype=np.uint8)
            for c in range(3):  # 각 채널에 색상 적용
                colored_mask[:, :, c] = out_mask[:, :, 0] * color[c]

            # 전체 마스크에 추가 (투명도 조절 가능)
            all_mask = cv2.addWeighted(all_mask, 1, colored_mask, 0.5, 0)

        # 원본 프레임과 마스크를 합성
        frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
