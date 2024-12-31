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
# 전역 변수
points = []
num_points = -1
points_entered = False
object_colors = {}  # 객체 ID별 고유 색상 저장
if_init = False


# Segmentation 수행 및 시각화
def segment_and_visualize(predictor, frame, points, num_points):
    global if_init, object_colors
    height, width = frame.shape[:2]
    all_mask = np.zeros_like(frame, dtype=np.uint8)  # 전체 마스크 초기화 (RGB)

    if not if_init:
        # 첫 번째 프레임 초기화
        if_init = True
        predictor.load_first_frame(frame)

        ann_frame_idx = 0
        ann_obj_ids = tuple(range(num_points))  # 객체 ID를 tuple로 설정
        labels = np.array(list(range(num_points)), dtype=np.int32)  # 모든 점에 대해 레이블 1로 설정
        points = np.array(points, dtype=np.float32)

        # 객체 ID별 색상 할당
        for obj_id in ann_obj_ids:
            if obj_id not in object_colors:
                object_colors[obj_id] = tuple(np.random.randint(0, 256, size=3))

        # 새로운 점과 객체 ID 추가
        _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
            frame_idx=ann_frame_idx, obj_id=ann_obj_ids, points=points, labels=labels
        )
    else:
        # 추적
        out_obj_ids, out_mask_logits = predictor.track(frame)

    print('out_obj_ids: ', out_obj_ids)
    print('out_mask_logits.shape: ', out_mask_logits.shape)
    # 각 객체별 마스크 시각화
    for i, obj_id in enumerate(out_obj_ids[0]):
        # obj_id = obj_id if isinstance(obj_id, int) else obj_id[0]  # 객체 ID가 튜플이면 첫 번째 값 사용
        
        print('obj_id: ',obj_id)
        color = object_colors[obj_id]  # 객체 ID에 대응하는 고유 색상
        out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        # 마스크를 RGB로 확장
        colored_mask = np.zeros_like(frame, dtype=np.uint8)
        for c in range(3):  # RGB 채널 순회
            colored_mask[:, :, c] = out_mask[:, :, 0] * color[c]

        # 전체 마스크에 추가 (투명도 조절 가능)
        all_mask = cv2.addWeighted(all_mask, 1, colored_mask, 0.5, 0)

    # 원본 프레임과 마스크를 합성
    return cv2.addWeighted(frame, 1, all_mask, 0.5, 0)

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

# 카메라 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    temp_frame = frame.copy()

    if num_points == -1:
        cv2.putText(temp_frame, "Select number of points", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Camera", temp_frame)
        key = cv2.waitKey(1)
        if key in range(48, 58):  # 숫자 입력 (0-9)
            num_points = key - 48
            points_entered = True
            points = []
        elif key == 27 or key == ord("q"):  # 종료
            break
    elif points_entered:
        # 점 선택 중
        for point in points:
            cv2.circle(temp_frame, tuple(point), 5, (255, 0, 0), -1)
        cv2.putText(temp_frame, f"Select {num_points} points", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Camera", temp_frame)
        key = cv2.waitKey(1)
        if key == 27 or key == ord("q"):  # 종료
            break
    else:
        # Segmentation 수행 및 시각화
        frame = segment_and_visualize(predictor, frame, points, num_points)
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
