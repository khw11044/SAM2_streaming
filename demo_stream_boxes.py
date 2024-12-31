import torch
import numpy as np
import cv2
from sam2.build_sam import build_sam2_camera_predictor

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # Ampere GPU 설정
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# SAM2 모델 초기화
sam2_checkpoint = "./checkpoints/sam2_hiera_small.pt"
model_cfg = "sam2/sam2_hiera_s.yaml"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

# 전역 변수 초기화
drawing = False
ix, iy, fx, fy = -1, -1, -1, -1
bboxes = []  # 바운딩 박스 리스트
num_bboxes = -1  # 사용할 바운딩 박스 수
enter_pressed = False
object_colors = {}  # 객체별 고유 색상
if_init = False

# 마우스 콜백 함수
def draw_rectangle(event, x, y, flags, param):
    global drawing, ix, iy, fx, fy, bboxes, enter_pressed
    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 버튼 누름
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:  # 마우스 이동
        if drawing:
            fx, fy = x, y
    elif event == cv2.EVENT_LBUTTONUP:  # 마우스 왼쪽 버튼 뗌
        drawing = False
        fx, fy = x, y
        bboxes.append([[ix, iy], [fx, fy]])  # 바운딩 박스를 올바른 형식으로 추가
        if len(bboxes) == num_bboxes:  # 모든 바운딩 박스가 선택되었으면
            enter_pressed = True

# 카메라 열기
cap = cv2.VideoCapture(0)
cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera", draw_rectangle)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # 좌우 반전
    height, width = frame.shape[:2]
    if not ret:
        print("Failed to grab frame")
        break

    if num_bboxes == -1:
        # 바운딩 박스 수 선택
        temp_frame = frame.copy()
        cv2.putText(temp_frame, "Enter number of bounding boxes (0-9):", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Camera", temp_frame)
        key = cv2.waitKey(1)
        if key in range(48, 58):  # 숫자 입력 (0-9)
            num_bboxes = key - 48
            bboxes = []
        elif key == 27 or key == ord('q'):  # ESC 또는 Q로 종료
            break
    elif not enter_pressed:
        # 바운딩 박스 선택
        temp_frame = frame.copy()
        for box in bboxes:  # 이미 선택된 바운딩 박스 표시
            cv2.rectangle(temp_frame, (box[0][0], box[0][1]), (box[1][0], box[1][1]), (0, 255, 0), 2)
        if drawing and ix >= 0 and iy >= 0:  # 드래그 중인 경우
            cv2.rectangle(temp_frame, (ix, iy), (fx, fy), (255, 0, 0), 2)
        cv2.imshow("Camera", temp_frame)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):  # ESC 또는 Q로 종료
            break
    else:
        # 세그멘테이션 및 시각화
        if not if_init:
            if_init = True
            predictor.load_first_frame(frame)

            ann_frame_idx = 0
            ann_obj_ids = tuple(range(1, num_bboxes + 1))
            labels = np.array(list(range(1, num_bboxes + 1)), dtype=np.int32)
            bbox_array = np.array(bboxes, dtype=np.float32)  # numpy 배열로 변환
            print('bbox_array: ', bbox_array)
            # 객체 색상 설정
            for i, obj_id in enumerate(ann_obj_ids):
                object_colors[obj_id] = tuple(np.random.randint(0, 256, size=3))  # 랜덤 색상

            # SAM2에 바운딩 박스 추가
            _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                frame_idx=ann_frame_idx, obj_id=ann_obj_ids, bbox=bbox_array
            )
        else:
            out_obj_ids, out_mask_logits = predictor.track(frame)

        # 객체별 마스크 생성 및 색상 적용
        all_mask = np.zeros_like(frame, dtype=np.uint8)
        for i, obj_id in enumerate(out_obj_ids):
            color = object_colors[obj_id]  # 객체별 색상
            out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

            # RGB 채널별 마스크 색상 적용
            colored_mask = np.zeros_like(frame, dtype=np.uint8)
            for c in range(3):
                colored_mask[:, :, c] = out_mask[:, :, 0] * color[c]

            # 전체 마스크에 합성
            all_mask = cv2.addWeighted(all_mask, 1, colored_mask, 0.5, 0)

        # 원본 프레임과 마스크 합성
        frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
