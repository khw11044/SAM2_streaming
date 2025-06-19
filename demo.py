import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from sam2.build_sam import build_sam2_camera_predictor

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()        # use bfloat16 for the entire notebook

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ----------- argparse 추가 -----------
parser = argparse.ArgumentParser()
parser.add_argument("--model_version", type=str, default="sam2.1", help="모델 버전 (e.g., sam2, sam2.1)")
parser.add_argument("--video_path", type=str, default="./videos/aquarium.mp4", help="데모할 mp4 영상 경로")
parser.add_argument("--output_gif", type=str, default="./output_gif/segmentation.gif", help="저장할 GIF 파일명")
args = parser.parse_args()
# ------------------------------------

# 모델 설정
model_version = args.model_version
sam2_checkpoint = f"./checkpoints/{model_version}/{model_version}_hiera_small.pt"
model_cfg = f"{model_version}/{model_version}_hiera_s.yaml"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

def show_mask(mask, ax, obj_id=None, random_color=False, save_id=0):
    os.makedirs('./tmp', exist_ok=True)
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )

def show_bbox(bbox, ax, marker_size=200):
    tl, br = bbox[0], bbox[1]
    w, h = (br - tl)[0], (br - tl)[1]
    x, y = tl[0], tl[1]
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=None, edgecolor="blue", linewidth=2))

# GIF 저장을 위한 폴더 생성
os.makedirs(os.path.dirname(args.output_gif), exist_ok=True)

# GIF 저장을 위한 이미지 리스트 초기화
images = []

cap = cv2.VideoCapture(args.video_path)
ret, frame = cap.read()

if not ret:
    raise FileNotFoundError(f"Cannot read first frame from {args.video_path}")

width, height = frame.shape[:2][::-1]

predictor.load_first_frame(frame)
if_init = True

using_point = False
using_box = True
using_mask = False

ann_frame_idx = 0
ann_obj_id = 1

# Add bbox (필요시 좌표 수정)
bbox = np.array([[600, 214], [765, 286]], dtype=np.float32)

if using_box:
    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
        frame_idx=ann_frame_idx, obj_id=ann_obj_id, bbox=bbox
    )

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    width, height = frame.shape[:2][::-1]
    if not if_init:
        predictor.load_first_frame(frame)
        if_init = True
        ann_frame_idx = 0
        ann_obj_id = 1
        bbox = np.array([[600, 214], [765, 286]], dtype=np.float32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
            frame_idx=ann_frame_idx, obj_id=ann_obj_id, bbox=bbox
        )
    else:
        out_obj_ids, out_mask_logits = predictor.track(frame)
        all_mask = np.zeros((height, width, 1), dtype=np.uint8)
        for i in range(0, len(out_obj_ids)):
            out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                np.uint8
            ) * 255
            all_mask = cv2.bitwise_or(all_mask, out_mask)
        all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
        frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)

    images.append(Image.fromarray(frame))

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("frame", frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# GIF로 저장
images[0].save(
    args.output_gif,
    save_all=True,
    append_images=images[1:],
    optimize=False,
    duration=int(1000 / 40),  # 40fps
    loop=0
)

print(f"GIF 저장 완료: {args.output_gif}")
