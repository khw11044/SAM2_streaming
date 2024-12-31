import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from IPython import display
from sam2.build_sam import build_sam2_camera_predictor, build_sam2_video_predictor


# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    

model_version='sam2'
sam2_checkpoint = f"./checkpoints/{model_version}/{model_version}_hiera_small.pt"
model_cfg = f"{model_version}/{model_version}_hiera_s.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)


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
    # fig = ax.get_figure()
    # fig.savefig(f"./tmp/output_mask_{save_id}.png", bbox_inches="tight", dpi=300)


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
    print(x, y, w, h)
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=None, edgecolor="blue", linewidth=2))

cap = cv2.VideoCapture("./videos/aquarium.mp4")
ret, frame = cap.read()

width, height = frame.shape[:2][::-1]

# predictor.load_first_frame(frame)
if_init = True

using_point = True # if True, we use point prompt
using_box = False # if True, we use point prompt
using_mask= False  # if True, we use mask prompt

ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = (1)
# Let's add a positive click at (x, y) = (210, 350) to get started

# using point prompt
points = np.array([[670, 247]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], dtype=np.int32)
bbox = np.array([[600, 214], [765, 286]], dtype=np.float32)

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(frame)


if using_point:
    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )
    show_points(points, labels, plt.gca())

elif using_box:
    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
        frame_idx=ann_frame_idx, obj_id=ann_obj_id, bbox=bbox
    )
    show_bbox(bbox, plt.gca())

    

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    width, height = frame.shape[:2][::-1]
    if not if_init:

        predictor.load_first_frame(frame)
        if_init = True

        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
        # Let's add a positive click at (x, y) = (210, 350) to get started

        ## ! add bbox
        bbox = np.array([[600, 214], [765, 286]], dtype=np.float32)
        
        _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
            frame_idx=ann_frame_idx, obj_id=ann_obj_id, bbox=bbox
        )

    else:
        out_obj_ids, out_mask_logits = predictor.track(frame)

        all_mask = np.zeros((height, width, 1), dtype=np.uint8)
        # print(all_mask.shape)
        for i in range(0, len(out_obj_ids)):
            out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                np.uint8
            ) * 255

            all_mask = cv2.bitwise_or(all_mask, out_mask)

        all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
        frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()