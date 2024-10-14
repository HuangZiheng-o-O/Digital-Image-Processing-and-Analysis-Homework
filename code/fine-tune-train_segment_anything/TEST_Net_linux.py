import numpy as np
import torch
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os
# use bfloat16 for the entire script (memory efficient)
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

# Load image
# image_path = r"sample_image.jpg"  # path to image
# mask_path = r"sample_mask.png"  # path to mask, the mask will define the image region to segment
dir_path = 'LabPicsV1/Complex/Test/Koen2All_Chemistry experiment 2. - Coloured flask.-screenshot'
image_path = os.path.join(dir_path, 'Image.png')
mask_path = r"LabPicsV1/Complex/Test/Koen2All_Chemistry experiment 2. - Coloured flask.-screenshot/FinalMask2.png"
def read_image(image_path, mask_path):
    img = cv2.imread(image_path)[..., ::-1]  # read image as rgb
    mask = cv2.imread(mask_path, 0)  # mask of the region we want to segment

    # Resize image to maximum size of 1024
    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), interpolation=cv2.INTER_NEAREST)
    return img, mask

image, mask = read_image(image_path, mask_path)
num_samples = 30  # number of points/segment to sample

def get_points(mask, num_points):
    points = []
    for i in range(num_points):
        coords = np.argwhere(mask > 0)
        yx = np.array(coords[np.random.randint(len(coords))])
        points.append([[yx[1], yx[0]]])
    return np.array(points)

input_points = get_points(mask, num_samples)

# Load model
sam2_checkpoint = "sam2_hiera_small.pt"  # "sam2_hiera_large.pt"
model_cfg = "sam2_hiera_s.yaml"  # "sam2_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

# Build net and load weights
predictor = SAM2ImagePredictor(sam2_model)
predictor.model.load_state_dict(torch.load("model.torch"))

# Predict mask
with torch.no_grad():
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=np.ones([input_points.shape[0], 1])
    )

# Short predicted masks from high to low score
masks = masks[:, 0].astype(bool)
shorted_masks = masks[np.argsort(scores[:, 0])][::-1].astype(bool)

# Stitch predicted mask into one segmentation mask
seg_map = np.zeros_like(shorted_masks[0], dtype=np.uint8)
occupancy_mask = np.zeros_like(shorted_masks[0], dtype=bool)

for i in range(shorted_masks.shape[0]):
    mask = shorted_masks[i]
    if (mask * occupancy_mask).sum() / mask.sum() > 0.15:
        continue
    mask[occupancy_mask] = 0
    seg_map[mask] = i + 1
    occupancy_mask[mask] = 1

# Create colored annotation map
height, width = seg_map.shape
rgb_image = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)

for id_class in range(1, seg_map.max() + 1):
    rgb_image[seg_map == id_class] = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]

# Save results
cv2.imwrite("annotation.png", rgb_image)
cv2.imwrite("mixed_image.png", (rgb_image / 2 + image / 2).astype(np.uint8))

# Save masks and scores
np.save("masks.npy", masks)
np.save("scores.npy", scores)

print("Results saved: annotation.png, mixed_image.png, masks.npy, scores.npy")


