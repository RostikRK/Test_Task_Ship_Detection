import pandas as pd
import os

from segmentation_train import preprocess_input
from utils import calculate_iou_score, rle_to_mask, calculate_dice_score, predict_on_tiles, reconstruct_from_tiles, crop_image
import cv2
import numpy as np
from sklearn.metrics import jaccard_score

os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow.keras.models import load_model


# Load the CSV file into a DataFrame
df = pd.read_csv('path_to_file')

# Group the DataFrame by 'ImageId' to get a list of masks for each image
grouped = df.groupby('ImageId')['EncodedPixels'].apply(list)

model = load_model('path_to_model', compile=False)

# Get a list of unique image IDs
image_ids = grouped.index.tolist()
# Calculate the 90% mark
split_index = int(len(image_ids) * 0.995)
# Get the test image IDs
test_image_ids = image_ids[split_index:]


# Path to the images
image_dir = "path_to_images"

# Initialize variables to store the true and predicted masks
true_masks = []
pred_masks = []

for image_id in test_image_ids:
    # Load the image
    image = cv2.imread(os.path.join(image_dir, image_id))

    # Create tiles
    tiles = crop_image(image)

    # Predict on tiles
    tile_masks = predict_on_tiles(tiles, classification_model, segmentation_model)

    # Reconstruct the full mask from the tiles
    full_mask = reconstruct_from_tiles(tile_masks, image.shape, tile_size=256, stride=256)

    # Add the predicted mask to the list
    pred_masks.append(full_mask)

    # Get the true mask for this image
    rle_encoded_mask = grouped.loc[image_id]

    # Decode the RLE-encoded mask
    true_mask = rle_to_mask(rle_encoded_mask, (768, 768))  # replace with your function to decode the mask

    # Add the true mask to the list
    true_masks.append(true_mask)

binary_pred_masks = np.where(np.concatenate(pred_masks) > 0.5, 1, 0)

# Convert binary_pred_masks to single-channel
binary_pred_masks = np.max(binary_pred_masks, axis=-1)


# Calculate the IOU and Dice scores
iou_score = calculate_iou_score(true_masks, binary_pred_masks)
dice_score = calculate_dice_score(np.concatenate(true_masks), binary_pred_masks.ravel())

print(f"IOU Score: {iou_score}")
print(f"Dice Score: {dice_score}")
