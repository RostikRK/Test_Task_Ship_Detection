import pandas as pd
from PIL import Image
from albumentations import HorizontalFlip, VerticalFlip, Compose, RandomBrightnessContrast, ShiftScaleRotate, GaussNoise
import os
import numpy as np

from sklearn.metrics import jaccard_score

os.environ["SM_FRAMEWORK"] = "tf.keras"


def crop_image(image, crop_size=256):
    if isinstance(image, str):  # If the image is a file path
        img = Image.open(image)
    elif isinstance(image, np.ndarray):  # If the image is a numpy array
        img = Image.fromarray(image)

    width, height = img.size

    crops = []
    for i in range(0, height, crop_size):
        for j in range(0, width, crop_size):
            box = (j, i, j + crop_size, i + crop_size)
            crop = img.crop(box)
            crops.append(np.array(crop))

    return crops


def rle_to_mask(rle_list, shape=(768, 768)):
    # Convert the run-length encoding to a binary mask
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for rle in rle_list:
        if pd.isnull(rle):  # If the RLE is NaN, skip this loop iteration
            continue
        starts, lengths = map(np.asarray, (rle.split()[0:][::2], rle.split()[1:][::2]))
        starts = starts.astype(int) - 1
        lengths = lengths.astype(int)  # Convert lengths to int
        ends = starts + lengths
        for start, end in zip(starts, ends):
            mask[start:end] = 1

    return mask.reshape(shape).T  # Reshape the mask


# Function to augment images and masks
def augment_image(image, mask):
    transform = Compose([
        HorizontalFlip(p=0.4),
        VerticalFlip(p=0.4),
        RandomBrightnessContrast(p=0.2),
        ShiftScaleRotate(p=0.1),
        GaussNoise(p=0.2)
    ])
    data = {"image": np.array(image), "mask": mask}
    augmented = transform(**data)

    return augmented["image"], augmented["mask"]


def calculate_iou_score(true_masks, pred_masks):
    # Flatten the masks and compute the IOU score
    iou_score = jaccard_score(np.concatenate(true_masks).ravel(), pred_masks.ravel())
    return iou_score


def calculate_dice_score(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1.)


def predict_on_tiles(tiles, classification_model, segmentation_model):
    tile_masks = []
    for tile in tiles:
        # Normalize the tile
        tile = tile / 255.0

        # Predict with classification model
        classification_pred = classification_model.predict(tile[None, ...], verbose=0)[0]

        if classification_pred > 0.5:  # If the classification model predicts there's a ship
            # Predict with segmentation model
            tile_mask = segmentation_model.predict(tile[None, ...], verbose=0)[0]
            tile_masks.append(tile_mask)
        else:
            tile_masks.append(np.zeros((tile.shape[0], tile.shape[1], 1)))  # Else, return an empty mask

    return tile_masks


def reconstruct_from_tiles(tiles, original_image_shape, tile_size, stride):
    image = np.zeros(original_image_shape)
    count = np.zeros(original_image_shape, dtype=int)
    i = 0
    for x in range(0, original_image_shape[0], stride):
        for y in range(0, original_image_shape[1], stride):
            end_x = min(x + tile_size, original_image_shape[0])
            end_y = min(y + tile_size, original_image_shape[1])
            image[x:end_x, y:end_y] += tiles[i][:end_x - x, :end_y - y]
            count[x:end_x, y:end_y] += 1
            i += 1
    return np.divide(image, count, out=np.zeros_like(image), where=count != 0)
