"""Auto-crop utility for OCTA images: remove black borders."""

import numpy as np
def auto_crop_octa_images(fake_image, real_image, threshold_ratio=0.05, vertical_padding=10):
    """Crop OCTA image pair by removing black borders.

    Args:
        fake_image: Generated OCTA image (H, W)
        real_image: Real OCTA image (H, W)
        threshold_ratio: Threshold to distinguish content from background
        vertical_padding: Extra pixels to retain above and below content

    Returns:
        fake_cropped, real_cropped, bbox_info
    """

    def _find_bbox(image, threshold_ratio):
        row_means = np.mean(image, axis=1)
        threshold = np.max(image) * threshold_ratio

        valid_rows = np.where(row_means > threshold)[0]
        if len(valid_rows) == 0:
            return 0, 0, image.shape[1] - 1, image.shape[0] - 1

        y_min, y_max = valid_rows[0], valid_rows[-1]

        col_means = np.mean(image[y_min:y_max + 1, :], axis=0)
        valid_cols = np.where(col_means > threshold)[0]
        if len(valid_cols) == 0:
            x_min, x_max = 0, image.shape[1] - 1
        else:
            x_min, x_max = valid_cols[0], valid_cols[-1]

        return x_min, y_min, x_max, y_max

    h, w = fake_image.shape

    bbox_fake = _find_bbox(fake_image, threshold_ratio)
    bbox_real = _find_bbox(real_image, threshold_ratio)

    # Union of both bounding boxes
    x_min = min(bbox_fake[0], bbox_real[0])
    y_min = min(bbox_fake[1], bbox_real[1])
    x_max = max(bbox_fake[2], bbox_real[2])
    y_max = max(bbox_fake[3], bbox_real[3])

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w - 1, x_max)
    y_max = min(h - 1, y_max)

    # Apply vertical padding
    y_min = max(0, y_min - vertical_padding)
    y_max = min(h - 1, y_max + vertical_padding)

    expanded_bbox = (x_min, y_min, x_max, y_max)

    fake_cropped = fake_image[y_min:y_max + 1, x_min:x_max + 1]
    real_cropped = real_image[y_min:y_max + 1, x_min:x_max + 1]

    bbox_info = {'expanded_bbox': expanded_bbox}

    return fake_cropped, real_cropped, bbox_info