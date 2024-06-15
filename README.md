# Image-Segmentation-using-K-Means

This repository contains a Python script (`image_segmentation.py`) that performs image segmentation using K-Means clustering with OpenCV and NumPy libraries.

### Features

- **Image Segmentation**: Uses K-Means clustering to segment an input image into `k` clusters.
- **Customizable Clusters**: Allows customization of the number of clusters (`k`) for segmentation.
- **Output**: Saves the segmented image in the specified output path.

### Requirements

- Python 3.x
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)

### Usage

To segment an image, run the script with the following command:

```bash
python image_segmentation.py --image /path/to/your/input_image.jpg --k 4 --output /path/to/your/output_segmented_image.jpg
```

#### Arguments

- `--image`: Path to the input image (required).
- `--k`: Number of clusters for K-Means (default: 4).
- `--output`: Path to save the segmented image (required).

### Example

```bash
python image_segmentation.py --image /path/to/your/input_image.jpg --k 4 --output /path/to/your/output_segmented_image.jpg
```

### Notes

- Adjust paths (`/path/to/your/`) according to your specific file locations.
- Ensure that OpenCV (`cv2`) and NumPy (`np`) libraries are installed before running the script.
- Experiment with different values of `--k` to see how the segmentation changes based on the number of clusters.
