import cv2
import numpy as np
import argparse

def segment_image(image_path, k, output_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not open or find the image: {image_path}")
        return

    # Convert image to RGB (OpenCV uses BGR by default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Define the criteria and apply KMeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8 bit values
    centers = np.uint8(centers)

    # Map the labels to the center values
    segmented_image = centers[labels.flatten()]

    # Reshape back to the original image
    segmented_image = segmented_image.reshape(image.shape)

    # Convert segmented image to BGR for saving with OpenCV
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)

    # Save the segmented image
    cv2.imwrite(output_path, segmented_image)
    print(f"Segmented image saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Segmentation using K-Means Clustering')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--k', type=int, default=4, help='Number of clusters for K-Means')
    parser.add_argument('--output', type=str, required=True, help='Path to save the segmented image')

    args = parser.parse_args()

    segment_image(args.image, args.k, args.output)
