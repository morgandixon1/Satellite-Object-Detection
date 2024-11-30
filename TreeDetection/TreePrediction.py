import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import load_model

def pad_image(img, target_size):
    """
    Pad the image to the target size.
    """
    height, width, channels = img.shape
    pad_height = target_size[0] - height
    pad_width = target_size[1] - width

    # Ensure padding is not negative
    pad_top = pad_height // 2 if pad_height > 0 else 0
    pad_bottom = pad_height - pad_top if pad_height > 0 else 0
    pad_left = pad_width // 2 if pad_width > 0 else 0
    pad_right = pad_width - pad_left if pad_width > 0 else 0
    padded_img = cv2.copyMakeBorder(
        img,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_REFLECT,
    )
    return padded_img

def get_image_patch(image, center_x, center_y, target_size=(64, 64)):
    """
    Extracts a square patch from the image centered at (center_x, center_y),
    and pads it to the target_size (height, width).
    """
    half_height = target_size[0] // 2
    half_width = target_size[1] // 2
    y1 = max(0, center_y - half_height)
    y2 = min(image.shape[0], center_y + half_height)
    x1 = max(0, center_x - half_width)
    x2 = min(image.shape[1], center_x + half_width)

    patch = image[y1:y2, x1:x2]
    padded_patch = pad_image(patch, target_size)
    return padded_patch

class StandalonePredictor:
    def __init__(self):
        self.model = None

    def load(self, model_path):
        self.model = load_model(model_path)
        print("Model loaded successfully.")

    def predict(self, image_patch):
        """
        Make predictions using the loaded model.
        image_patch: a numpy array of shape (height, width, 3)
        Returns the predicted probability of the 'tree' class.
        """
        if self.model is None:
            raise ValueError("Model must be loaded before prediction.")
        image_patch = np.expand_dims(image_patch, axis=0)
        # Normalize if needed
        # image_patch = image_patch / 255.0  # Already normalized
        probabilities = self.model.predict(image_patch)
        return probabilities[0][0]

def predict_point(predictor, image, x, y, target_size=(64, 64)):
    """Make prediction for a single point."""
    image_patch = get_image_patch(image, x, y, target_size)
    try:
        probability = predictor.predict(image_patch)
        return probability
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None

def create_mask_for_processing(image, threshold_ranges):
    """
    Creates a binary mask indicating areas that should be processed for tree detection.
    Filters out unnatural colors unlikely to be trees and expands processable areas.
    
    Args:
        image: RGB image normalized to [0, 1]
        threshold_ranges: dict with color ranges to exclude
    
    Returns:
        Binary mask where True indicates areas to process
    """
    
    mask = np.ones(image.shape[:2], dtype=bool)
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    
    channel_std = np.std(image, axis=2)
    mean_brightness = np.mean(image, axis=2)
    saturation = np.max(image, axis=2) - np.min(image, axis=2)
    
    # 1. Detect gray/white areas (low color variance AND high brightness)
    gray_mask = (channel_std < 0.08) & (mean_brightness > 0.5)
    
    # 2. Detect blue-black areas (high blue relative to red/green AND low overall brightness)
    blue_black_mask = (b > (r + 0.05)) & (b > (g + 0.05)) & (mean_brightness < 0.25)
    
    # 3. Detect highly saturated/neon colors
    neon_mask = (saturation > 0.5) & (mean_brightness > 0.4)
    
    # 4. Detect pure red areas
    red_mask = (r > (g + 0.15)) & (r > (b + 0.15)) & (r > 0.3)
    
    # 5. Detect brown/tan areas (high red, moderate green, low blue)
    brown_mask = (r > (g + 0.1)) & (g > (b + 0.1)) & (r > 0.4) & (b < 0.3)
    
    # 6. Detect pure/bright yellow (high red AND green, low blue)
    yellow_mask = (r > 0.5) & (g > 0.5) & (b < 0.3) & (np.abs(r - g) < 0.1)
    
    # 7. Super bright areas (likely glare or artificial)
    bright_mask = mean_brightness > 0.8
    
    # Combine all masks
    combined_mask = (gray_mask | blue_black_mask | neon_mask | red_mask | 
                    brown_mask | yellow_mask | bright_mask)
    
    # Convert to uint8 for morphological operations
    mask_uint8 = (~combined_mask).astype(np.uint8) * 255
    
    # Create a kernel for dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Dilate the processable areas (this will shrink the areas we skip)
    dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=1)
    
    # Convert back to boolean and update mask
    mask = dilated_mask > 0
    
    return mask

def process_image(predictor, image_path, target_size=(64, 64), 
                                  grid_spacing=20, threshold=0.5):
    """Process a single image with color-based preprocessing."""
    
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    h, w = image.shape[:2]
    
    process_mask = create_mask_for_processing(image, {})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Show original image
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Show image with skipped points
    ax2.imshow(image)
    ax2.set_title('Skipped Points (Red)')
    ax2.axis('off')
    
    tree_count = 0
    circles = []
    skipped_points = []
    y_range = range(grid_spacing//2, h - grid_spacing//2 + 1, grid_spacing)
    x_range = range(grid_spacing//2, w - grid_spacing//2 + 1, grid_spacing)
    
    # Count total processable points and collect skipped points
    processable_points = 0
    total_points = 0
    
    for y in y_range:
        for x in x_range:
            total_points += 1
            if process_mask[y, x]:
                processable_points += 1
            else:
                skipped_points.append((x, y))
    
    # Progress tracking
    processed = 0
    last_progress = 0
    
    # Process only points within the mask
    for y in y_range:
        for x in x_range:
            if process_mask[y, x]:
                pred = predict_point(predictor, image, x, y, target_size)
                processed += 1
                current_progress = (processed * 100) // processable_points
                if current_progress >= last_progress + 10:
                    print(f"Processing image {Path(image_path).name}: {current_progress}% complete...")
                    last_progress = current_progress
                
                if pred is not None and pred >= threshold:
                    tree_count += 1
                    circles.append(plt.Circle((x, y), grid_spacing // 2, color='r', fill=False))
    
    for circle in circles:
        ax1.add_patch(circle)
    
    # Plot skipped points
    skipped_x = [p[0] for p in skipped_points]
    skipped_y = [p[1] for p in skipped_points]
    ax2.scatter(skipped_x, skipped_y, c='red', s=10, alpha=0.5, label='Skipped Points')
    
    # Update titles with results
    skipped_percentage = (len(skipped_points) / total_points) * 100
    
    ax1.set_title(f'Detected Trees: {tree_count}')
    ax2.set_title(f'Skipped Points in Red\n({skipped_percentage:.1f}% of points skipped)')
    
    plt.draw()
    print(f"\nCompleted {Path(image_path).name}:")
    print(f"- Found {tree_count} likely trees")
    print(f"- Skipped {skipped_percentage:.1f}% of points")
    print(f"- Processed {processable_points} points instead of {total_points}")
    print("Press any key to proceed to the next image...")
    plt.waitforbuttonpress()
    plt.close(fig)

def main():
    try:
        predictor = StandalonePredictor()
        predictor.load('tree_cnn_classifier.h5')
        script_dir = Path(__file__).parent
        image_folder = script_dir / "images"
        
        # Alternative Method 2: Use relative path from current working directory
        # image_folder = Path("./images")
        
        image_files = list(image_folder.glob('*.png')) + list(image_folder.glob('*.jpg'))

        if not image_files:
            print(f"No images found in {image_folder}")
            return

        current_image_idx = 0

        while current_image_idx < len(image_files):
            image_path = image_files[current_image_idx]
            process_image(
                predictor,
                image_path,
                target_size=(64, 64),
                grid_spacing=20,
                threshold=0.5
            )
            current_image_idx += 1

    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()