import numpy as np
import cv2
import json
from pathlib import Path
import random

class TreeDataset:
    @classmethod
    def load_dataset(cls):
        dataset = cls(
            images_dir='/TreeDetection/images',
            tree_annotations_dir='/TreeDetection/annotations',
            not_tree_annotations_dir='/TreeDetection/not_tree_annotations'
        )
        return dataset.get_patches_and_labels()

    def __init__(self, images_dir, tree_annotations_dir, not_tree_annotations_dir,
                 augment=True, pad_size=(64, 64)):
        self.images_dir = Path(images_dir)
        self.tree_annotations_dir = Path(tree_annotations_dir)
        self.not_tree_annotations_dir = Path(not_tree_annotations_dir)
        self.tree_patches = []
        self.not_tree_patches = []
        self.augment = augment
        self.pad_size = pad_size
        self.processed_images = 0
        self.max_images = None
        print("\nInitializing TreeDataset...")
        print(f"Images directory: {self.images_dir}")
        print(f"Tree Annotations directory: {self.tree_annotations_dir}")
        print(f"Not-Tree Annotations directory: {self.not_tree_annotations_dir}")
        self._extract_patches()

    def _extract_patches(self):
        image_files = sorted([
            f for f in self.images_dir.glob("*.png")
            if "tree" not in f.name and "not_tree" not in f.name
        ])
        print(f"\nFound {len(image_files)} image files")

        for img_path in image_files:
            print(f"\nProcessing {img_path.name}")

            if self.max_images and self.processed_images >= self.max_images:
                break

            try:
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"Warning: Could not read image {img_path.name}")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w = image.shape[:2]
                print(f"Image dimensions: {w}x{h}")

                # Process tree annotations with original naming pattern
                tree_ann_path = self.tree_annotations_dir / f"{img_path.stem}_annotation.json"
                if tree_ann_path.exists():
                    self._process_annotations(
                        image, tree_ann_path, is_tree=True
                    )
                else:
                    print(f"Warning: No tree annotation file found for {img_path.name}")

                # Process not-tree annotations with new naming pattern
                not_tree_ann_path = self.not_tree_annotations_dir / f"{img_path.stem}_not_tree_annotation.json"
                if not_tree_ann_path.exists():
                    self._process_annotations(
                        image, not_tree_ann_path, is_tree=False
                    )
                else:
                    print(f"Warning: No not-tree annotation file found for {img_path.name}")

                self.processed_images += 1

            except Exception as e:
                print(f"Error processing {img_path.name}: {str(e)}")
                continue

        # Balance the dataset
        self._balance_dataset()

        total_tree_samples = len(self.tree_patches)
        total_not_tree_samples = len(self.not_tree_patches)

        print(f"\nTotal tree samples: {total_tree_samples}")
        print(f"Total non-tree samples: {total_not_tree_samples}")

        if total_tree_samples == 0 or total_not_tree_samples == 0:
            raise ValueError("Insufficient samples collected for one or both classes!")

    def _process_annotations(self, image, ann_path, is_tree=True):
        """
        Process annotations and extract patches.
        """
        h, w = image.shape[:2]
        try:
            with open(ann_path) as f:
                annotation_data = json.load(f)

            if not (annotation_data.get("points") and annotation_data.get("radii")):
                print(f"Warning: Invalid annotation format in {ann_path}")
                return

            class_name = 'tree' if is_tree else 'not-tree'
            print(f"Found {len(annotation_data['points'])} annotated {class_name}s in {ann_path.name}")

            # Process each annotation
            for idx, (point, radius) in enumerate(zip(annotation_data["points"], annotation_data["radii"])):
                try:
                    if len(point) != 2:
                        print(f"Warning: Invalid point format at index {idx}: {point}")
                        continue
                    
                    y, x = point
                    x, y = int(x), int(y)
                    r = int(radius)

                    # Extract region
                    y1, y2 = max(0, y - r), min(h, y + r)
                    x1, x2 = max(0, x - r), min(w, x + r)
                    
                    # Check if the region is valid
                    if y1 >= y2 or x1 >= x2:
                        print(f"Warning: Invalid region coordinates at index {idx}: ({x1}, {y1}, {x2}, {y2})")
                        continue
                        
                    region = image[y1:y2, x1:x2]
                    padded_region = self._pad_image(region, self.pad_size)

                    if is_tree:
                        self.tree_patches.append(padded_region)
                    else:
                        self.not_tree_patches.append(padded_region)

                    # Augmentation
                    if self.augment:
                        augmented_regions = self._augment_region(region)
                        for aug_region in augmented_regions:
                            padded_augmented_region = self._pad_image(aug_region, self.pad_size)
                            if is_tree:
                                self.tree_patches.append(padded_augmented_region)
                            else:
                                self.not_tree_patches.append(padded_augmented_region)
                                
                except Exception as e:
                    print(f"Warning: Error processing point at index {idx}: {str(e)}")
                    continue

        except Exception as e:
            print(f"Error processing annotation file {ann_path}: {str(e)}")
            return

    def _balance_dataset(self):
        """
        Balance the dataset by sampling from the majority class.
        """
        num_tree = len(self.tree_patches)
        num_not_tree = len(self.not_tree_patches)
        print(f"\nBalancing the dataset...")
        print(f"Number of tree samples: {num_tree}")
        print(f"Number of non-tree samples: {num_not_tree}")

        if num_tree > num_not_tree:
            self.tree_patches = random.sample(self.tree_patches, num_not_tree)
        elif num_not_tree > num_tree:
            self.not_tree_patches = random.sample(self.not_tree_patches, num_tree)

        print(f"Balanced number of samples per class: {len(self.tree_patches)}")

    def _pad_image(self, img, target_size):
        height, width = img.shape[:2]
        pad_height = target_size[0] - height
        pad_width = target_size[1] - width

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

    def _augment_region(self, region):
        augmented_regions = []

        # Flip horizontally
        aug_flip_h = cv2.flip(region, 1)
        augmented_regions.append(aug_flip_h)
        # Flip vertically
        aug_flip_v = cv2.flip(region, 0)
        augmented_regions.append(aug_flip_v)
        # Rotate 90 degrees
        aug_rot_90 = cv2.rotate(region, cv2.ROTATE_90_CLOCKWISE)
        augmented_regions.append(aug_rot_90)
        # Rotate 180 degrees
        aug_rot_180 = cv2.rotate(region, cv2.ROTATE_180)
        augmented_regions.append(aug_rot_180)

        return augmented_regions

    def get_patches_and_labels(self):
        patches = self.tree_patches + self.not_tree_patches
        labels = [1] * len(self.tree_patches) + [0] * len(self.not_tree_patches)
        return patches, labels

def main():
    """
    Main function for running diagnostics and getting detailed statistics.
    This is used when running the file directly for testing/debugging.
    """
    dataset = TreeDataset.load_dataset()

    patches, labels = dataset
    print("\nDetailed Dataset Statistics:")
    print("----------------------------")
    total_samples = len(labels)
    total_trees = labels.count(1)
    total_not_trees = labels.count(0)

    print(f"\nTotal samples: {total_samples}")
    print(f"Total tree samples: {total_trees}")
    print(f"Total non-tree samples: {total_not_trees}")
    print("\nSample patches and labels:")
    for i in range(min(5, len(patches))):
        label = labels[i]
        patch = patches[i]
        print(f"Sample {i+1}: Label = {'Tree' if label == 1 else 'Not Tree'}, Patch shape = {patch.shape}")

if __name__ == "__main__":
    main()