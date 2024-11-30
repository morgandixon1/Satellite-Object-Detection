import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.widgets import Button
import platform
import json
import time
import os
from pathlib import Path
import cv2
import traceback

class NotTreeVisualizationTools:
    """Enhanced visualization tools with mask generation capabilities for non-tree annotations"""
    def __init__(self, image, image_name=None):
        # Ensure image is in correct format
        if image is not None:
            if not isinstance(image, np.ndarray):
                raise ValueError("Image must be a numpy array")
            if image.dtype != np.uint8:
                print(f"Converting image from {image.dtype} to uint8")
                image = (image * 255).astype(np.uint8) if image.dtype == np.float32 else image.astype(np.uint8)
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
                image = image[:, :, :3]  # Convert to RGB
                
        self.image = image
        self.image_name = image_name
        print(f"Initialized with image shape: {image.shape if image is not None else None}, dtype: {image.dtype if image is not None else None}")
        self.base_radius = 6
        self.min_radius = 4
        self.max_radius_multiplier = 1.4
        self.points = []
        self.radii = []
        self.circles = []
        self.predictions = []
        self.circle_colors = []
        self.dragging = False
        self.start_drag = None
        self.drag_rect = None
        self.add_mode = True
        self.proceed_flag = False
        self.next_image_flag = False
        
        # Setup project directories with new folder names
        self.setup_directories()
        
        # Try to load previous state if it exists
        if image_name:
            self.load_previous_state(image_name)
        
        # Setup plot with fixed layout
        self.fig = plt.figure(figsize=(8, 8))
        self.fig.set_tight_layout(False)
        self.ax = self.fig.add_subplot(111)
        if image is not None:
            self.ax.imshow(image)
        else:
            self.ax.set_title("No image loaded")
            
        self.current_size = 6
        self.size_step = 1.0
        self.min_size = 3
        self.max_size = 12
        
        # Add brush-related attributes
        self.brush_mode = False
        self.brush_radius = 20
        self.brush_indicator = None
        self.last_brush_pos = None
        
        self.connect_events()
        self.setup_buttons()
        self.fig.canvas.draw_idle()
        self.update_title()


    def setup_buttons(self):
        """Setup UI buttons with improved layout"""
        button_width = 0.15
        spacing = 0.02
        bottom_margin = 0.02
        height = 0.05

        button_configs = [
            ('Save & Next', 'lightgreen', self.save_and_next),
            ('Save Points', 'lightyellow', self.save_points),
            ('Load Points', 'lightpink', self.load_points_dialog),
            ('Add/Remove', 'lightblue', self.toggle_mode),
            ('Clear All', 'salmon', self.clear_all)
        ]

        start_x = 0.98 - button_width
        self.buttons = {}
        for i, (name, color, callback) in enumerate(button_configs):
            ax_btn = self.fig.add_axes([
                start_x - (i * (button_width + spacing)),
                bottom_margin,
                button_width,
                height
            ])
            btn = Button(ax_btn, name, color=color)
            btn.on_clicked(callback)
            self.buttons[name] = btn

        self.ax.set_position([0.1, 0.1, 0.85, 0.85])

    def connect_events(self):
        """Connect mouse events and keyboard events"""
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

    def on_scroll(self, event):
        """Handle scroll events to adjust brush size"""
        if self.brush_mode:
            if event.button == 'up':
                self.brush_radius = min(100, self.brush_radius + 5)
            else:
                self.brush_radius = max(10, self.brush_radius - 5)
            title = self.ax.get_title()
            if "BRUSH MODE" in title:
                self.ax.set_title(f"BRUSH MODE ({self.brush_radius}px) | {title.split('|')[1].strip()}")
                self.fig.canvas.draw_idle()

    def handle_brush(self, y, x):
        """Updated handle_brush with improved incorrect point logging"""
        if self.last_brush_pos is None:
            self.last_brush_pos = (y, x)
            points_to_check = [(y, x)]
        else:
            y0, x0 = self.last_brush_pos
            points_to_check = []
            steps = max(abs(y - y0), abs(x - x0)) * 2
            if steps > 0:
                for i in range(steps + 1):
                    t = i / steps
                    check_y = int(y0 + (y - y0) * t)
                    check_x = int(x0 + (x - x0) * t)
                    points_to_check.append((check_y, check_x))
            self.last_brush_pos = (y, x)

        if not self.add_mode:
            points_to_remove = []
            for i, ((py, px), r) in enumerate(zip(self.points, self.radii)):
                for check_y, check_x in points_to_check:
                    if (py - check_y)**2 + (px - check_x)**2 <= self.brush_radius**2:
                        points_to_remove.append((py, px))
                        break
            
            if points_to_remove:
                self.remove_points(points_to_remove)
                self.update_display()

    def toggle_mode(self, event):
        """Toggle between Add and Remove mode"""
        self.add_mode = not self.add_mode
        self.buttons['Add/Remove'].label.set_text('Add Mode' if self.add_mode else 'Remove Mode')
        self.buttons['Add/Remove'].color = 'lightblue' if self.add_mode else 'salmon'
        self.fig.canvas.draw_idle()

    def clear_display(self):
        """Clear all circles from display"""
        for circle in self.circles:
            circle.remove()
        self.circles = []
        
    def clear_predictions(self):
        """Clear only prediction circles"""
        self.predictions = []

    def add_prediction(self, point_dict):
        """Add a prediction point with special visualization"""
        self.predictions.append(point_dict)
        self.update_display()

    def update_display(self):
        """Update the display with current points and circles"""
        self.ax.clear()
        if self.image is not None:
            self.ax.imshow(self.image)
        
        if not self.circle_colors or len(self.circle_colors) != len(self.points):
            self.circle_colors = ['y'] * len(self.points)
        
        for (y, x), radius, color in zip(self.points, self.radii, self.circle_colors):
            circle = plt.Circle((x, y), radius, fill=False, color=color)
            self.ax.add_patch(circle)
            
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        """Handle mouse release for adding or removing points"""
        if not self.dragging or event.inaxes != self.ax:
            return

        if self.drag_rect:
            self.drag_rect.remove()
            self.drag_rect = None

        drag_duration = time.time() - self.drag_start_time
        y1, x1 = self.start_drag
        y2, x2 = int(event.ydata), int(event.xdata)
        if drag_duration < 0.2 and abs(y2 - y1) < 3 and abs(x2 - x1) < 3:
            # Single point operation
            if self.add_mode:
                self.add_single_point(y1, x1)
            else:
                self.remove_single_point(y1, x1)
        else:
            # Drag operation
            y_min, y_max = sorted([y1, y2])
            x_min, x_max = sorted([x1, x2])

            if self.add_mode:
                self.add_points_in_area(y_min, y_max, x_min, x_max)
            else:
                points_to_remove = [
                    (p, r) for p, r in zip(self.points, self.radii)
                    if y_min <= p[0] <= y_max and x_min <= p[1] <= x_max
                ]
                for (removed_y, removed_x), removed_radius in points_to_remove:
                    self.points.remove((removed_y, removed_x))
                    self.radii.remove(removed_radius)
                    self.save_incorrect_point(removed_y, removed_x, removed_radius)

                print(f"Removed and logged {len(points_to_remove)} points")

        self.dragging = False
        self.start_drag = None
        self.update_display()

    def check_point_overlap(self, y, x, spacing):
        """Check if a point would overlap with existing points"""
        for existing_point in self.points:
            dist = np.sqrt((y - existing_point[0])**2 + (x - existing_point[1])**2)
            if dist < spacing:
                return True
        return False
    
    def is_valid_point(self, y, x, spacing=None):
        """Check if a point is valid (within bounds and not overlapping)"""
        if spacing is None:
            spacing = int(self.base_radius * 2)
            
        if not (0 <= y < self.image.shape[0] and 0 <= x < self.image.shape[1]):
            return False
            
        return not any(
            np.sqrt((y - py)**2 + (x - px)**2) < spacing 
            for py, px in self.points
        )

    def on_key_press(self, event):
        """Handle key press events"""
        if event.key == 'b':
            self.brush_mode = True
            self.ax.set_title(f"BRUSH MODE ({self.brush_radius}px) | " + self.ax.get_title())
            self.fig.canvas.draw_idle()
        elif event.key in ['+', '=']:
            self.current_size = min(self.max_size, self.current_size + self.size_step)
            self.update_title()
        elif event.key == '-':
            self.current_size = max(self.min_size, self.current_size - self.size_step)
            self.update_title()
        elif event.key == 'h':
            print("\nKeyboard Controls:")
            print("+ or = : Increase tree size")
            print("-     : Decrease tree size")
            print("b     : Toggle brush mode")
            print("h     : Show this help message")
            print(f"Current size: {self.current_size:.1f}px")

    def generate_random_radius(self):
        """Generate a random radius with controlled variation"""
        radius = np.random.triangular(
            self.min_radius,
            self.base_radius,
            self.base_radius * self.max_radius_multiplier
        )
        return radius

    def remove_single_point(self, y, x):
        """Updated remove_single_point with improved incorrect point logging"""
        if not self.points:
            return
        distances = [(i, np.sqrt((y - py)**2 + (x - px)**2)) 
                    for i, (py, px) in enumerate(self.points)]
        closest_idx, min_dist = min(distances, key=lambda x: x[1])
        if min_dist <= self.radii[closest_idx] * 1.5:
            removed_y, removed_x = self.points[closest_idx]
            removed_radius = self.radii[closest_idx]
            
            # Remove point
            self.points.pop(closest_idx)
            self.radii.pop(closest_idx)
            
            # Log the incorrect point
            self.save_incorrect_point(removed_y, removed_x, removed_radius)
            print(f"Removed and logged incorrect point near ({removed_x}, {removed_y})")
            self.update_display()

    def on_press(self, event):
        """Handle mouse press"""
        if event.inaxes != self.ax:
            return
        self.start_drag = (int(event.ydata), int(event.xdata))
        self.dragging = True
        self.drag_start_time = time.time()

    def on_key_release(self, event):
        """Handle key release events and clean up brush indicator"""
        if event.key == 'b':
            self.brush_mode = False
            self.last_brush_pos = None
            if hasattr(self, 'brush_indicator') and self.brush_indicator:
                self.brush_indicator.remove()
                self.brush_indicator = None
            title = self.ax.get_title()
            if "BRUSH MODE" in title:
                self.ax.set_title(title.split("|")[1].strip())
            self.fig.canvas.draw_idle()

    def on_motion(self, event):
        """Handle drag motion"""
        if not (self.dragging and event.inaxes == self.ax):
            return
            
        if self.brush_mode:
            self.handle_brush(int(event.ydata), int(event.xdata))
            self.update_display()
        else:
            if self.drag_rect:
                self.drag_rect.remove()
            y1, x1 = self.start_drag
            y2, x2 = int(event.ydata), int(event.xdata)
            self.drag_rect = plt.Rectangle(
                (min(x1, x2), min(y1, y2)),
                abs(x2 - x1), abs(y2 - y1),
                linewidth=1, edgecolor='blue', facecolor='none', linestyle='--'
            )
            self.ax.add_patch(self.drag_rect)
            self.fig.canvas.draw_idle()

    def on_motion(self, event):
        """Handle drag motion with visible brush area"""
        if not (event.inaxes == self.ax):
            return
                
        if self.brush_mode and event.inaxes == self.ax:
            if hasattr(self, 'brush_indicator') and self.brush_indicator:
                self.brush_indicator.remove()
                
            self.brush_indicator = plt.Circle(
                (event.xdata, event.ydata), 
                self.brush_radius, 
                color='yellow' if self.add_mode else 'red',
                alpha=0.2,
                fill=True
            )
            self.ax.add_patch(self.brush_indicator)
            
            if self.dragging:
                self.handle_brush(int(event.ydata), int(event.xdata))
                self.update_display()
            
            self.fig.canvas.draw_idle()
        elif self.dragging:
            if self.drag_rect:
                self.drag_rect.remove()
            y1, x1 = self.start_drag
            y2, x2 = int(event.ydata), int(event.xdata)
            self.drag_rect = plt.Rectangle(
                (min(x1, x2), min(y1, y1)),
                abs(x2 - x1), abs(y2 - y1),
                linewidth=1, edgecolor='blue', facecolor='none', linestyle='--'
            )
            self.ax.add_patch(self.drag_rect)
            self.fig.canvas.draw_idle()

    def add_points_in_area(self, y_min, y_max, x_min, x_max):
        """Add points in area with more consistent sizing"""
        size_categories = {
            'slightly_smaller': {
                'radius_range': (self.current_size * 0.9, self.current_size),
                'spacing_multiplier': 1.5,
                'weight': 0.3
            },
            'current': {
                'radius_range': (self.current_size, self.current_size * 1.05),
                'spacing_multiplier': 1.5,
                'weight': 0.4
            },
            'slightly_larger': {
                'radius_range': (self.current_size * 1.05, self.current_size * 1.1),
                'spacing_multiplier': 1.5,
                'weight': 0.3
            }
        }
        
        potential_points = []
        base_spacing = int(self.current_size * 2)
        
        for y in range(y_min, y_max + 1, base_spacing):
            for x in range(x_min, x_max + 1, base_spacing):
                offset_y = np.random.randint(-1, 2)
                offset_x = np.random.randint(-1, 2)
                new_y = int(y + offset_y)
                new_x = int(x + offset_x)
                
                if not (0 <= new_y < self.image.shape[0] and 0 <= new_x < self.image.shape[1]):
                    continue
                
                size_type = np.random.choice(
                    list(size_categories.keys()),
                    p=[cat['weight'] for cat in size_categories.values()]
                )
                category = size_categories[size_type]
                
                min_rad, max_rad = category['radius_range']
                radius = np.random.uniform(min_rad, max_rad)
                
                potential_points.append({
                    'y': new_y,
                    'x': new_x,
                    'radius': radius,
                    'spacing_multiplier': category['spacing_multiplier']
                })
        
        np.random.shuffle(potential_points)
        points_added = 0
        
        for point in potential_points:
            y, x = point['y'], point['x']
            radius = point['radius']
            spacing_multiplier = point['spacing_multiplier']
            is_valid = True   
            for (py, px), r in zip(self.points, self.radii):
                dist = np.sqrt((y - py)**2 + (x - px)**2)
                min_allowed_distance = max(radius, r) * spacing_multiplier
                if dist < min_allowed_distance:
                    is_valid = False
                    break
            
            if is_valid:
                self.points.append((y, x))
                self.radii.append(radius)
                points_added += 1
        
        print(f"Added {points_added} trees (size ~{self.current_size:.1f}px) in selected area")
        self.update_display()

    def check_point_overlap(self, y, x, radius=None):
        """Enhanced overlap check using radius-based spacing"""
        if radius is None:
            radius = self.generate_random_radius()
        
        for (py, px), r in zip(self.points, self.radii):
            dist = np.sqrt((y - py)**2 + (x - px)**2)
            min_allowed_distance = max(radius, r) * 2
            if dist < min_allowed_distance:
                return True
        return False

    def add_single_point(self, y, x):
        """Add a single point with current size and minimal variation"""
        offset_y = np.random.randint(-1, 2)
        offset_x = np.random.randint(-1, 2)
        
        y += offset_y
        x += offset_x
        if not (0 <= y < self.image.shape[0] and 0 <= x < self.image.shape[1]):
            return
        variation = np.random.uniform(-0.05, 0.05)
        radius = self.current_size * (1 + variation)
        if not self.check_point_overlap(y, x, radius * 1.5):
            self.points.append((y, x))
            self.radii.append(radius)
            print(f"Added tree at ({x}, {y}) with radius {radius:.1f}")
            self.update_display()
        else:
            print("Point overlaps with existing tree")

    def finish(self, event):
        """Signal to proceed with training"""
        self.proceed_flag = True
        plt.close(self.fig)

    def generate_mask(self):
        """Generate binary mask from circle annotations"""
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        
        for (y, x), radius in zip(self.points, self.radii):
            y, x = int(y), int(x)
            radius = int(radius)
            cv2.circle(mask, (x, y), radius, 1, -1)
            
        return mask

    def save_mask(self):
        """Save binary mask to file with not-tree specific naming"""
        if not self.image_name:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            mask_filename = f'not_tree_mask_{timestamp}.png'
        else:
            mask_filename = f'{Path(self.image_name).stem}_not_tree_mask.png'
            
        mask = self.generate_mask()
        mask_path = self.masks_dir / mask_filename
        cv2.imwrite(str(mask_path), mask * 255)
        print(f"Saved not-tree mask to {mask_path}")
        return mask_path

    def save_points(self, event):
        """Save circle annotations with consistent naming for not-tree dataset"""
        if not self.image_name:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            annotation_filename = f'not_tree_points_{timestamp}.json'
        else:
            annotation_filename = f'{Path(self.image_name).stem}_not_tree_annotation.json'
            
        annotation_path = self.annotations_dir / annotation_filename
            
        circle_data = {
            'points': self.points,
            'radii': self.radii,
            'image_shape': self.image.shape[:2],
            'metadata': {
                'base_radius': self.base_radius,
                'min_radius': self.min_radius,
                'max_radius_multiplier': self.max_radius_multiplier,
                'timestamp': time.strftime('%Y%m%d_%H%M%S'),
                'total_points': len(self.points),
                'image_name': self.image_name,
                'dataset_type': 'not_tree'
            }
        }
        
        try:
            with open(annotation_path, 'w') as f:
                json.dump(circle_data, f, indent=2)
            print(f"Successfully saved {len(self.points)} not-tree points to {annotation_path}")
            mask_path = self.save_mask()
            self.ax.set_title(f"Saved {len(self.points)} not-tree points and mask")
            self.fig.canvas.draw_idle()
            
        except Exception as e:
            print(f"Error saving data: {str(e)}")
            self.ax.set_title("Error saving data!")
            self.fig.canvas.draw_idle()

    def save_and_next(self, event):
        """Save current annotations and masks, then signal to move to next image"""
        self.save_points(event)
        self.next_image_flag = True
        plt.close(self.fig)

    @staticmethod
    def get_image_files(directory):
        """Get list of image files in directory with proper path handling"""
        project_root = Path(__file__).parent.absolute()
        images_dir = project_root / "images"
        print(f"Project root: {project_root}")
        print(f"Images directory: {images_dir}")
        
        image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
        files = []
        if not images_dir.exists():
            print(f"Warning: Images directory not found at {images_dir}")
            return files
            
        for ext in image_extensions:
            found_files = list(images_dir.glob(f'*{ext}'))
            files.extend(found_files)
            if found_files:
                print(f"Found {len(found_files)} files with extension {ext}")
                
        for file in files:
            print(f"Found image file: {file}")
            
        return sorted(files)

    @staticmethod
    def load_points(filename):
        """Static method to load saved circle data"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            required_keys = ['points', 'radii', 'image_shape', 'metadata']
            if not all(key in data for key in required_keys):
                raise ValueError("Invalid file format - missing required data")
            
            if len(data['points']) != len(data['radii']):
                raise ValueError("Data corruption - points and radii counts don't match")
            
            print(f"Successfully loaded {len(data['points'])} points")
            return data
            
        except Exception as e:
            print(f"Error loading points: {str(e)}")
            return None

    def create_from_saved_data(image, saved_data):
        """Create new visualization from saved data"""
        if saved_data is None:
            return None
            
        if image.shape[:2] != tuple(saved_data['image_shape']):
            print("Warning: Current image dimensions don't match saved data")
        
        vis = TreeVisualizationTools(image)
        vis.points = saved_data['points']
        vis.radii = saved_data['radii']
        
        if 'metadata' in saved_data:
            vis.base_radius = saved_data['metadata'].get('base_radius', vis.base_radius)
            vis.min_radius = saved_data['metadata'].get('min_radius', vis.min_radius)
            vis.max_radius_multiplier = saved_data['metadata'].get('max_radius_multiplier', vis.max_radius_multiplier)
        
        vis.update_display()
        return vis
    
    def load_points_direct(self, filepath):
        """Load points directly from a file path with improved error handling"""
        try:
            if not filepath or not isinstance(filepath, str):
                print("Invalid file path")
                return False
                
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, dict):
                print("Invalid JSON format: expected dictionary")
                return False
                
            if not all(key in data for key in ['points', 'radii', 'metadata']):
                print(f"Invalid file format. Missing required keys. Found keys: {list(data.keys())}")
                return False
                
            if not isinstance(data['points'], list) or not isinstance(data['radii'], list):
                print("Invalid data format: points and radii must be lists")
                return False
                
            if len(data['points']) != len(data['radii']):
                print("Data mismatch: number of points doesn't match number of radii")
                return False
                
            for point in data['points']:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    print("Invalid point format")
                    return False
                    
            self.points = [tuple(p) for p in data['points']]
            self.radii = data['radii']
            
            if 'metadata' in data and isinstance(data['metadata'], dict):
                self.base_radius = data['metadata'].get('base_radius', self.base_radius)
                self.min_radius = data['metadata'].get('min_radius', self.min_radius)
                self.max_radius_multiplier = data['metadata'].get('max_radius_multiplier', self.max_radius_multiplier)
            
            self.update_display()
            print(f"Successfully loaded {len(self.points)} points")
            
            old_title = self.ax.get_title()
            self.ax.set_title(f"Loaded {len(self.points)} points")
            self.fig.canvas.draw_idle()
            
            timer = self.fig.canvas.new_timer(interval=2000)
            timer.add_callback(lambda: self.ax.set_title(old_title))
            timer.add_callback(self.fig.canvas.draw_idle)
            timer.start()
            
            return True
            
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            return False
        except json.JSONDecodeError:
            print(f"Invalid JSON format in file: {filepath}")
            return False
        except Exception as e:
            print(f"Error loading points: {str(e)}")
            return False

    def load_points_dialog(self, event=None):
        """Load points from the hardcoded file path"""
        filepath = "/TreeDetection/tree_points_20241120_185556.json"
        print(f"Loading points from: {filepath}")
        return self.load_points_direct(filepath)

    def fallback_load(self):
        """Fallback method to load most recent points file"""
        try:
            point_files = [f for f in os.listdir('.') 
                        if f.startswith('tree_points_') and f.endswith('.json')]
            
            if not point_files:
                print("No saved point files found in current directory")
                return False
                
            latest_file = max(point_files, key=os.path.getctime)
            print(f"Attempting to load most recent file: {latest_file}")
            return self.load_points_direct(latest_file)
            
        except Exception as e:
            print(f"Fallback load failed: {str(e)}")
            return False

    def setup_directories(self):
        """Create necessary project directories with new names for not-tree dataset"""
        self.project_dir = Path(__file__).parent
        self.images_dir = self.project_dir / "images"
        self.annotations_dir = self.project_dir / "not_tree_annotations"
        self.masks_dir = self.project_dir / "not_tree_masks"
        self.incorrect_points_dir = self.project_dir / "not_tree_incorrect_points"
        
        for directory in [self.images_dir, self.annotations_dir, 
                         self.masks_dir, self.incorrect_points_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Created or verified directory: {directory}")

    def load_previous_state(self, image_name):
        """Load previous not-tree annotations if they exist"""
        annotation_filename = f'{Path(image_name).stem}_not_tree_annotation.json'
        annotation_path = self.annotations_dir / annotation_filename
        
        if annotation_path.exists():
            try:
                with open(annotation_path, 'r') as f:
                    data = json.load(f)
                
                if all(key in data for key in ['points', 'radii', 'metadata']):
                    user_input = input(f"\nFound previous not-tree annotations for {image_name}. Would you like to:\n"
                                     "1. Load and edit previous annotations\n"
                                     "2. Start fresh (previous will be backed up)\n"
                                     "Enter 1 or 2: ")
                    
                    if user_input.strip() == "1":
                        self.points = [tuple(p) for p in data['points']]
                        self.radii = data['radii']
                        print(f"Loaded {len(self.points)} not-tree points from previous session")
                        
                        if 'metadata' in data:
                            self.base_radius = data['metadata'].get('base_radius', self.base_radius)
                            self.min_radius = data['metadata'].get('min_radius', self.min_radius)
                            self.max_radius_multiplier = data['metadata'].get('max_radius_multiplier', 
                                                                            self.max_radius_multiplier)
                    else:
                        backup_path = self.annotations_dir / f'{Path(image_name).stem}_not_tree_backup_{int(time.time())}.json'
                        import shutil
                        shutil.copy2(annotation_path, backup_path)
                        print(f"Previous not-tree annotations backed up to {backup_path}")
                        
            except Exception as e:
                print(f"Error loading previous not-tree annotations: {str(e)}")
                print("Starting fresh...")

    def clear_all(self, event):
        """Clear all annotations after confirmation"""
        if len(self.points) > 0:
            user_input = input("Are you sure you want to clear all annotations? (y/n): ")
            if user_input.lower().strip() == 'y':
                self.points = []
                self.radii = []
                self.update_display()
                print("All annotations cleared")

    def save_incorrect_point(self, y, x, radius):
        """Save removed point to incorrect points file for training improvement"""
        try:
            if self.image_name:
                base_name = Path(self.image_name).stem
            else:
                base_name = time.strftime('%Y%m%d_%H%M%S')
            
            incorrect_points_file = self.project_dir / 'incorrect_points' / f'{base_name}_incorrect.json'
            incorrect_points_file.parent.mkdir(exist_ok=True)
            incorrect_points = []
            if incorrect_points_file.exists():
                try:
                    with open(incorrect_points_file, 'r') as f:
                        data = json.load(f)
                        incorrect_points = data.get('points', [])
                except json.JSONDecodeError:
                    print("Error reading existing incorrect points file")
            
            incorrect_points.append({
                'y': int(y),
                'x': int(x),
                'radius': int(radius),
                'timestamp': time.strftime('%Y%m%d_%H%M%S')
            })
            
            with open(incorrect_points_file, 'w') as f:
                json.dump({
                    'image_name': self.image_name,
                    'image_shape': self.image.shape[:2] if self.image is not None else None,
                    'points': incorrect_points
                }, f, indent=2)
            
            print(f"Saved incorrect point at ({x}, {y}) to {incorrect_points_file}")
            
        except Exception as e:
            print(f"Error saving incorrect point: {str(e)}")

    def update_title(self):
        """Update the plot title with current size information"""
        mode = "Add Mode" if self.add_mode else "Remove Mode"
        self.ax.set_title(f"{mode} | Size: {self.current_size:.1f}px (Use +/- to adjust)")
        self.fig.canvas.draw_idle()

def main():
    """Main function for batch processing images for not-tree dataset"""
    project_dir = Path(__file__).parent
    images_dir = project_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    print(f"Looking for images in: {images_dir}")
    image_files = [f for f in images_dir.glob("*") 
                   if f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}]
    
    print(f"Found {len(image_files)} image files")
    
    if not image_files:
        print("No image files found in images directory")
        print(f"Please add images to: {images_dir}")
        return

    print("\nStarting not-tree annotation process...")
    for i, image_file in enumerate(image_files):
        try:
            print(f"\nProcessing image {i+1}/{len(image_files)}: {image_file.name}")
            
            try:
                image = Image.open(image_file)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image = np.array(image)
                print(f"Loaded image shape: {image.shape}, dtype: {image.dtype}")
            except Exception as e:
                print(f"Error loading image {image_file.name}: {str(e)}")
                continue
            
            tool = NotTreeVisualizationTools(image, image_file.name)
            plt.show(block=True)
            
            if not tool.next_image_flag:
                print("Not-tree annotation cancelled by user")
                break
                
            plt.close(tool.fig)
            del tool
            
        except Exception as e:
            print(f"Error processing image {image_file.name}: {str(e)}")
            traceback.print_exc()
            continue
            
    print("\nFinished processing all images for not-tree dataset!")

if __name__ == "__main__":
    main()