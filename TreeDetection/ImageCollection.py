import numpy as np
import requests
from PIL import Image
from io import BytesIO
import math
import time
from pathlib import Path
import json

class SatelliteMapUtils:
    """
    Utility class for downloading and managing satellite imagery from Mapbox.
    Handles image fetching, coordinate calculations, and metadata management.
    """
    EARTH_CIRCUMFERENCE = 40075016.686  # in meters
    TILE_SIZE = 512
    MAX_IMAGE_SIZE = 1280

    def __init__(self, base_dir=None):
        """
        Initialize the SatelliteMapUtils with a base directory for saving images.
        
        Args:
            base_dir (str or Path, optional): Base directory for saving images.
                                            Defaults to './images' in current directory.
        """
        # Set path to images directory
        self.images_dir = Path(base_dir) if base_dir else Path("./images")
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.images_dir / "image_metadata.json"
        print(f"Images will be saved to: {self.images_dir}")
        print(f"Metadata will be saved to: {self.metadata_file}")

    @staticmethod
    def calculate_zoom_for_meters(meters, latitude):
        """Calculate appropriate zoom level based on desired image size in meters."""
        latitude_radians = math.radians(latitude)
        zoom = math.log2((SatelliteMapUtils.MAX_IMAGE_SIZE * math.cos(latitude_radians) * 2 * math.pi * 6378137) / (meters * 256))
        return max(0, min(22, math.floor(zoom - 1)))

    @classmethod
    def fetch_satellite_image(cls, access_token, center_lat, center_lon, meters=500):
        """
        Fetch a satellite image for the given coordinates.
        
        Args:
            access_token (str): Mapbox access token
            center_lat (float): Center latitude
            center_lon (float): Center longitude
            meters (int): Approximate width/height of area to capture in meters
            
        Returns:
            tuple: (PIL Image, corner coordinates dict, zoom level)
        """
        zoom = cls.calculate_zoom_for_meters(meters, center_lat)
        print(f"Using zoom level: {zoom}")
        
        img = cls.fetch_mapbox_satellite(access_token, center_lon, center_lat, zoom, cls.MAX_IMAGE_SIZE)
        
        meters_per_pixel = (math.cos(math.radians(center_lat)) * 2 * math.pi * 6378137) / (256 * math.pow(2, zoom))
        actual_meters = meters_per_pixel * cls.MAX_IMAGE_SIZE
        
        bounds = cls.calculate_bounds(center_lon, center_lat, meters)
        west, south, east, north = bounds
        corner_coords = {
            "northwest": (west, north),
            "northeast": (east, north),
            "southwest": (west, south),
            "southeast": (east, south)
        }
        
        return img, corner_coords, zoom

    @staticmethod
    def fetch_mapbox_satellite(access_token, center_lon, center_lat, zoom, size):
        """Make request to Mapbox API for satellite imagery."""
        url = (
            f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/"
            f"{center_lon},{center_lat},{zoom},0,0/"
            f"{size}x{size}?"
            f"access_token={access_token}"
        )
        
        print(f"Requesting URL: {url}")
        
        response = requests.get(url)
        response.raise_for_status()
        
        if 'image' not in response.headers['Content-Type']:
            raise Exception(f"Unexpected content type: {response.headers['Content-Type']}")
        
        return Image.open(BytesIO(response.content)).convert('RGB')

    @staticmethod
    def calculate_bounds(lon, lat, meters):
        """Calculate bounding box coordinates given center point and size."""
        lat_span = (meters / 2) / 111320.0
        lon_span = (meters / 2) / (111320.0 * abs(np.cos(np.radians(lat))))
        return (lon - lon_span, lat - lat_span, lon + lon_span, lat + lat_span)

    def batch_download_images(self, token, locations, meters=500):
        """
        Download multiple satellite images from a list of locations.
        
        Args:
            token (str): Mapbox access token
            locations (list): List of tuples (name, latitude, longitude)
            meters (int): Approximate width/height of each image in meters
            
        Returns:
            dict: Metadata for all downloaded images
        """
        metadata = {}
        
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                print(f"Loaded existing metadata for {len(metadata)} images")
            except Exception as e:
                print(f"Error loading existing metadata: {e}")
        
        for idx, (name, lat, lon) in enumerate(locations, 1):
            try:
                print(f"\nFetching image {idx}/{len(locations)}: {name}")
                
                filename = f"satellite_{name.lower().replace(' ', '_')}_{idx}.png"
                filepath = self.images_dir / filename
                
                if filepath.exists():
                    print(f"Skipping {filename} - already exists")
                    continue
                
                img, coords, zoom = self.fetch_satellite_image(token, lat, lon, meters)
                
                img.save(filepath)
                print(f"Saved {filename}")
                
                metadata[filename] = {
                    "name": name,
                    "coordinates": {
                        "latitude": lat,
                        "longitude": lon
                    },
                    "zoom": zoom,
                    "corners": coords,
                    "size_meters": meters,
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                with open(self.metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Sleep to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"Error downloading {name}: {e}")
                continue
        
        return metadata

def get_sample_locations():
    """
    Return a list of sample locations for downloading satellite imagery.
    Each location is a tuple of (name, latitude, longitude).
    
    Add or modify locations as needed.
    """
    return [
        # Example location - add your own locations here
        ("Central_Park", 40.785091, -73.968285),
        # ("Location_Name", latitude, longitude),
    ]

if __name__ == "__main__":
    # Create a config.py file with your Mapbox access token
    try:
        from config import MAPBOX_TOKEN
    except ImportError:
        print("Please create a config.py file with your Mapbox access token:")
        print('MAPBOX_TOKEN = "your_token_here"')
        exit(1)
    
    utils = SatelliteMapUtils()
    locations = get_sample_locations()
    
    print(f"Starting download of {len(locations)} locations...")
    metadata = utils.batch_download_images(MAPBOX_TOKEN, locations)
    
    print("\nDownload complete!")
    print(f"Downloaded {len(metadata)} images")
    print(f"Images saved in: {utils.images_dir}")
    print(f"Metadata saved in: {utils.metadata_file}")