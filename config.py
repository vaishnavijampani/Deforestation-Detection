"""
Configuration file for defining region, time periods, URLs for GEE,
and training hyperparameters.
"""

# GEE asset ID (2024 version)
HANSEN_ASSET = "UMD/hansen/global_forest_change_2024_v1_12"

# Region–of–interest (ROIs): India SDS via GeoJSON of your region or lat/lon box
# For you: you might draw a polygon in geojson or use WGS84 coordinates.
ROI_GEOJSON = "regions.geojson"

# Sentinel‑2 vs Forest‑loss periods
TIME_BEFORE = "2018-01-01"
TIME_AFTER = "2024-01-01"

# Patch size in meters and pixels
PATCH_SIZE_PX = 256
PATCH_SIZE_M = 256 * 10  # Sentinel‑2 native 10 m resolution

# NDVI band names (GEE uses these names in L2A)
NDVI_INPUT_BANDS = ["NDVI", "B04", "B08", "B03"]  # use Red (B04), NIR (B08), Green (B03)

# Train/test split fraction (by region)
SPLIT_BY_REGION = 0.7

# Training hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
MAX_EPOCHS = 30

# Paths for data export and training files
EXPORT_FOLDER = "export_samples"
PATCH_NPY_DIR = "patches_npz"
MODEL_WEIGHTS = "unet_best.pth"

# Model architecture hyperparams
ENCODER_CHANNELS = [64, 128, 256, 512]
