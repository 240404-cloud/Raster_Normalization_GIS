RasterNormalize GIS
===================
Advanced Raster Data Standardization, Normalization & Visualization.

Workflow
--------
Upload -> Detect -> Standardize to TIFF when needed -> Inspect -> Analyze -> Min-Max Normalize -> Visualize -> Download.

Supported inputs: TIFF, GeoTIFF, JPG, JPEG, PNG.
JPG/PNG/JPEG are converted to a real TIFF using Pillow + Rasterio. These ordinary images remain non-georeferenced unless spatial metadata exists in the source. GeoTIFF CRS/transform metadata is preserved.

Normalization
-------------
X_norm = (X - X_min) / (X_max - X_min)

The output is Float32 TIFF with valid values scaled to 0-1. NoData is excluded from statistics and normalization.

Run locally
-----------
pip install -r requirements.txt
python app.py

Deployment
----------
Gunicorn command: gunicorn app:app
