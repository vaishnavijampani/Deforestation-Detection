import ee
import numpy as np
import torch
from model import UNet
from config import *
import rasterio
from attrs import Registry

ee.Initialize()

def load_model():
    model = UNet(in_ch=8, out_ch=1)
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    model.eval()
    return model

def get_composite(roi, start, end):
    return composite_period(roi, start, end)  # re-use from data_download

def run_on_tile(model, before_arr, after_arr):
    x = np.concatenate([before_arr, after_arr], axis=-1)
    x = torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0).float()
    with torch.no_grad():
        pred = model(x).squeeze(0).squeeze(0).numpy()
    return pred

def export_prediction(roi_geojson, save_path="deforestation_map.tif"):
    with open(roi_geojson) as f:
        fc = ee.FeatureCollection(json.load(f))
    before = composite_period(fc, TIME_BEFORE, f"{int(TIME_BEFORE[:4])+1}")
    after = composite_period(fc, TIME_AFTER, f"{int(TIME_AFTER[:4])+1}")
    # Pull full arrays as numpy (careful with huge ROIs)
    arr_before = geemap.ee_to_numpy(before, region=fc.geometry(), bands=NDVI_INPUT_BANDS)
    arr_after = geemap.ee_to_numpy(after, region=fc.geometry(), bands=NDVI_INPUT_BANDS)
    pred = run_on_tile(load_model(), arr_before, arr_after) * 255
    with rasterio.open(
        save_path,
        'w',
        driver='GTiff',
        height=pred.shape[0],
        width=pred.shape[1],
        count=1,
        dtype='uint8',
        crs="EPSG:4326",
        transform=geemap.ee_proj_to_affine(before.projection())
    ) as dst:
        dst.write(pred.astype(np.uint8), 1)
    print("Saved to", save_path)

if __name__ == "__main__":
    export_prediction(ROI_GEOJSON)
