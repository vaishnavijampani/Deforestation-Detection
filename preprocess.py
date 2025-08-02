import os
import numpy as np
import tensorflow as tf
from glob import glob
from config import *

def parse_tfrecord(example_proto):
    feature_description = {
        "glc_before": tf.io.FixedLenFeature((PATCH_SIZE_PX*PATCH_SIZE_PX*4,), tf.float32),
        "glc_after":  tf.io.FixedLenFeature((PATCH_SIZE_PX*PATCH_SIZE_PX*4,), tf.float32),
        "deforested": tf.io.FixedLenFeature((PATCH_SIZE_PX*PATCH_SIZE_PX,), tf.int64)
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    bb = tf.reshape(parsed["glc_before"], (PATCH_SIZE_PX, PATCH_SIZE_PX, 4))
    aa = tf.reshape(parsed["glc_after"], (PATCH_SIZE_PX, PATCH_SIZE_PX, 4))
    mask = tf.reshape(parsed["deforested"], (PATCH_SIZE_PX, PATCH_SIZE_PX))
    return np.concatenate([bb, aa], axis=-1), mask.astype(np.uint8)

def main():
    os.makedirs(PATCH_NPY_DIR, exist_ok=True)
    files = glob(os.path.join(EXPORT_FOLDER, "*.tfrecord"))
    for tf_file in files:
        ds = tf.data.TFRecordDataset(tf_file)
        for i, ex in enumerate(ds):
            x, y = parse_tfrecord(ex)
            # optionally filter out all-zero or mostly-zero mask patches
            if y.sum() < 100: continue
            fname = os.path.splitext(os.path.basename(tf_file))[0]
            np.savez_compressed(os.path.join(PATCH_NPY_DIR, f"{fname}_{i}.npz"), x=x, y=y)

if __name__ == "__main__":
    main()
