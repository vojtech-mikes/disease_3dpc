import os
from pathlib import Path
from typing import Literal
import plyfile
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


def create_feature_vectors(data_path: Path, typ: Literal["ill", "ok"]) -> pd.DataFrame:
    plant_paths = os.path.join(data_path, typ)

    plant_files = [
        os.path.join(plant_paths, item)
        for item in os.listdir(plant_paths)
        if os.path.isfile(os.path.join(plant_paths, item))
    ]

    assert len(plant_files) > 0, "Plant directory is empty"

    # Load PLY files as PlyData instance
    plant_clouds = []

    for scan in plant_files:
        plant_clouds.append(plyfile.PlyData.read(scan))

    assert len(plant_clouds) > 0, "Loaded zero scans"

    # Load values for statistics
    plants_nir = []

    for plant in plant_clouds:
        plants_nir.append(plant.elements[0].data["scalar_nir"])

    assert len(plants_nir) > 0, "Failed to read scalar_nir values"

    # Extract features from the NIR data and create feature vector
    feature_vectors = []
    for nir in plants_nir:
        feature_vectors.append(
            np.array(
                [
                    np.mean(nir),
                    np.median(nir),
                    min(nir),
                    max(nir),
                    np.std(nir),
                    np.percentile(nir, 25),
                    np.percentile(nir, 50),
                    np.percentile(nir, 75),
                    skew(nir),
                    kurtosis(nir),
                ]
            )
        )

    labels = [
        "mean",
        "median",
        "min",
        "max",
        "std",
        "percentile_25",
        "percentile_50",
        "percentile_75",
        "skewness",
        "kurtosis",
    ]

    final_dataframe = pd.DataFrame(feature_vectors)

    final_dataframe.columns = labels
    final_dataframe.index = [f"{typ}_{Path(x).stem}" for x in plant_files]

    return final_dataframe
