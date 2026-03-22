"""
gprof_nn.satrain
================

Provides an interface to run GPROF-NN retrieval on SatRain input data.
"""
import numpy as np
import torch
import xarray as xr

from pytorch_retrieve.inference import SequentialInferenceRunner

from .retrieval import get_model
from . import sensors
from .data.training_data import load_ancillary_data


class GPROFNNRetrieval:
    """
    Interface class for running GPROF-NN retrievals on the SatRain dataset.
    """

    def __init__(
            self,
            sensor: str = "gmi",
            device: str = "cuda:0",
            dtype: str = "bfloat16",
            ancillary_config: str = "CLI"
    ):
        sensor = getattr(sensors, sensor.upper())
        self.model = get_model(sensor).eval()
        self.sensor = sensor
        self.ancillary_config = ancillary_config

        profiles = [
            "snow_water_content",
            "rain_water_content",
            "cloud_water_content",
            "latent_heat"
        ]
        for prof in profiles:
            if prof in self.model.heads:
                self.model.heads.pop(prof)


    def __call__(self, retrieval_input: xr.Dataset) -> xr.Dataset:
        """
        Run GPROF-NN retrieval on input.

        Args:
            retrieval_input: xarray.Dataset containing the retrieval input as provided by the SatRain
                evaluator.

        Return:
            A new xarray.Dataset containing the retrieval results.
        """
        tbs = torch.tensor(retrieval_input[f"obs_{self.sensor.name.lower()}"].data.astype(np.float32))[None]
        eia = torch.tensor(retrieval_input[f"eia_{self.sensor.name.lower()}"].data.astype(np.float32))[None]
        shape = tbs.shape[-2:]
        tbs_full = torch.nan * torch.zeros((1, 15,) + shape)
        eia_full = torch.nan * torch.zeros((1, 15,) + shape)

        chan_inds = list(self.sensor.gprof_channels.keys())
        tbs_full[:, chan_inds] = tbs
        eia_full[:, chan_inds] = eia

        anc = torch.nan * torch.zeros((1, 14,) + shape)

        inpt = {
            "brightness_temperatures": tbs_full,
            "earth_incidence_angles": eia_full,
            "ancillary_data": anc,
        }

        runner = SequentialInferenceRunner(
            self.model,
            [inpt],
            self.model.inference_config,
        )
        results = runner.run(output_path=None)[0]
        results = results[["surface_precip", "probability_of_precipitation"]]
        results = results.rename({
            "probability_of_precipitation": "probability_of_precip"
        })
        return results.rename_dims(y="pixel", x="scan")
