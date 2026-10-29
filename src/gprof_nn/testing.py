"""
gprof_nn.testing
================

Provides testing functionality for GPROF-NN retrievals.
"""
from copy import deepcopy
import logging
from pathlib import Path
from typing import Dict, List, Optional

import click
import numpy as np
from pytorch_retrieve.inference import to_rec
from pytorch_retrieve import metrics
from pytorch_retrieve.metrics import ScalarMetric
from pytorch_retrieve.architectures import load_model
from pytorch_retrieve.tensors import MaskedTensor
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import xarray as xr

from gprof_nn.data.preprocessor import calculate_frozen_fraction
from gprof_nn.data.training_data import (
    GPROFNN1DDataset,
    GPROFNN3DDataset,
    GPROFNNLightDataset
)


LOGGER = logging.getLogger(__name__)


_BINS = {
    "cloud_water_content": np.logspace(-3, np.log10(0.5), 101),
    "cloud_water_path": np.logspace(-3, np.log10(20), 101),
    "convective_precip": np.logspace(-2, 2, 101),
    "ice_water_path":  np.logspace(-3, 2, 101),
    "latent_heat": np.linspace(-10, 10, 101),
    "rain_water_content": np.logspace(-3, 1, 101),
    "rain_water_path": np.logspace(-3, 1, 101),
    "snow_water_content": np.logspace(-3, 1, 101),
    "surface_precip": np.logspace(-2, 2, 101),
}


def run_tests(
        model: nn.Module,
        test_dataset: DataLoader,
        scalar_metrics: Dict[str, List[ScalarMetric]],
        surface_type_metrics: Optional[Dict[str, List[ScalarMetric]]] = None,
        device: str = "cuda",
        dtype: str = "float32"
) -> xr.Dataset:
    """
    Evaluate retrieval model on test set.

    Args:
        model: A trained retrieval model.
        test_dataset: A dataset providing access to the test data.
        scalar_metrics: A dictionary mapping target names to corresponding
             metrics to evaluate.
        surface_type_metrics: Dictionary of metrics conditioned on surface type.
        device: The device on which to perform the evaluation.
        dtype: The dtype to use.

    Return:
        A the xarray.Dataset containing the calculated error metrics.
    """
    model = model.to(device=device, dtype=dtype).eval()

    for x, y in tqdm(test_dataset):

        x = to_rec(x, device=device, dtype=dtype)
        y = to_rec(y, device=device, dtype=dtype)

        for key, target in y.items():
            mask = torch.isnan(target)
            if mask.any():
                y[key] = MaskedTensor(target, mask=mask)

        with torch.no_grad():
            pred = model(x)

        for key, pred_k in pred.items():
            mtrcs = scalar_metrics.get(key, [])

            pred_k = pred_k.expected_value()
            ref = y[key]
            cond = y["surface_type"]

            for metric in mtrcs:
                metric = metric.to(device=device)
                metric.update(pred_k, ref)

            mtrcs = surface_type_metrics.get(key, [])
            for metric in mtrcs:
                metric = metric.to(device="cpu")
                cond_b = torch.clone(cond)
                if ref.shape[1] == 28:
                    cond_b = torch.broadcast_to(cond_b, ref.shape)
                metric.update(
                    pred_k.to(device="cpu"),
                    ref.to(device="cpu"),
                    conditional={"surface_type": cond_b.to(device="cpu")}
                )

    retrieval_results = {}
    for name, mtrcs in scalar_metrics.items():
        for metric in mtrcs:
            res_name = name + "_" + metric.name.lower()
            res = metric.compute().cpu().numpy()
            if 1 < res.ndim:
                dims = metric.dims
                dims = [dim + f"_{name}" for dim in dims]
                retrieval_results[res_name] = (dims, res)
            else:
                retrieval_results[res_name] = res

    for name, mtrcs in surface_type_metrics.items():
        for metric in mtrcs:
            res_name = name + "_" + metric.name.lower() + "_surface_type"
            res = metric.compute().cpu().numpy()
            if 2 < res.ndim:
                dims = ("surface_type", f"predicted_{name}", f"reference_{name}", )
            else:
                dims = ("surface_type",)
            retrieval_results[res_name] = (dims, res)

    if len(retrieval_results) > 0:
        retrieval_results = xr.Dataset(retrieval_results)
    else:
        retrieval_results = None

    return retrieval_results


def run_tests_light(
        model: nn.Module,
        test_dataset: DataLoader,
        scalar_metrics: Dict[str, List[ScalarMetric]],
        scalar_metrics_rain: Dict[str, List[ScalarMetric]],
        scalar_metrics_snow: Dict[str, List[ScalarMetric]],
        device: str = "cuda",
        dtype: str = "float32"
) -> xr.Dataset:
    """
    Evaluate retrieval model on light precipitation test data.

    Args:
        model: A trained retrieval model.
        test_dataset: A dataset providing access to the test data.
        scalar_metrics: A dictionary mapping target names to corresponding
             metrics to evaluate.
        scalar_metrics_rain: A dictionary mapping target names to corresponding
             metrics to evaluate for liquid precipitation.
        scalar_metrics_snow: A dictionary mapping target names to corresponding
             metrics to evaluate for frozen precipitation.
        tile_size: A tile size to use for the evaluation.
        device: The device on which to perform the evaluation.
        dtype: The dtype to use.

    Return:
        A the xarray.Dataset containing the calculated error metrics.
    """
    model = model.to(device=device, dtype=dtype).eval()

    for x, y in tqdm(test_dataset):

        x = to_rec(x, device=device, dtype=dtype)
        y = to_rec(y, device=device, dtype=dtype)

        for key, target in y.items():
            mask = torch.isnan(target)
            if mask.any():
                y[key] = MaskedTensor(target, mask=mask)

        with torch.no_grad():
            pred = model(x)

        sp = pred["surface_precip"].mean()

        twb = y["wet_bulb_temperature"].cpu().float().numpy()
        land_frac = y["land_fraction"].cpu().float().numpy()
        ice_frac = y["ice_fraction"].cpu().float().numpy()
        surface_precip_snow = y["surface_precip_snow"].cpu().float().numpy()
        ffrac = calculate_frozen_fraction(twb, land_frac)

        valid_obs = torch.isfinite(x["brightness_temperatures"]).any(1).cpu().numpy()[:, None]

        sp_pred = pred["surface_precip"].expected_value().cpu().float().numpy()
        sp_ref = y["surface_precip"].cpu().float().numpy()
        snow = (0.5 < ffrac) * (1e-2 < sp_ref) * (1e-2 < sp_pred)
        rain = ffrac < 0.01
        ocean = (land_frac == 0.0) * (ice_frac == 0.0)

        try:
            valid = np.isfinite(sp_ref) * ocean * valid_obs
            for metric in scalar_metrics["surface_precip"]:
                metric.update(torch.tensor(sp_pred[valid]), torch.tensor(sp_ref[valid]))

            valid = np.isfinite(sp_ref) * ocean * rain * valid_obs
            for metric in scalar_metrics_rain["surface_precip"]:
                metric.update(torch.tensor(sp_pred[valid]), torch.tensor(sp_ref[valid]))

            valid = np.isfinite(sp_ref) * ocean * snow * valid_obs
            for metric in scalar_metrics_snow["surface_precip"]:
                metric.update(torch.tensor(sp_pred[valid]), torch.tensor(sp_ref[valid]))

            sp_ref = y["light_precip"].cpu().float().numpy()
            sp_pred = pred["light_precip"].expected_value().cpu().float().numpy()
            snow = (0.5 < ffrac) * (1e-2 < sp_ref) * (1e-2 < sp_pred)
            valid = np.isfinite(sp_ref) * ocean * valid_obs * (sp_ref < 30.0)

            for metric in scalar_metrics["light_precip"]:
                metric.update(torch.tensor(sp_pred[valid]), torch.tensor(sp_ref[valid]))

            valid = np.isfinite(sp_ref) * ocean * rain * valid_obs * (sp_ref < 30.0)
            for metric in scalar_metrics_rain["light_precip"]:
                metric.update(torch.tensor(sp_pred[valid]), torch.tensor(sp_ref[valid]))

            valid = np.isfinite(sp_ref) * ocean * snow * valid_obs * (sp_ref < 30.0)
            for metric in scalar_metrics_snow["light_precip"]:
                if valid.any():
                    metric.update(torch.tensor(sp_pred[valid]), torch.tensor(sp_ref[valid]))
        except Exception as exc:
            LOGGER.exception("Ecountered an error during testing.")
            pass

    retrieval_results = {}
    for name, mtrcs in scalar_metrics.items():
        for metric in mtrcs:
            res_name = name + "_" + metric.name.lower()
            res = metric.compute().cpu().numpy()
            if 1 < res.ndim:
                dims = metric.dims
                dims = [dim + f"_{name}" for dim in dims]
                retrieval_results[res_name] = (dims, res)
            else:
                retrieval_results[res_name] = res

    for name, mtrcs in scalar_metrics_rain.items():
        for metric in mtrcs:
            res_name = name + "_rain_" + metric.name.lower()
            res = metric.compute().cpu().numpy()
            if 1 < res.ndim:
                dims = metric.dims
                dims = [dim + f"_{name}" for dim in dims]
                retrieval_results[res_name] = (dims, res)
            else:
                retrieval_results[res_name] = res

    for name, mtrcs in scalar_metrics_snow.items():
        for metric in mtrcs:
            res_name = name + "_snow_" + metric.name.lower()
            res = metric.compute().cpu().numpy()
            if 1 < res.ndim:
                dims = metric.dims
                dims = [dim + f"_{name}" for dim in dims]
                retrieval_results[res_name] = (dims, res)
            else:
                retrieval_results[res_name] = res

    if len(retrieval_results) > 0:
        retrieval_results = xr.Dataset(retrieval_results)
    else:
        retrieval_results = None

    return retrieval_results


@click.argument("kind")
@click.argument("model")
@click.argument("test_data_path")
@click.argument("output_filename")
@click.option("--device", type=str, default="cuda")
@click.option("--dtype", type=str, default="bfloat16")
@click.option("--batch_size", type=int, default=32)
@click.option("--subsample", type=int, default=1)
@click.option("--use_combined", is_flag=True)
@click.option("-v", "--verbose", count=True)
def cli(
        kind: str,
        model: Path,
        test_data_path: str,
        output_filename: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        batch_size: int = 32,
        subsample: int = 1,
        verbose: int = 0,
        use_combined: bool = False
) -> int:
    """
    Calculate test data accuracy for a given GPROF-NN MODEL using the test data located in TEST_DATA_PATH.
    """
    model = load_model(model).eval()

    test_data_path = Path(test_data_path)

    targets = [name for name in model.to_config_dict()["output"].keys()]
    kind = kind.lower()

    device = torch.device(device)
    dtype = getattr(torch, dtype)

    # Special Handling of Light Precipitation
    if kind == "light":
        test_dataset = GPROFNNLightDataset(
            cloudsat_path=test_data_path / "cs_smooth",
            training_paths=test_data_path / "sim",
            augment=False,
            validation=True,
            targets = (
                targets +
                ["surface_type", "wet_bulb_temperature"] +
                [ "land_fraction", "ice_fraction", "surface_precip_snow"] +
                ["rain_water_content"]
            ),
            subsample=subsample,
        )
        data_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        scalar_metrics = {
            name: [
                metrics.RelativeBias(),
                metrics.MAE(),
                metrics.SMAPE(),
                metrics.MSE(),
                metrics.CorrelationCoef(),
                metrics.ScatterPlot(bins=np.logspace(-2, 2, 41)),
                metrics.Histogram(bins=np.logspace(-2, 2, 41))
            ] for name in model.to_config_dict()["output"].keys()
        }
        scalar_metrics_rain = deepcopy(scalar_metrics)
        scalar_metrics_snow = deepcopy(scalar_metrics)

        retrieval_results = run_tests_light(
            model,
            data_loader,
            scalar_metrics=scalar_metrics,
            scalar_metrics_rain=scalar_metrics_rain,
            scalar_metrics_snow=scalar_metrics_snow,
            device=device,
            dtype=dtype,
        )
        if retrieval_results is not None:
            retrieval_results.to_netcdf(output_filename)
        return 0

    if kind == "1d":
        test_dataset = GPROFNN1DDataset(
            test_data_path,
            targets = targets + ["surface_type"]
        )
        batch_size = None
    elif kind == "3d":
        test_dataset = GPROFNN3DDataset(
            test_data_path,
            augment=True,
            validation=False,
            targets = targets + ["surface_type"],
            subsample=subsample,
            use_combined=use_combined
        )

    data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    scalar_metrics = {
        name: [
            metrics.RelativeBias(),
            metrics.MAE(),
            metrics.SMAPE(),
            metrics.MSE(),
            metrics.CorrelationCoef(),
            metrics.ScatterPlot(bins=_BINS.get(name, np.logspace(-2, 2, 101)))
        ] for name in model.to_config_dict()["output"].keys()
    }

    cond = {"surface_type": (0.5, 18.5, 19)}
    surface_type_metrics = {
        name: [
            metrics.RelativeBias(conditional=cond),
            metrics.MAE(conditional=cond),
            metrics.SMAPE(conditional=cond),
            metrics.MSE(conditional=cond),
            metrics.CorrelationCoef(conditional=cond),
            metrics.ScatterPlot(
                bins=_BINS.get(name, np.logspace(-2, 2, 101)),
                conditional=cond
            )
        ] for name in model.to_config_dict()["output"].keys()
    }

    retrieval_results = run_tests(
        model,
        data_loader,
        scalar_metrics=scalar_metrics,
        surface_type_metrics=surface_type_metrics,
        device=device,
        dtype=dtype,
    )
    if retrieval_results is not None:
        retrieval_results.to_netcdf(output_filename)
