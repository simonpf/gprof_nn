"""
gprof_nn.testing
================

Provides testing functionality for GPROF-NN retrievals.
"""
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

from gprof_nn.data.training_data import (
    GPROFNN1DDataset,
    GPROFNN3DDataset,
    GPROFNNLightDataset
)


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
                metric.update(
                    pred_k.to(device="cpu"),
                    ref.to(device="cpu"),
                    conditional={"surface_type": cond.to(device="cpu")}
                )

    retrieval_results = {}
    for name, mtrcs in scalar_metrics.items():
        for metric in mtrcs:
            res_name = name + "_" + metric.name.lower()
            retrieval_results[res_name] = metric.compute().cpu().numpy()
    for name, mtrcs in surface_type_metrics.items():
        for metric in mtrcs:
            res_name = name + "_" + metric.name.lower() + "_surface_type"
            extra_dims = getattr(metric, "dims", ())
            retrieval_results[res_name] = (("surface_type",) + extra_dims, metric.compute().cpu().numpy())
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
    elif kind == "light":
        test_dataset = GPROFNNLightDataset(
            cloudsat_path=test_data_path / "cs",
            training_paths=test_data_path / "sim",
            augment=False,
            validation=False,
            targets = targets + ["surface_type"],
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
            metrics.MSE(),
            metrics.CorrelationCoef()
        ] for name in model.to_config_dict()["output"].keys()
    }
    cond = {"surface_type": (0.5, 18.5, 19)}
    surface_type_metrics = {
        name: [
            metrics.RelativeBias(conditional=cond),
            metrics.MAE(conditional=cond),
            metrics.MSE(conditional=cond),
            metrics.CorrelationCoef(conditional=cond),
            metrics.ScatterPlot(bins=np.logspace(-3, 2, 41), conditional=cond)
        ] for name in model.to_config_dict()["output"].keys()
    }

    device = torch.device(device)
    dtype = getattr(torch, dtype)

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
