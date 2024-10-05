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

from gprof_nn.data.training_data import GPROFNN1DDataset, GPROFNN3DDataset


def run_tests(
        model: nn.Module,
        test_dataset: DataLoader,
        scalar_metrics: Dict[str, List[ScalarMetric]],
        surface_type_metrics: Optional[Dict[str, List[ScalarMetric]]] = None,
        device: str = "cuda",
        dtype: str = "float32"
) -> xr.Dataset:
    """
    Evaluate retrieval module on test set.

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
    model = model.to(device=device, dtype=dtype)

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
            for metric in mtrcs:
                metric = metric.to(device=device)
                metric.update(pred_k.expected_value(), y[key])

            mtrcs = surface_type_metrics.get(key, [])
            for metric in mtrcs:
                metric = metric.to(device=device)
                metric.update(
                    pred_k.expected_value(),
                    y[key],
                    conditional={"surface_type": y["surface_type"]}
                )

    retrieval_results = {}
    for name, mtrcs in scalar_metrics.items():
        for metric in mtrcs:
            res_name = name + "_" + metric.name.lower()
            retrieval_results[res_name] = metric.compute().cpu().numpy()
    for name, mtrcs in surface_type_metrics.items():
        for metric in mtrcs:
            res_name = name + "_" + metric.name.lower() + "_surface_type"
            retrieval_results[res_name] = (("surface_type",), metric.compute().cpu().numpy())
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
@click.option("-v", "--verbose", count=True)
def cli(
        kind: str,
        model: Path,
        test_data_path: str,
        output_filename: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        batch_size: int = 32,
        verbose: int = 0,
) -> int:
    """
    Process input files.
    """
    model = load_model(model).eval()

    targets = [name for name in model.to_config_dict()["output"].keys()]
    if kind == "1d":
        test_dataset = GPROFNN1DDataset(
            test_data_path,
            targets = targets + ["surface_type"]
        )
    else:
        test_dataset = GPROFNN3DDataset(
            test_data_path,
            augment=False,
            validation=True,
            targets = targets + ["surface_type"]
        )

    data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    scalar_metrics = {
        name: [
            metrics.Bias(),
            metrics.MSE(),
            metrics.CorrelationCoef()
        ] for name in model.to_config_dict()["output"].keys()
    }
    cond = {"surface_type": (0.5, 18.5, 19)}
    surface_type_metrics = {
        name: [
            metrics.Bias(conditional=cond),
            metrics.MSE(conditional=cond),
            metrics.CorrelationCoef(conditional=cond)
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
