"""
===========================
gprof_nn.data.training_data
===========================

This module defines the dataset classes that provide access to
the training data for the GPROF-NN retrievals.
"""
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import cache, partial
import io
import itertools
import math
import logging
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
from pansat.utils import resample_data
from pyresample import SwathDefinition
from rich.progress import track
from scipy.ndimage import rotate
from scipy.signal import convolve
import torch
from torch.utils.data import Dataset, IterableDataset
import torchvision.transforms.functional
from tqdm import tqdm
import xarray as xr

from gprof_nn import sensors
from gprof_nn.utils import (
    calculate_interpolation_weights,
    calculate_interpolation_indices,
    interpolate
)
from gprof_nn.data.utils import (
    apply_limits,
    compressed_pixel_range,
    load_variable,
    decompress_scene,
    remap_scene,
    upsample_scans
)
from gprof_nn.utils import expand_tbs
from gprof_nn.geometry import (
    calculate_footprints_conical,
    calculate_footprints_xtrack,
    viewing_to_incidence
)
from gprof_nn.definitions import (
    ANCILLARY_VARIABLES,
    ANCILLARY_CFGS,
    MASKED_OUTPUT,
    LAT_BINS,
    TIME_BINS,
    LIMITS,
    ALL_TARGETS,
    PROFILE_TARGETS
)
from gprof_nn.data.preprocessor import PreprocessorFile
from gprof_nn.data.utils import merge_precipitation

LOGGER = logging.getLogger(__name__)


def get_central_latitude(path: Path) -> float:
    """
    Calculate central latitude of a training scene.

    Args:
        path: A path object pointint to a GPROF-NN 3D training file in NetCDF4 format.

    Return:
        The central latitude.
    """
    with xr.open_dataset(path) as data:
        lats = data.latitude.data
        mask = -100 < lats
        lats = lats[mask]
        center = lats.mean()
    return center


@cache
def sample_centers(
        paths: Tuple[Path],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Array contaning the mean latitude coordinates of all training samples.

    Args:
        paths: A tuple containing the paths pointing to the training data directories to consider.

    Return:
        A tuple containing the identified filenames and corresponding resampling weights to achieve
        uniform latitude coverage.
    """
    files = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise RuntimeError(
                "The provided path %s does not exists.",
                path
            )
        files += sorted(list(path.glob("**/3d*/*_*_*.nc")))

    files = files

    cached = Path("sample_weights.npz")
    if cached.exists():
        cached = np.load(cached, allow_pickle=True)
        files_cached = cached["files"]
        centers_cached = cached["centers"]
        if (files_cached.size == len(files)) and (files_cached == np.array(files)).all():
            return files_cached, centers_cached

    pool = ProcessPoolExecutor(max_workers=4)
    centers = []
    tasks = []
    for path in files:
        tasks.append(pool.submit(get_central_latitude, path))

    for task in tqdm(tasks, desc="Calculating sample coordinates"):
        centers.append(task.result())

    np.savez("sample_weights.npz", files=files, centers=centers)

    return files, np.array(centers)


def apply_augmentations_3d(
        tbs: np.ndarray,
        eia: np.ndarray,
        anc: np.ndarray,
        sensor: sensors.Sensor,
        rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies data augmentation mimicking common observation corruptions.

    Args:
        tbs: A 3D numpy.ndarray containing the full brightness temperatures loaded for the sensor.
        eia: A 3D numpy.ndarray containing the corresponding earth-incidence angles.
        anc: A 3D numpy.ndarray contraining the ancillary data.
        sensor: The senor object representing the sensor.
        rng: A random generator.

    Return:
        A tuple ``(tbs, eia)`` containing the modified brightness temperatures (tbs) and earth-incidence
        angles (eia).
    """
    # Drop channels
    if sensor.channel_drop is not None:
        p = sensor.channel_drop
        n_chans = tbs.shape[0]
        for chan in range(n_chans):
            if rng.random() <= p:
                tbs[chan] = torch.nan
                eia[chan] = torch.nan

    # Drop scanlines
    if sensor.scanline_drop is not None:
        p = sensor.scanline_drop
        chans = torch.where(torch.isfinite(tbs).any(1).any(1))
        if rng.random() <= p:

            n_lines = rng.integers(1, 30)
            start = rng.integers(0, tbs.shape[1] - n_lines)
            end = start + n_lines

            for chan_ind in chans:
                tbs[chan_ind, start:end] = torch.nan
                eia[chan_ind, start:end] = torch.nan

            if rng.random() <= 0.5:
                anc[:, start:end] = torch.nan

    # Erroneous scan lines
    if sensor.scanline_drop is not None:
        p = sensor.scanline_drop
        chans = torch.where(torch.isfinite(tbs).any(1).any(1))
        val = 330.0 + 20.0 * rng.random()
        if rng.random() <= p:

            n_lines = rng.integers(1, 10)
            start = rng.integers(0, tbs.shape[1] - n_lines)
            end = start + n_lines

            for chan_ind in chans:
                tbs[chan_ind, start:end] = val

            if rng.random() <= 0.5:
                anc[:, start:end] = torch.nan

    return tbs, eia


def calculate_resampling_indices(latitudes, time, sensor):
    """
    Calculate scene indices based on latitude and local times.

    Args:
        latitudes: Central latitudes of the scenes.
        local_time: Time of day in minuts for each sample.
        sensor: The sensor object to whose latitude and local
            time sampling to to resample the scenes.

    Return:
        None if the provided sensor has no latitude ratios
        attribute. Otherwise an array of scene indices that
        resamples the scenes to match the latitude distribution
        of the sensor.
    """
    latitude_ratios = getattr(sensor, "latitude_ratios", None)
    if latitude_ratios is None:
        return None

    lat_indices = np.digitize(latitudes, LAT_BINS[1:-1])
    time_indices = np.digitize(time, TIME_BINS[1:-1])

    if latitude_ratios.ndim == 1:
        weights = latitude_ratios[lat_indices]
    else:
        weights = latitude_ratios[lat_indices, time_indices]
    weights = np.nan_to_num(weights, 0.0)
    indices = np.arange(latitudes.size)
    probs = weights / weights.sum()
    return np.random.choice(indices, size=latitudes.size, p=probs)


def decompress_and_load(filename):
    """
    Load a potentially gzipped NetCDF file and return the
    data as 'xarray.Dataset'.

    Args:
        filename: The filename to store the file to.

    Return:
        An 'xarray.Dataset' containing the loaded data.
    """
    LOGGER.debug("Decompressing %s.", filename)
    filename = Path(filename)
    if not filename.exists():
        if Path(filename).suffix == ".gz":
            raise ValueError(f"The file '{filename}' doesn't exist. ")
        elif Path(filename).suffix == ".lz4":
            raise ValueError(f"The file '{filename}' doesn't exist. ")
        else:
            filename_gz = Path(str(filename) + ".gz")
            if not filename_gz.exists():
                filename_lz4 = Path(str(filename) + ".lz4")
                if not filename_lz4.exists():
                    raise ValueError(
                        f"Neither the file '{filename}' nor '{filename}.gz' exist."
                    )
                filename = filename_lz4
            else:
                filename = filename_gz

    if Path(filename).suffix == ".gz":
        decompressed = io.BytesIO()
        args = ["gunzip", "-c", str(filename)]
        with subprocess.Popen(args, stdout=subprocess.PIPE) as proc:
            decompressed.write(proc.stdout.read())
        decompressed.seek(0)
        data = xr.load_dataset(decompressed, engine="h5netcdf")
    elif Path(filename).suffix == ".lz4":
        with TemporaryDirectory() as tmp:
            tmpfile = Path(tmp) / filename.stem
            with open(tmpfile, "wb") as decompressed:
                subprocess.run(
                    ["unlz4", "-c", str(filename)], stdout=decompressed, check=True
                )
            data = xr.load_dataset(tmpfile)
            Path(tmpfile).unlink()
    else:
        data = xr.open_dataset(filename)
    return data


def load_tbs_1d_gmi(
        training_data: xr.Dataset,
) -> torch.Tensor:
    """
    Load brightness temperatures for GMI training data.

    The training data for GMI contains the actual L1C observations and
    thus doesn't need any additional modifications.

    Args:
        training_data: The xarray.Dataset containing the training data.

    Return:
        A torch tensor containing the loaded brightness temperatures.
    """
    tbs = training_data["brightness_temperatures"].data
    return torch.tensor(tbs)


def load_tbs_1d_xtrack_sim(
        training_data: xr.Dataset,
        angles: np.ndarray,
        sensor: sensors.Sensor,
        targets: List[str],
        satformer: bool = False
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Load brightness temperatures for cross-track scanning sensors from simulator
    collocations.

    Args:
        training_data: An xarray.Dataset containing training data extracted from
            GPROF simulator files.
        angles: A np.ndarray cotaining the viewing angle of the tbs to load.
        sensor: The sensor from which the TBs are loaded
        targets: A list of the targets to load along with the brightness temperatures.
        satformer: Set to True to load Satformer Tbs instead of the legacy GPROF V7
            simulator.

    Return:
        A tuple containing the loaded brightness temperatures and a dictionary
        containing the loaded targets.
    """
    samples = np.arange(training_data.samples.size)
    samples = xr.DataArray(samples, dims="samples")
    angles = xr.DataArray(np.abs(angles), dims="samples")

    if not satformer:
        training_data = training_data[
            ["simulated_brightness_temperatures", "brightness_temperature_biases"] +
            targets
        ]
        training_data = training_data.interp(
            samples=samples,
            angles=angles,
            method="nearest"
        )
        tbs = training_data.simulated_brightness_temperatures.data
        tbs_full = np.nan * np.zeros((tbs.shape[0], 15), dtype=np.float32)
        tbs_full[:, sensor.gprof_channel_indices] = tbs
        biases = training_data.brightness_temperature_biases.data
        biases_full = np.nan * np.zeros((tbs.shape[0], 15), dtype=np.float32)
        biases_full[:, sensor.gprof_channel_indices] = biases
        biases = (
            biases_full /
            (
                np.cos(np.deg2rad(EIA_GMI)) *
                np.cos(np.deg2rad(angles.data[..., None]))
            )
        )
        tbs_full = tbs_full - biases
    else:
        training_data = training_data[
            ["satformer_tbs", "satformer_tbs_rand"] +
            targets
        ]
        tbs = training_data["satformer_tbs_rand"]
        tbs_rand = training_data["satformer_tbs_rand"]
        noise = tbs_rand - tbs

        noise = noise.interp(
            samples=samples,
            angles=angles,
            method="nearest"
        )
        tbs = tbs.interp(
            samples=samples,
            angles=angles,
            method="nearest"
        )

        tbs = tbs.data #+ noise.data
        tbs_full = np.nan * np.zeros((tbs.shape[0], 15), dtype=np.float32)
        tbs_full[:, sensor.gprof_channel_indices] = tbs

    #ang_ind = np.argmin(training_data.angles.data)
    #targets = load_targets_1d(training_data[{"angles": ang_ind}], targets)
    targets = load_targets_1d(training_data, targets)

    return torch.tensor((tbs_full).astype(np.float32)), targets


def load_tbs_1d_conical_sim(
        training_data: xr.Dataset,
        sensor: sensors.Sensor,
        satformer: bool = True
) -> torch.Tensor:
    """
    Load brightness temperatures for cross-track scanning sensors from simulator
    collocations.

    Args:
        training_data: An xarray.Dataset containing training data extracted from
            GPROF simulator files.
        sensor: The sensor from which the TBs are loaded
        satformer: Whether or not to load the Satformer Tbs

    Return:
        A torch tensor containing the loaded brightness temperatures.

    """
    sensor_inds = list(sensor.gprof_channels.keys())
    gmi_chans = list(sensors.GMI.gprof_channels.keys())
    gmi_inds = []
    for ind in sensor_inds:
        if ind in gmi_chans:
            gmi_inds.append(gmi_chans.index(ind))
        else:
            gmi_inds.append(gmi_inds[-1])
            
    if satformer:
        training_data = training_data[["satformer_tbs_rand"]]
        tbs = training_data.satformer_tbs_rand
    else:
        training_data = training_data[["simulated_brightness_temperatures", "brightness_temperature_biases",]]
        tbs = training_data.simulated_brightness_temperatures.data
        biases = training_data.brightness_temperature_biases.data
        gmi_angs = sensors.GMI.earth_incidence_angle[gmi_inds]
        angs = sensor.earth_incidence_angle
        corr = np.cos(np.deg2rad(gmi_angs)) / np.cos(np.deg2rad(angs)) * biases
        tbs = tbs - biases

    tbs_full = np.nan * np.zeros((tbs.shape[0], 15), dtype=np.float32)
    tbs_full[:, sensor_inds] = tbs
    return torch.tensor(tbs_full)


def load_tbs_1d_xtrack_other(
        training_data: xr.Dataset,
        sensor: sensors.Sensor
) -> torch.Tensor:
    """
    Load brightness temperatures for cross-track scanning sensors from collocations
    with real observations, i.e., MRMS or ERA5 collocations.

    Args:
        training_data: An xarray.Dataset containing training data extracted from
            GPROF simulator files.
        sensor: The sensor from which the TBs are loaded

    Return:
        A tuple ``(tbs, angs)`` containing the brightness temperatures ``tbs``
        and corresponding earth incidence angles ``angs``.
    """
    tbs = training_data["brightness_temperatures"].data
    tbs_full = np.nan * np.zeros((tbs.shape[0], 15), dtype=np.float32)
    tbs_full[:, sensor.gprof_channel_indices] = tbs
    angles = training_data["earth_incidence_angle"].data
    angles_full = np.nan * np.zeros_like(tbs_full)
    angles_full[:, sensor.gprof_channel_indices] = angles[..., None]

    tbs = torch.tensor(tbs_full.astype("float32"))
    angles = torch.tensor(angles_full.astype("float32"))
    return tbs, angles


def load_tbs_1d_conical_other(
        training_data: xr.Dataset,
        sensor: sensors.Sensor
) -> torch.Tensor:
    """
    Load brightness temperatures for non-GMI conical scanner from collocations
    with real observations, i.e., MRMS or ERA5 collocations.

    Args:
        training_data: An xarray.Dataset containing training data extracted from
            GPROF simulator files.
        sensor: The sensor from which the TBs are loaded

    Return:
        A tuple ``(tbs, angs)`` containing the brightness temperatures ``tbs``
        and corresponding earth incidence angles ``angs``.
    """
    tbs = training_data["brightness_temperatures"].data
    tbs_full = np.nan * np.ones(tbs.shape[:-1] + (15,), dtype="float32")
    tbs_full[:, sensor.gprof_channel_indices] = tbs
    angles = training_data["earth_incidence_angle"].data
    angles_full = np.nan * np.ones(tbs.shape[:-1] + (15,), dtype="float32")
    angles_full[:, sensor.gprof_channel_indices] = angles
    tbs = torch.tensor(tbs_full.astype("float32"))
    angles = torch.tensor(angles_full.astype("float32"))
    return tbs, angles


def determine_ancillary_config(input_data: xr.Dataset) -> bool:
    """
    Determine configuration of ancillary data available from preprocessor.
    """
    t2m = input_data.two_meter_temperature.data
    if (t2m < 0).all():
        return "NRT"
    snow_depth = input_data.snow_depth.data
    if (snow_depth < 0).all():
        return "STD"
    return "CLI"


def load_ancillary_data(
        training_data: xr.Dataset,
        configuration: str,
        stack_dim: 0
) -> torch.Tensor:
    """
    Load ancillary data from training sample.

    Args:
        training_data: The xarray.Dataset containing the training data.
        configuration: The name of the ancillary data configuration.
        stack_dim: Along which dimension to stack the ancillary variables.

    Return:
        A torch tensor containign the ancillary data stacked along the
        'stack_dim'.
    """
    LOGGER.debug("Loading ancillary data for configuration '%s'.", configuration)
    data = []
    if configuration not in ANCILLARY_CFGS:
        LOGGER.warning(
            "Ancillary-data configuration '%s' is not known. Known configurations are "
            "'%s'.",
            configuration,
            list(ANCILLARY_CFGS.keys())
        )
    var_inds = ANCILLARY_CFGS.get(configuration, [])
    for ind, var in enumerate(ANCILLARY_VARIABLES):
        if var in training_data:
            data_v = training_data[var].data.copy().astype(np.float32)
        else:
            data_v = training_data["two_meter_temperature"].data.copy().astype(np.float32)

        data_v[data_v < -9000] = np.nan
        if ind in var_inds:
            data.append(data_v)
        else:
            data.append(np.nan * data_v)
    data = np.stack(data, axis=stack_dim)
    return torch.tensor(data.astype(np.float32))


def load_targets_1d(
        training_data: xr.Dataset,
        targets: List[str]
) -> Dict[str, torch.Tensor]:
    """
    Load retrieval target tensors from training data file.

    Args:
        training_data: The xarray.Dataset containing the training data.
        targets: List of the targets to load.
    """
    targs = {}
    for var in targets:

        if var in training_data:
            data_t = training_data[var].data
        else:
            n_samples = training_data.samples.size
            if var in PROFILE_TARGETS:
                shape = (n_samples, 28)
            else:
                shape = (n_samples)
            data_t = np.zeros(shape, dtype=np.float32)

        if var in PROFILE_TARGETS:
            if data_t.ndim == 2:
                data_t = data_t[:, None]
        else:
            if data_t.ndim == 1:
                data_t = data_t[..., None]
        data_t[data_t < -900] = np.nan
        targs[var] = torch.tensor(data_t.astype("float32").squeeze())
    return targs


def load_targets_1d_xtrack(
        training_data: xr.Dataset,
        angles: np.ndarray,
        targets: List[str]
) -> Dict[str, torch.Tensor]:
    """
    Load retrieval target tensors from training data file for x-track scanners.
    Since the 'surface_precip' and 'convective_precip' variables are

    Args:
        training_data: The xarray.Dataset containing the training data.
        targets: List of the targets to load.
    """
    samples = np.arange(training_data.samples.size)
    samples = xr.DataArray(samples, dims="samples")
    angles = xr.DataArray(np.abs(angles), dims="samples")

    ang_ind = np.argmin(training_data.angles.data)
    training_data = training_data[targets][{"angles": ang_ind}]

    targs = {}
    for var in targets:
        data_t = training_data[var].data
        if data_t.ndim == 1:
            data_t = data_t[..., None]
        data_t[data_t < -900] = np.nan
        targs[var] = torch.tensor(data_t.astype("float32"))
    return targs


class GPROFNN1DDataset(IterableDataset):
    """
    Dataset class for loading the training data for GPROF-NN 1D retrieval.
    """
    merge = 8
    resample_datasets = 1

    def __init__(
        self,
        paths: Union[Path, List[Path]],
        targets: Optional[List[str]] = None,
        augment: bool = True,
        validation: bool = False,
        satformer: bool = False,
        batch_size: int = 2048,
        ancillary_cfg: Optional[str] = None,
        sensor: Optional[str] = None
    ):
        """
        Create GPROF-NN 1D dataset.

        The GPROF-NN 1D data is split up into separate files by orbit. This
        dataset loads the training data from all available files. And provides
        an iterable over the samples in the dataset.

        Args:
            paths: A single path or a list of paths containing the training files.
            targets: A list of the target variables to load.
            augment: Whether or not to apply data augmentation to the loaded
                data.
            validation: If set to 'True', data  loaded in consecutive iterations
                over the dataset will be identical.
            satformer: Set to 'True' to load Tbs from Satformer.
            batch_size: The size of batches to load.
            ancillary_cfg: The ancillary data configuration to load.
            sensor: Optional name of the sensor to load the training data for.
        """
        super().__init__()

        if targets is None:
            targets = ALL_TARGETS

        self.targets = targets
        self.validation = validation
        self.satformer = satformer
        self.augment = augment
        self.ancillary_cfg = ancillary_cfg

        if not isinstance(paths, list):
            paths = [Path(paths)]
        else:
            paths = [Path(path) for path in paths]

        self.files = []
        for path in paths:
            if not path.exists():
                raise RuntimeError(
                    "The provided path does not exist."
                )

            files = sorted(list(path.glob("**/1d/*_*_*.nc")))
            if len(files) == 0:
                raise RuntimeError(
                    "Could not find any GPROF-NN 1D training data files "
                    f"in {path}."
                )
            self.files += [str(path) for path in files]

        self.init_rng()
        self.files = self.rng.permutation(self.files)
        self.batch_size = batch_size
        self.training_data = None
        self.indices = None
        self.pool = ThreadPoolExecutor(max_workers=1)

        if sensor is not None:
            sensor = sensors.get_sensor(sensor)
        self.sensor = sensor


    def init_rng(self, w_id=0):
        """
        Initialize random number generator.

        Args:
            w_id: The worker ID which of the worker process..
        """
        if self.validation:
            seed = 42
        else:
            seed = int.from_bytes(os.urandom(4), "big") + w_id

        self.rng = np.random.default_rng(seed)
        self.n_workers = 1

    def worker_init_fn(self, w_id: int) -> None:
        """
        Initializes the worker state for parallel data loading.

        Args:
            w_id: The ID of the worker.
        """
        self.init_rng(w_id)
        winfo = torch.utils.data.get_worker_info()
        n_workers = winfo.num_workers
        self.n_workers = n_workers
        self.files = self.files[w_id::n_workers]
        self.pool = ThreadPoolExecutor(max_workers=1)

    def load_training_data(self, dataset: xr.Dataset) -> Dict[str, torch.Tensor]:

        if self.sensor is None:
            sensor = sensors.get_sensor(dataset.attrs["sensor"])
        else:
            sensor = self.sensor
        targets = self.targets
        ref_target = targets[0]

        if self.ancillary_cfg is not None:
            cfg = self.ancillary_cfg
        elif self.validation:
            cfg = "CLI"
        else:
            cfg = self.rng.choice(["NONE", "NRT", "STD", "CLI"])

        if sensor == sensors.GMI:
            tbs = dataset["brightness_temperatures"].data
            y_t = dataset[ref_target].data
            valid_input = np.any(tbs > 0, -1)
            valid_target = np.isfinite(y_t).any(tuple(range(1, y_t.ndim)))
            mask = valid_input * valid_target
            dataset = dataset[{"samples": mask}]

            tbs = load_tbs_1d_gmi(dataset)

            anc = load_ancillary_data(dataset, configuration=cfg, stack_dim=1)

            targets = load_targets_1d(dataset, self.targets)
            angs = torch.tensor(np.broadcast_to(EIA_GMI.astype("float32"), tbs.shape))

        elif isinstance(sensor, sensors.CrossTrackScanner):

            dataset = dataset.compute()

            if dataset.attrs["source"] == "sim":

                dataset = dataset.sortby("angles").compute()
                dataset.simulated_brightness_temperatures.compute()
                dataset.brightness_temperature_biases.compute()
                tbs = dataset["simulated_brightness_temperatures"].data
                tb_biases = dataset["brightness_temperature_biases"].data
                y_t = dataset[ref_target].data
                if self.satformer:
                    valid_input = np.any(tbs > 0, (-2, -1))
                else:
                    valid_input = np.all(tbs > 0, (-2, -1)) * np.all(np.abs(tb_biases) < 50, -1)
                valid_target = np.isfinite(y_t).any(tuple(range(1, y_t.ndim)))
                mask = valid_input * valid_target

                dataset = dataset[{"samples": mask}].compute()
                angles = dataset["angles"].data
                dataset_l = dataset[{"angles": [0]}].assign_coords(angles=[angles[0] - 3.0])
                dataset_r = dataset[{"angles": [-1]}].assign_coords(angles=[angles[-1] + 3.0])
                dataset_f = xr.concat([dataset_l, dataset, dataset_r], dim="angles")
                angles = dataset_f["angles"].data
                angs = self.rng.uniform(
                    angles.min(),
                    angles.max(),
                    size=dataset_f.samples.size,
                ).astype(np.float32)
                tbs, targets = load_tbs_1d_xtrack_sim(
                    dataset_f,
                    angs,
                    sensor,
                    targets,
                    satformer=self.satformer
                )
                angs = torch.tensor(np.broadcast_to(angs[..., None], tbs.shape))

            else:
                tbs = dataset["brightness_temperatures"].data
                y_t = dataset[ref_target].data
                valid_input = np.any(tbs > 0, -1)
                valid_target = np.isfinite(y_t).any(tuple(range(1, y_t.ndim)))
                mask = valid_input * valid_target
                dataset = dataset[{"samples": mask}]
                tbs, angs = load_tbs_1d_xtrack_other(dataset, sensor)
                targets = load_targets_1d(dataset, self.targets)

            anc = load_ancillary_data(dataset, configuration=cfg, stack_dim=1)

        elif isinstance(sensor, sensors.ConstellationScanner):

            if dataset.source == "sim":
                tbs = load_tbs_1d_conical_sim(dataset, sensor, satformer=self.satformer)
                angs = torch.nan * torch.zeros_like(tbs)
                inds = list(sensor.gprof_channels.keys())
                angs[:, inds] = torch.tensor(
                    sensor.earth_incidence_angle,
                    dtype=torch.float32
                )
            else:
                tbs, angs = load_tbs_1d_conical_other(dataset, sensor)
            anc = load_ancillary_data(dataset, configuration=cfg, stack_dim=1)
            targets = load_targets_1d(dataset, self.targets)


        # Drop channels
        if sensor.channel_drop is not None:
            p = sensor.channel_drop
            n_chans = tbs.shape[1]
            for chan in range(n_chans):
                mask = self.rng.random(size=tbs.shape[0]) <= p
                tbs[mask, chan] = torch.nan

        x = {
            "brightness_temperatures": tbs.to(torch.float32),
            "ancillary_data": anc.to(torch.float32),
            "earth_incidence_angles": angs.to(torch.float32)
        }
        return x, targets

    def load_data(self, files):
        """
        Sets
        """
        all_files = self.rng.permutation(files)
        inputs = {}
        targets = {}

        for ind, path in enumerate(all_files):

            with xr.load_dataset(path, engine="h5netcdf") as input_file:
                try:
                    inputs_f, targets_f = self.load_training_data(input_file)
                except Exception as exc:
                    LOGGER.exception("Failed loading training data from file %s", path)
                    continue
                for name, tensor in inputs_f.items():
                    inputs.setdefault(name, []).append(tensor)
                for name, tensor in targets_f.items():
                    targets.setdefault(name, []).append(tensor)

        input_file.close()
        del input_file

        inputs = {name: torch.cat(data, 0) for name, data in inputs.items()}
        targets = {name: torch.cat(data, 0) for name, data in targets.items()}
        n_samples = inputs["brightness_temperatures"].shape[0]
        inds = torch.tensor(self.rng.permutation(n_samples))
        inputs = {name: data[inds] for name, data in inputs.items()}
        targets = {name: data[inds] for name, data in targets.items()}
        return inputs, targets


    def __iter__(self):

        tasks = []
        all_files = self.rng.permutation(self.files)[::10]
        for ind in range(math.ceil(len(all_files) / self.merge)):
            files = all_files[self.merge * ind : self.merge * (ind + 1)]
            tasks.append(self.pool.submit(self.load_data, files))

        for task in tasks:

            try:
                inputs, targets = task.result()
            except Exception as exc:
                continue

            start_ind = 0
            n_samples = inputs["brightness_temperatures"].shape[0]
            for _ in range(self.resample_datasets):
                perm_inds = self.rng.permutation(n_samples)
                while start_ind < n_samples:
                    inds = np.arange(start_ind, start_ind + self.batch_size) % n_samples
                    inds = perm_inds[inds]
                    batch_inputs = {name: data[inds] for name, data in inputs.items()}
                    batch_targets = {name: data[inds] for name, data in targets.items()}
                    start_ind += self.batch_size
                    yield batch_inputs, batch_targets

            del inputs
            del targets
            del task

    def __repr__(self):
        return f"GPROFNN1DDataset(path={self.paths}, targets={self.targets})"

    #def __len__(self):
    #    """
    #    The number of samples in the training dataset.
    #    """
    #    tot_samples = 0
    #    for path in self.files:
    #        with xr.open_dataset(path) as data:
    #            tot_samples += data.samples.size
    #    return tot_samples // self.batch_size

    #def __getitem__(self, ind):
    #    """
    #    Return batch from training dataset.

    #    Args:
    #        ind: The index of the batch to return.
    #    """
    #    if self.training_data is None:
    #        self.load_data()

    #    if ind < self.n_workers or self.indices is None:
    #        self.indices = np.random.permutation(
    #            self.training_data[0]["brightness_temperatures"].shape[0]
    #        )

    #    n_samples = self.training_data[0]["brightness_temperatures"].shape[0]

    #    inputs, targets = self.training_data
    #    inds = self.indices[ind * self.batch_size: (ind + 1) * self.batch_size]
    #    inputs = {
    #        name: tensor[inds % n_samples] for name, tensor in inputs.items()
    #    }
    #    targets = {
    #        name: tensor[inds % n_samples] for name, tensor in targets.items()
    #    }
    #    return inputs, targets



def load_training_data_3d_gmi(
        scene: xr.Dataset,
        targets: List[str],
        augment: bool = False,
        rng: np.random.Generator = None,
        ancillary_config: Optional[str] = None
) -> Tuple[Dict[str, torch.Tensor]]:
    """
    Load GPROF-NN 3D training scene for GMI.

    Args:
        scene: An xarray.Dataset containing the scene from which to load
            the training data.
        targets: A list containing a list of the targets to load.
        augment: Whether or not to augment the input data.
        rng: A numpy random number generator to use for the augmentation.
        ancillary_config: A string specifying the ancillary data configuration to load.

    Return:
        A tuple ``(x, y)`` of dictionaries ``x`` and ``y`` containing the
        training input data in ``x`` and the training reference data in ``y``.
    """
    source = scene.source
    if source == "cloudsat":
        targets = targets + ["total_precip", "precip_flag", "surface_precip_snow"]

    variables = [
        name for name in targets + ["latitude", "longitude"]
        if name in scene
    ]
    scene = decompress_scene(scene, variables)

    if augment:
        p_x_o = rng.random()
        p_x_i = rng.random()
        p_y = rng.random()
    else:
        p_x_o = 0.5
        p_x_i = 0.5
        p_y = rng.random()

    lats = scene.latitude.data
    lons = scene.longitude.data

    if source == "sim":
        sensor = sensors.GMI

        lons_fp_gmi = scene.longitude.data
        lats_fp_gmi = scene.latitude.data
        remap_coords = calculate_footprints_conical(
            lons_fp_gmi,
            lats_fp_gmi,
            sensor.viewing_geometry.altitude,
            sensor.earth_incidence_angle[0],
            (-0.5 * sensor.viewing_geometry.scan_range, 0.5 * sensor.viewing_geometry.scan_range),
            sensor.viewing_geometry.scan_range / sensor.viewing_geometry.pixels_per_scan,
            64,
            128,
            sensor.viewing_geometry.scan_offset,
            subsample=10,
            rng=rng
        )
        swath = SwathDefinition(remap_coords.longitude.data, remap_coords.latitude.data)
        scene = resample_data(scene, swath, radius_of_influence=15e3, new_dims=("scans", "pixels"))
        scene = scene.transpose("levels", "scans", "pixels", ...)
    else:
        scene = scene.transpose("scans", "pixels", ...)
    tbs = torch.tensor(scene.brightness_temperatures.data, dtype=torch.float32)
    angs = torch.nan * torch.zeros_like(tbs)
    inds = list(sensors.GMI.gprof_channels.keys())
    angs[..., inds] = torch.tensor(
        sensors.GMI.earth_incidence_angle[inds],
        dtype=torch.float32
    )

    if ancillary_config is not None:
        cfg = ancillary_config
    elif augment:
        cfg = rng.choice(["NONE", "NRT", "NRT_SNOW", "STD", "CLI"])
    else:
        cfg = "CLI"
    anc = load_ancillary_data(scene, configuration=cfg, stack_dim=0)

    tbs = torch.permute(tbs, (2, 0, 1))
    angs = torch.permute(angs, (2, 0, 1))

    apply_augmentations_3d(
        tbs,
        angs,
        anc,
        sensors.GMI,
        rng
    )

    if augment:
        # Simulate missing higher frequency channels
        r = rng.random()
        n_p = rng.integers(10, 30)
        if r > 0.6:
            tbs[10:15, :, :n_p] = torch.nan

        r = rng.random()
        if r > 0.8:
            blobs = rng.integers(1, 4)
            for ind in range(blobs):
                y = np.arange(tbs.shape[1])
                x = np.arange(tbs.shape[2])
                xx, yy = np.meshgrid(x, y)
                row_center = rng.integers(tbs.shape[1])
                col_center = rng.integers(tbs.shape[2])
                r = np.sqrt((xx - col_center) ** 2 + (yy - row_center) ** 2)
                noise = 200 * np.exp(np.log(0.5) * (r / 1.5) ** 2)
                chan_var = rng.random(size=15)
                chan_var[9:] = 0.0
                noise = (chan_var[..., None, None] * noise[None]).astype(np.float32)

                tbs = tbs + noise
                tbs[tbs > 350] = np.nan

    x = {
        "brightness_temperatures": tbs,
        "earth_incidence_angles": angs,
        "ancillary_data": anc
    }

    y = {}
    for target in targets:
        # MRMS collocations don't contain all targets.
        if target not in scene:
            if target in PROFILE_TARGETS:
                empty = torch.nan * torch.zeros((28, 128, 64))
            else:
                empty = torch.nan * torch.zeros((1, 128, 64))
            y[target] = empty.squeeze()
            continue

        data_t = scene[target].data
        if data_t.ndim < 3:
            data_t = data_t[None]
        if np.issubdtype(data_t.dtype, np.floating):
            data_t[data_t < -900] = np.nan
        data = torch.tensor(data_t.astype("float32"))

        if target in PROFILE_TARGETS and data.shape[-1] == 28:
            data = torch.permute(data, (2, 0, 1)).clone()

        y[target] = data.squeeze()

    # Also flip data if requested.
    if augment:
        prob = rng.random()
        if prob > 0.5:
            x = {key: torch.flip(tensor, (-2,)) for key, tensor in x.items()}
            y = {key: torch.flip(tensor, (-2,)) for key, tensor in y.items()}
        prob = rng.random()
        if prob > 0.5:
            x = {key: torch.flip(tensor, (-1,)) for key, tensor in x.items()}
            y = {key: torch.flip(tensor, (-1,)) for key, tensor in y.items()}
    y = {key: tensor[None] if tensor.dim() == 2 else tensor for key, tensor in y.items()}

    if source == "cloudsat":

        angle = rng.uniform(-20, 20)
        scale = rng.uniform(1.0, 1.2)
        transform = partial(
            torchvision.transforms.functional.affine,
            angle=angle,
            scale=scale,
            translate=(0.0, 0.0),
            shear=0.0,
            fill=torch.nan
        )
        x = {key: transform(tensor) for key, tensor in x.items()}
        y = {key: tensor[None] if tensor.dim() == 2 else tensor for key, tensor in y.items()}
        y = {key: transform(tensor) for key, tensor in y.items()}
        surface_precip = y.pop("surface_precip")
        total_precip = y.pop("total_precip")
        precip_flag = y.pop("precip_flag")
        precip_snow = y.pop("surface_precip_snow")
        y["surface_precip"] = total_precip

    return x, y


def load_training_data_3d_xtrack_sim(
        sensor: sensors.Sensor,
        scene: xr.Dataset,
        targets: List[str],
        augment: bool = False,
        rng: np.random.Generator = None,
        ancillary_config: Optional[str] = None
) -> Tuple[Dict[str, torch.Tensor]]:
    """
    Load GPROF-NN 3D training scene for cross-track scannres from
    sim-file training data.

    Args:
        sensor: The sensor from which the training data was extracted.
        scene: An xarray.Dataset containing the scene from which to load
            the training data.
        targets: A list containing a list of the targets to load.
        augment: Whether or not to augment the input data.
        rng: A numpy random number generator to use for the augmentation.
        ancillary_config: An optional string specifying the ancillary data configuration
            to load.

    Return:
        A tuple ``(x, y)`` of dictionaries ``x`` and ``y`` containing the
        training input data in ``x`` and the training reference data in ``y``.
    """
    required = [
        "latitude",
        "longitude",
        "satformer_tbs_rand"
    ]
    variables = [
        name for name in targets + required
        if name in scene
    ]
    angle_grid = scene.angles.data
    scene = decompress_scene(scene, variables)

    lons_fp_gmi = scene.longitude.data
    lats_fp_gmi = scene.latitude.data
    va_range = sensor.viewing_geometry.scan_range
    va_max = va_range / 2
    eia_max = viewing_to_incidence(va_max, sensor.viewing_geometry.altitude)
    eia_range = (-eia_max, eia_max)
    remap_coords = calculate_footprints_xtrack(
        lons_fp_gmi,
        lats_fp_gmi,
        sensor.viewing_geometry.altitude,
        eia_range=eia_range,
        vai=2 * va_max / sensor.viewing_geometry.pixels_per_scan,
        n_pixels=64,
        n_scans=128,
        scan_dist=sensor.viewing_geometry.scan_offset,
        subsample=20,
        rng=rng
    )
    swath = SwathDefinition(remap_coords.longitude.data, remap_coords.latitude.data)
    scene = resample_data(scene, swath, radius_of_influence=15e3, new_dims=("scans", "pixels"))
    scene = scene.transpose("levels", "scans", "pixels", ...)

    angs = np.abs(remap_coords.earth_incidence_angle.data)

    weights = calculate_interpolation_weights(angs, angle_grid)
    # Calculate brightness temperatures
    tbs_sim = scene.satformer_tbs_rand.data
    tbs_sim = interpolate(tbs_sim, weights)
    tbs_sim[tbs_sim > 350] = np.nan

    full_shape = tbs_sim.shape[:2] + (15,)
    tbs_full = np.nan * np.ones(full_shape, dtype="float32")
    tbs_full[:, :, sensor.gprof_channel_indices] = tbs_sim
    tbs_full = torch.permute(torch.tensor(tbs_full), (2, 0, 1))
    tbs_full[tbs_full < 20] = torch.nan

    chan_inds = list(sensor.gprof_channels.keys())
    angs_full = np.nan * np.ones(full_shape, dtype="float32")
    angs_full[:, :, chan_inds] = np.abs(angs[..., None])
    angs_full = torch.permute(torch.tensor(angs_full), (2, 0, 1))

    invalid = torch.isnan(tbs_full).all(0)
    angs_full[..., invalid] = torch.nan

    if ancillary_config is not None:
        cfg = ancillary_config
    elif augment:
        cfg = rng.choice(["NONE", "NRT", "NRT_SNOW", "STD", "CLI"])
    else:
        cfg = "CLI"
    anc = load_ancillary_data(scene, configuration=cfg, stack_dim=0)
    anc[..., invalid] = torch.nan

    apply_augmentations_3d(
        tbs_full,
        angs_full,
        anc,
        sensor,
        rng
    )

    x = {
        "brightness_temperatures": tbs_full,
        "earth_incidence_angles": angs_full,
        "ancillary_data": anc
    }

    y = {}
    for target in targets:
        # MRMS collocations don't contain all targets.
        if target not in scene:
            if target in PROFILE_TARGETS:
                empty = torch.nan * torch.zeros((28, 128, 64))
            else:
                empty = torch.nan * torch.zeros((1, 128, 64))
            y[target] = empty.squeeze()
            continue

        dims = ("scans", "pixels")
        if "levels" in scene[target].dims:
            dims = ("levels",) + dims
        if "angles" in scene[target].dims:
            dims = dims + ("angles",)
        data = scene[target].transpose(*dims)

        data = data.data.astype("float32")

        if "angles" in scene[target].dims:
            data = interpolate(data, weights)

        if data.ndim < 3:
            data = data[..., None]

        data[data < -900] = np.nan
        data = torch.tensor(data)
        y[target] = data.squeeze()

    # Also flip data if requested.
    if augment:
        prob = rng.random()
        if prob > 0.5:
            x = {key: torch.flip(tensor, (-2,)) for key, tensor in x.items()}
            y = {key: torch.flip(tensor, (-2,)) for key, tensor in y.items()}
        prob = rng.random()
        if prob > 0.5:
            x = {key: torch.flip(tensor, (-1,)) for key, tensor in x.items()}
            y = {key: torch.flip(tensor, (-1,)) for key, tensor in y.items()}

    return x, y


def load_training_data_3d_conical_sim(
        sensor: sensors.Sensor,
        scene: xr.Dataset,
        targets: List[str],
        augment: bool = False,
        rng: np.random.Generator = None,
        ancillary_config: Optional[str] = None
) -> Tuple[Dict[str, torch.Tensor]]:
    """
    Load GPROF-NN 3D training scene for non-GMI conical scanners from
    sim-file training data.
    Args:
        sensor: The sensor from which the training data was extracted.
        scene: An xarray.Dataset containing the scene from which to load
            the training data.
        targets: A list containing a list of the targets to load.
        augment: Whether or not to augment the input data.
        rng: A numpy random number generator to use for the augmentation.
        ancillary_config: Optional string specifying the ancillary data configuration to load.

    Return:
        A tuple ``(x, y)`` of dictionaries ``x`` and ``y`` containing the
        training input data in ``x`` and the training reference data in ``y``.
    """
    required = [
        "latitude",
        "longitude",
        "satformer_tbs_rand"
    ]
    variables = [
        name for name in targets + required
        if name in scene
    ]
    scene = decompress_scene(scene, variables)

    width = 64
    height = 128

    lons_fp_gmi = scene.longitude.data
    lats_fp_gmi = scene.latitude.data

    remap_coords = calculate_footprints_conical(
        lons_fp_gmi,
        lats_fp_gmi,
        sensor.viewing_geometry.altitude,
        sensor.earth_incidence_angle[0],
        (-0.5 * sensor.viewing_geometry.scan_range, 0.5 * sensor.viewing_geometry.scan_range),
        sensor.viewing_geometry.scan_range / sensor.viewing_geometry.pixels_per_scan,
        64,
        128,
        sensor.viewing_geometry.scan_offset,
        subsample=10,
        rng=rng
    )

    from pansat.utils import resample_data
    from pyresample import SwathDefinition
    swath = SwathDefinition(remap_coords.longitude.data, remap_coords.latitude.data)
    scene = resample_data(scene, swath, radius_of_influence=15e3, new_dims=("scans", "pixels"))
    scene = scene.transpose("levels", "scans", "pixels", ...)

    # Calculate brightness temperatures
    tbs_sim = scene.satformer_tbs_rand.data.astype(np.float32)
    full_shape = tbs_sim.shape[:2] + (15,)
    if tbs_sim.shape[-1] < 15:
        tbs_full = np.nan * np.ones(full_shape, dtype="float32")
        tbs_full[:, :, sensor.gprof_channel_indices] = tbs_sim
    else:
        tbs_full = tbs_sim
    tbs_full = torch.permute(torch.tensor(tbs_full), (2, 0, 1))
    tbs_full[tbs_full > 350] = np.nan

    angs_full = torch.nan * torch.zeros_like(tbs_full)
    angs_full[sensor.gprof_channel_indices] = torch.tensor(
        sensor.earth_incidence_angle[..., None, None],
        dtype=torch.float32
    )

    if ancillary_config is not None:
        cfg = ancillary_config
    elif augment:
        cfg = rng.choice(["NONE", "NRT", "NRT_SNOW", "STD", "CLI"])
    else:
        cfg = "CLI"
    anc = load_ancillary_data(scene, configuration=cfg, stack_dim=0)

    tbs_full, angs_full = apply_augmentations_3d(
        tbs_full,
        angs_full,
        anc,
        sensor,
        rng
    )

    x = {
        "brightness_temperatures": tbs_full,
        "earth_incidence_angles": angs_full,
        "ancillary_data": anc
    }

    y = {}
    scene = scene.transpose("levels", "scans", "pixels", ...)
    for target in targets:
        # MRMS collocations don't contain all targets.
        if target not in scene:
            if target in PROFILE_TARGETS:
                empty = torch.nan * torch.zeros((28, 128, 64))
            else:
                empty = torch.nan * torch.zeros((1, 128, 64))
            y[target] = empty.squeeze()
            continue

        data = scene[target].data
        data[data < -900] = np.nan
        if data.ndim < 3:
            data = data[None]
        data = torch.tensor(scene[target].data.astype("float32"))
        y[target] = data.squeeze()

    # Also flip data if requested.
    if augment:
        prob = rng.random()
        if prob > 0.5:
            x = {key: torch.flip(tensor, (-2,)) for key, tensor in x.items()}
            y = {key: torch.flip(tensor, (-2,)) for key, tensor in y.items()}
        prob = rng.random()
        if prob > 0.5:
            x = {key: torch.flip(tensor, (-1,)) for key, tensor in x.items()}
            y = {key: torch.flip(tensor, (-1,)) for key, tensor in y.items()}

    return x, y


def load_training_data_3d_other(
        sensor: sensors.Sensor,
        scene: xr.Dataset,
        targets: List[str],
        augment: bool = False,
        rng: np.random.Generator = None,
        ancillary_config: Optional[str] = None
) -> Tuple[Dict[str, torch.Tensor]]:
    """
    Load training data for non-GMI sensors that are training scenes extracted
    from actual observations, i.e., not .sim-file derived.

    Args:
        sensor: The sensor object from which the training data was extracted.
        scene: An xarray.Dataset containing the scene from which to load
            the training data.
        targets: A list containing a list of the targets to load.
        augment: Whether or not to augment the input data.
        rng: A numpy random number generator to use for the augmentation.
        ancillary_config: Optional string specifying the ancillary data configuration to load.

    Return:
        A tuple ``(x, y)`` of dictionaries ``x`` and ``y`` containing the
        training input data in ``x`` and the training reference data in ``y``.
    """
    required = [
        "latitude",
        "longitude",
    ]
    variables = [
        name for name in targets + required
        if name in scene
    ]

    width = 64
    height = 128

    if augment:
        pix_start = rng.integers(0, scene.pixels.size - width + 1)
        scn_start = rng.integers(0, scene.scans.size - height + 1)
    else:
        pix_start = (scene.pixels.size - width) // 2
        scn_start = (scene.scans.size - height) // 2
    pix_end = pix_start + width
    scn_end = scn_start + height
    scene = scene[{"pixels": slice(pix_start, pix_end), "scans": slice(scn_start, scn_end)}]

    # Calculate brightness temperatures
    tbs = scene.brightness_temperatures.data.copy()
    full_shape = tbs.shape[:2] + (15,)
    if tbs.shape != full_shape:
        tbs_full = np.nan * np.ones(full_shape, dtype="float32")

        scene_sensor = sensors.get_sensor(scene.attrs["sensor"])
        if scene_sensor != sensor:
            inds_in = scene_sensor.gprof_channel_indices
            inds_out = [np.searchsorted(inds_in, ind) for ind in sensor.gprof_channel_indices]
            tbs_full[:, :, sensor.gprof_channel_indices] = tbs[..., inds_out]
        else:
            tbs_full[:, :, sensor.gprof_channel_indices] = tbs
    else:
        tbs_full = tbs.astype(np.float32)
        for ch_ind in range(15):
            if ch_ind not in sensor.gprof_channel_indices:
                tbs_full[..., ch_ind] = np.nan

    tbs_full = torch.permute(torch.tensor(tbs_full), (2, 0, 1))

    angs_full = torch.nan * torch.zeros_like(tbs_full)
    eia = scene.earth_incidence_angle.data.copy()
    eia[eia < -99] = np.nan

    chan_inds = list(sensor.gprof_channels.keys())
    if eia.ndim == 2:
        angs_full[chan_inds] = torch.tensor(eia[None], dtype=torch.float32)
    else:
        angs_full[chan_inds] = torch.tensor(
            sensor.earth_incidence_angle[..., None, None],
            dtype=torch.float32
        )

    if ancillary_config is not None:
        cfg = ancillary_config
    elif augment:
        cfg = rng.choice(["NONE", "NRT", "NRT_SNOW", "STD", "CLI"])
    else:
        cfg = "CLI"
    anc = load_ancillary_data(scene, configuration=cfg, stack_dim=0)

    apply_augmentations_3d(
        tbs_full,
        angs_full,
        anc,
        sensor,
        rng
    )

    x = {
        "brightness_temperatures": tbs_full,
        "earth_incidence_angles": angs_full,
        "ancillary_data": anc
    }

    dims = ("scans", "pixels")
    if "levels" in scene.dims:
        dims = ("levels",) + dims
    scene = scene.transpose(*dims, ...)

    y = {}
    for target in targets:
        # MRMS collocations don't contain all targets.
        if target not in scene:
            if target in PROFILE_TARGETS:
                empty = torch.nan * torch.zeros((28, 128, 64))
            else:
                empty = torch.nan * torch.zeros((1, 128, 64))
            y[target] = empty.squeeze()
            continue


        data = scene[target].data.astype("float32")
        data[data < -900] = np.nan
        if data.ndim < 3:
            data = data[None]

        data = torch.tensor(data)
        dims = tuple(range(data.ndim))
        y[target] = data.squeeze()

    # Also flip data if requested.
    if augment:
        prob = rng.random()
        if prob > 0.5:
            x = {key: torch.flip(tensor, (-2,)) for key, tensor in x.items()}
            y = {key: torch.flip(tensor, (-2,)) for key, tensor in y.items()}
        prob = rng.random()
        if prob > 0.5:
            x = {key: torch.flip(tensor, (-1,)) for key, tensor in x.items()}
            y = {key: torch.flip(tensor, (-1,)) for key, tensor in y.items()}

    return x, y


class GPROFNN3DDataset(Dataset):
    """
    Dataset class for loading the training data for GPROF-NN 3D retrieval.
    """

    def __init__(
        self,
        path: Union[Path, List[Path]],
        targets: Optional[List[str]] = None,
        augment: bool = True,
        validation: bool = False,
        subsample: int = 1,
        resample_latitudes: bool = False,
        sensor: Optional[str] = None,
        use_combined: bool = False
    ):
        """
        Create GPROF-NN 3D dataset.

        The training data for the GPROF-NN 3D retrieval consists of 2D scenes
        in separate files.

        Args:
            path: The path or a list of paths containing the training data files.
            targets: A list of the target variables to load.
            augment: Whether or not to apply data augmentation to the loaded
                data.
            validation: If set to 'True', data  loaded in consecutive iterations
                over the dataset will be identical.
            subsample: A subsampling factor used to randomly subsample the training
                data.
            resample_latitude: Set to 'True' to resample training samples to achieve even latitude
                coverage.
            sensor: Optional sensor name in order to force loading of a specific channel configuration.
            use_combined: Set to 'True' to load CMB precip instead of combined MiRS-CMB precip.
        """
        super().__init__()

        if targets is None:
            targets = ALL_TARGETS
        self.targets = targets
        self.validation = validation
        self.augment = augment and not validation
        self.validation = validation
        self.subsample = subsample

        if sensor is not None:
            sensor = sensors.get_sensor(sensor)
        self.sensor = sensor

        if isinstance(path, list):
            paths = path
        else:
            paths = [path]
        self.path = paths

        files = []
        for path in paths:
            path = Path(path)
            if not path.exists():
                raise RuntimeError(
                    "The provided path %s does not exists.",
                    path
                )
            files += sorted(list(path.glob("**/3d*/*_*_*.nc")))

        folders = set([path.parent for path in files])

        if len(files) == 0:
            raise RuntimeError(
                "Could not find any GPROF-NN 3D training data files "
                f"in {self.path}."
            )
        self.files = files

        if resample_latitudes:

            files, centers = sample_centers(tuple(self.path))
            bins = np.linspace(-90, 90, 91)
            cts = np.histogram(centers, bins=bins)[0]
            k = np.ones(10)
            cts_s = convolve(cts, k, mode="same")
            lat_centers = 0.5 * (bins[1:] + bins[:-1])
            cts_s = cts_s / np.cos(np.deg2rad(lat_centers))
            sampling_weights = 1.0 / cts_s
            sampling_weights = np.minimum(10.0, sampling_weights / np.nanmin(sampling_weights))
            inds = np.digitize(centers, bins) - 1
            weights = sampling_weights[inds]
            weights = weights / weights.sum()
            self.files = np.random.choice(files, size=len(files), p=weights)

        self.init_rng()
        self.files = self.rng.permutation(self.files)

        self.use_combined = use_combined


    def init_rng(self, w_id=0):
        """
        Initialize random number generator.

        Args:
            w_id: The worker ID which of the worker process..
        """
        if self.validation:
            seed = 42
        else:
            seed = int.from_bytes(os.urandom(4), "big") + w_id
        self.rng = np.random.default_rng(seed)

    def worker_init_fn(self, w_id: int):
        """
        Pytorch retrieve interface.
        """
        self.init_rng(w_id)
        winfo = torch.utils.data.get_worker_info()
        n_workers = winfo.num_workers

    def __repr__(self):
        return f"GPROFNN3DDataset(path={self.path}, targets={self.targets})"

    def __len__(self):
        return len(self.files) // self.subsample

    def __getitem__(self, ind):

        ind_min = self.subsample * ind
        ind_max = ind_min + self.subsample
        ind = min(self.rng.integers(ind_min, ind_max), len(self.files) - 1)

        try:
            with xr.open_dataset(self.files[ind]) as scene:

                if self.sensor is None:
                    sensor = scene.attrs["sensor"]
                    sensor = getattr(sensors, sensor)
                else:
                    sensor = self.sensor

                if sensor == sensors.GMI:
                    if self.use_combined and "surface_precip_combined" in scene:
                        ocean_mask = (scene["land_fraction"].data < 10) * (scene["ice_fraction"].data < 10)
                        sp_cmb = scene["surface_precip_combined"].data
                        sp_light = scene["light_precip"].data
                        sp_merged = merge_precipitation(sp_cmb, sp_light)
                        scene["surface_precip"].data[ocean_mask] = sp_merged[ocean_mask]

                    targets = self.targets
                    x, y = load_training_data_3d_gmi(
                        scene,
                        targets=targets,
                        augment=self.augment,
                        rng=self.rng
                    )
                elif isinstance(sensor, sensors.CrossTrackScanner):
                    if scene.source == "sim":
                        if self.use_combined and "surface_precip_combined" in scene:
                            ocean_mask = (scene["land_fraction"].data < 10) * (scene["ice_fraction"].data < 10)
                            sp_cmb = scene["surface_precip_combined"].data
                            sp_light = scene["light_precip"].data
                            sp_merged = merge_precipitation(sp_cmb, sp_light[..., None])
                            scene["surface_precip"].data[ocean_mask] = sp_merged[ocean_mask]
                        x, y = load_training_data_3d_xtrack_sim(
                            sensor,
                            scene,
                            targets=self.targets,
                            augment=self.augment,
                            rng=self.rng
                        )
                    else:
                        x, y = load_training_data_3d_other(
                            sensor,
                            scene,
                            targets=self.targets,
                            augment=self.augment,
                            rng=self.rng
                        )
                elif isinstance(sensor, sensors.ConstellationScanner):
                    if scene.source == "sim":
                        if self.use_combined and "surface_precip_combined" in scene:
                            ocean_mask = (scene["land_fraction"].data < 10) * (scene["ice_fraction"].data < 10)
                            sp_cmb = scene["surface_precip_combined"].data
                            sp_light = scene["light_precip"].data
                            sp_merged = merge_precipitation(sp_cmb, sp_light)
                            scene["surface_precip"].data[ocean_mask] = sp_merged[ocean_mask]
                        x, y = load_training_data_3d_conical_sim(
                            sensor,
                            scene,
                            targets=self.targets,
                            augment=self.augment,
                            rng=self.rng
                        )
                    else:
                        x, y = load_training_data_3d_other(
                            sensor,
                            scene,
                            targets=self.targets,
                            augment=self.augment,
                            rng=self.rng
                        )
        except Exception as exc:
            LOGGER.warning(
                "Encountered an error when trying to load data from file '%s'.",
                self.files[ind]
            )
            new_ind = self.rng.integers(0, len(self))
            return self[new_ind]

        sp = y["surface_precip"]
        if torch.isfinite(sp).sum() < 5:
            new_ind = self.rng.integers(0, len(self))
            LOGGER.warning(
                "Less than 10 valid pixels in file %s. Falling back to another "
                " randomly-chosen sample.",
                self.files[ind]
            )
            return self[new_ind]

        return x, y


class GPROFNNLightDataset(Dataset):
    """
    Dataset for loading light-precipitation training data.
    """
    def __init__(
        self,
        cloudsat_path: Path,
        training_paths: Union[Path, List[Path]],
        targets: Optional[List[str]] = None,
        augment: bool = True,
        validation: bool = False,
        subsample: int = 1,
        resample_latitudes: bool = False,
        sensor: Optional[str] = None
    ):
        """
        Create GPROF-NN 3D dataset.

        The training data for the GPROF-NN 3D retrieval consists of 2D scenes
        in separate files.

        Args:
            cloudsat_path: The path containing the CloudSat collocations to use for training.
            training_paths: The paths containing the other files to use for training.
            targets: A list of the target variables to load.
            augment: Whether or not to apply data augmentation to the loaded
                data.
            validation: If set to 'True', data  loaded in consecutive iterations
                over the dataset will be identical.
            subsample: A subsampling factor used to randomly subsample the training
                data.
            resample_latitude: Set to 'True' to resample training samples to achieve even latitude
                coverage.
            sensor: Optional sensor name in order to force loading of a specific channel configuration.
        """
        super().__init__()

        if targets is None:
            targets = ALL_TARGETS
        self.targets = targets
        self.validation = validation
        self.augment = augment and not validation
        self.validation = validation
        self.subsample = subsample

        if sensor is not None:
            sensor = sensors.get_sensor(sensor)
        self.sensor = sensor

        cloudsat_path = Path(cloudsat_path)
        self.cloudsat_files = sorted(list(cloudsat_path.glob("**/3d*/*_*_*.nc")))

        other_files = []
        if not isinstance(training_paths, list):
            training_paths = [Path(training_paths)]
        for path in training_paths:
            path = Path(path)
            if not path.exists():
                raise RuntimeError(
                    "The provided path %s does not exists.",
                    path
                )
            other_files += sorted(list(path.glob("**/3d*/*_*_*.nc")))
        self.other_files = other_files

        self.init_rng()

    def init_rng(self, w_id=0):
        """
        Initialize random number generator.

        Args:
            w_id: The worker ID which of the worker process..
        """
        if self.validation:
            seed = 42
        else:
            seed = int.from_bytes(os.urandom(4), "big") + w_id
        self.rng = np.random.default_rng(seed)

    def worker_init_fn(self, w_id: int):
        """
        Pytorch retrieve interface.
        """
        self.init_rng(w_id)
        winfo = torch.utils.data.get_worker_info()
        n_workers = winfo.num_workers

    def __repr__(self):
        return f"GPROFNNLightDataset(path={self.path}, targets={self.targets})"

    def __len__(self):
        return len(self.cloudsat_files) * 2

    def __getitem__(self, ind):

        rem = ind % 2
        ind = ind // 2

        if rem == 0:
            subsample = self.subsample
            ind_min = subsample * ind
            ind_max = ind_min + subsample
            ind = min(self.rng.integers(ind_min, ind_max), len(self.cloudsat_files) - 1)
            sample_file = self.cloudsat_files[ind]
        else:
            subsample = int(
                self.subsample * len(self.other_files) / len(self.cloudsat_files)
            )
            ind_min = subsample * ind
            ind_max = ind_min + subsample
            ind = min(self.rng.integers(ind_min, ind_max), len(self.other_files) - 1)
            sample_file = self.other_files[ind]

        try:
            targs = self.targets
            with xr.open_dataset(sample_file) as scene:
                source = scene.source
                x, y = load_training_data_3d_gmi(
                    scene,
                    targets=targs + ["surface_precip_combined"],
                    augment=self.augment,
                    rng=self.rng
                )
        except Exception as exc:
            raise exc
            LOGGER.warning(
                "Encountered an error when trying to load data from file '%s'.",
                sample_file
            )
            new_ind = self.rng.integers(0, len(self))
            return self[new_ind]

        if rem == 0:
            sp = y.pop("surface_precip")
            y["surface_precip"] = torch.nan * torch.zeros_like(sp)
            y["light_precip"] = sp

            y.pop("surface_precip_combined")
        else:
            sp = y.pop("surface_precip_combined")
            if source != "mrms":
                y["surface_precip"] = sp
            else:
                sp = y["surface_precip"]
            y["light_precip"] = np.nan * torch.zeros_like(sp)

        if torch.isfinite(sp).sum() < 10:
            new_ind = self.rng.integers(0, len(self))
            LOGGER.warning(
                "Less than 10 valid pixels in file %s. Falling back to another "
                " randomly-chosen sample.",
                sample_file,
            )
            return self[new_ind]

        return x, y


class GPROFNNSimInputLoader(GPROFNN3DDataset):
    """
    Input loader for running GPROF-NN simulator models on GPROF-NN training
    files.
    """
    def __getitem__(self, ind) -> Tuple[Dict[str, torch.Tensor], Path]:
        """
        Return input data and name of input file.
        """
        with xr.open_dataset(self.files[ind]) as scene:

            tbs = torch.tensor(scene.brightness_temperatures.data.transpose((2, 0, 1)))
            angs = torch.tensor(np.broadcast_to(EIA_GMI.astype("float32")[0][..., None, None], tbs.shape))
            anc = torch.tensor(np.stack(
                [scene[anc_var].data.astype("float32") for anc_var in ANCILLARY_VARIABLES]
            ))

        inpt_data = {
            "brightness_temperatures": tbs[None],
            "earth_incidence_angles": angs[None],
            "ancillary_data": anc[None]
        }
        return inpt_data, self.files[ind]

    def finalize_results(self, results, input_file) -> Tuple[xr.Dataset, str]:
        """
        Save simulator results to training file.

        Args:
            results: A dictionary containing the results from the simulator.
            input_file: A path object pointing to the file the input data
                was loaded from.
        """
        output_data = xr.load_dataset(input_file)
        tbs_sim = results["simulated_brightness_temperatures"].cpu().numpy()[0]
        output_data["simulated_brightness_temperatures"] = (
            ("scans", "pixels", "channels"), tbs_sim.transpose((1, 2, 0))
        )
        output_data.simulated_brightness_temperatures.encoding = {
            "dtype": "uint16",
            "scale_factor": 0.01,
            "add_offset": 1,
            "_FillValue":  2 ** 16 - 1,
            "zlib": True
        }
        tb_biases = results["brightness_temperature_biases"].cpu().numpy()[0]
        output_data["brightness_temperature_biases"] = (
            (("scans", "pixels", "channels"), tb_biases.transpose(1, 2, 0))
        )
        output_data.brightness_temperature_biases.encoding = {
            "dtype": "int16",
            "scale_factor": 0.01,
            "add_offset": 0,
            "_FillValue":  2 ** 15 - 1,
            "zlib": True
        }
        return output_data, input_file.name


def transform_observations_satformer(
        observations: np.ndarray,
        meta: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split satformer observations into mean, std. dev., and scaled anomaly.

    Args:
        observations: A 2D np.ndarray containing the observations.
        meta: A 3D np.ndarray containing the meta data corresponding to the observations.
    """
    obs = observations
    valid = np.isfinite(obs)
    obs = observations[None]
    meta[..., ~valid] = np.nan
    return obs, meta


def load_ancillary_data_satformer(
        scene: xr.Dataset,
        rng: np.random.Generator,
        drop: float = 0.5,
        anc_vars: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Load ancillary variables for Satformer model.

    Args:
        scene: A xarray.Dataset containing the training data.
        rng: A numpy.random.Generator object for applying random ancillary variable dropout.
        drop: The probability with which to drop ancillary data.
        anc_vars: An optional list of ancillary variables to load.
    """
    inpt = {}

    shape = (scene.scans.size, scene.pixels.size)

    if anc_vars is None:
        all_vars = ANCILLARY_VARIABLES
    for anc_var in anc_vars:

        if anc_var in scene:
            anc_data = torch.tensor(scene[anc_var].data).to(dtype=torch.float32)
        else:
            anc_data = torch.nan * torch.zeros(shape)

        if self.rng.random() <= drop:
            anc_data = np.nan * torch.tensor(data[anc_var].data).to(dtype=torch.float32)
        if anc_data.ndim < 3:
            anc_data = anc_data[None]
        if anc_data.ndim < 4:
            anc_data = anc_data[:, None]

        n_chans, n_seq, n_y, n_x = anc_data.shape
        anc_mask = torch.isnan(anc_data).all()[None]
        inpt[anc_var] = anc_data
        inpt[anc_var + "_mask"] = anc_mask

    return inpt


class SatformerDataset:
    """
    Dataset for training a Satformer to produce simulated brightness temperatures.
    """
    def __init__(
            self,
            path: Path,
            seq_len_in: int = 13,
            seq_len_out: int = 6,
            validation: bool = False,
            channel_dropout: float = 0.1,
            sampling_rate: float = 1.0
    ):
        """
        Args:
            path: Path pointing to the training data folder.
            seq_len_in: The number of input observations.
            seq_len_out: The number of output observations.
            validation: If 'True' data loading will be deterministic.
            channel_dropout: The number of input observations to drop and reproduce
        """
        self.input_files = np.array(sorted(list(Path(path).glob("*.nc"))))
        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out
        self.validation = validation
        self.channel_dropout = channel_dropout
        self.sampling_rate = sampling_rate
        self.init_rng()

    def init_rng(self, w_id=0):
        """
        Initialize random number generator.

        Args:
            w_id: The worker ID which of the worker process..
        """
        if self.validation:
            seed = 42
        else:
            seed = int.from_bytes(os.urandom(4), "big") + w_id
        self.rng = np.random.default_rng(seed)

    def worker_init_fn(self, w_id: int):
        """
        Pytorch retrieve interface.
        """
        self.init_rng(w_id)
        winfo = torch.utils.data.get_worker_info()
        n_workers = winfo.num_workers


    def __len__(self) -> int:
        return int(self.sampling_rate * len(self.input_files))

    def __getitem__(self, ind: int) -> Dict[str, torch.Tensor]:

        lower = math.trunc(ind / self.sampling_rate)
        upper = min(math.trunc((ind  + 1) / self.sampling_rate), len(self.input_files) - 1)
        if upper > lower:
            ind = self.rng.integers(lower, upper)
        else:
            ind = lower

        try:
            data = xr.open_dataset(self.input_files[ind])
            n_chans_in = data.input_channels.size
            n_chans_out = data.target_channels.size
            chans_in = self.rng.permutation(n_chans_in)
            chans_out = self.rng.permutation(n_chans_out)
            valid = np.isfinite(data.lon.data)
        except Exception:
            return self[self.rng.integers(0, len(self))]

        input_observations = data.input_observations.data.astype("float32")
        input_meta = data.input_meta_data.data.astype("float32")
        dropped_observations = data.target_observations.data.astype("float32")
        dropped_meta = data.target_meta_data.data.astype("float32")
        target_observations = data.target_observations.data.astype("float32")
        target_meta = data.target_meta_data.data.astype("float32")

        obs_in = []
        meta_in = []
        obs_out = []
        meta_out = []
        obs_dropped = []
        meta_dropped = []

        for input_ind in range(self.seq_len_in):
            if input_ind < len(chans_in):
                obs = input_observations[chans_in[input_ind]]
                meta = input_meta[chans_in[input_ind]]

                rand = self.rng.random()
                if (rand >= self.channel_dropout):
                    obs, meta = transform_observations_satformer(obs, meta)
                    obs_in.append(torch.tensor(obs.astype(np.float32)))
                    meta_in.append(torch.tensor(meta.astype(np.float32)))
                else:
                    obs_in.append(torch.nan * torch.zeros((1, 128, 128)))
                    meta_in.append(torch.nan * torch.zeros((8, 128, 128)))
                    obs_out.append(torch.tensor(obs.astype(np.float32))[None])
                    meta_out.append(torch.tensor(meta.astype(np.float32)))
            else:
                obs_in.append(torch.nan * torch.zeros((1, 128, 128)))
                meta_in.append(torch.nan * torch.zeros((8, 128, 128)))

        for output_ind in range(self.seq_len_out - len(obs_out)):
            if output_ind < len(chans_out):
                obs_out.append(torch.tensor(target_observations[[chans_out[output_ind]]]))
                meta_out.append(torch.tensor(target_meta[chans_out[output_ind]]))
            else:
                obs_out.append(torch.nan * torch.zeros_like(obs_out[-1]))
                meta_out.append(torch.nan * torch.zeros_like(meta_out[-1]))

        inpt = {
            "observations": torch.stack(obs_in, 1),
            "input_observation_props": torch.stack(meta_in, 1),
            "output_observation_props": torch.stack(meta_out, 1),
        }

        for anc_var in ANCILLARY_VARIABLES + ["ir_observations"]:
            anc_data = torch.tensor(data[anc_var].data).to(dtype=torch.float32)
            if self.rng.random() > 0.5:
                anc_data = np.nan * torch.tensor(data[anc_var].data).to(dtype=torch.float32)
            if anc_data.ndim < 3:
                anc_data = anc_data[None]
            if anc_data.ndim < 4:
                anc_data = anc_data[:, None]

            n_chans, n_seq, n_y, n_x = anc_data.shape
            anc_mask = torch.isnan(anc_data).all()[None]
            inpt[anc_var] = anc_data
            inpt[anc_var + "_mask"] = anc_mask

        obs = inpt["observations"]
        if torch.isfinite(obs).to(dtype=torch.float32).mean() < 0.5:
            LOGGER.info(
                "Less than 10 percent valid input observations in sample %s. Falling back to another, "
                "randomly chosen sample.",
                self.input_files[ind]
            )
            return self[self.rng.integers(0, len(self))]

        props = inpt["output_observation_props"]
        if torch.isfinite(props[0]).to(dtype=torch.float32).mean() < 0.5:
            LOGGER.info(
                "Less than 10 percent valid output observations in sample %s. Falling back to another, "
                "randomly chosen sample.",
                self.input_files[ind]
            )
            return self[self.rng.integers(0, len(self))]


        mask = torch.isnan(inpt["observations"]).all(0).all(-1).all(-1)
        inpt["input_observation_mask"] = mask
        target = {
            "output_observations": obs_out,
        }

        data.close()

        # Flip vertically
        if not self.validation and self.rng.random() > 0.5:
            for key in inpt:
                if not key.endswith("mask"):
                    inpt[key] = torch.flip(inpt[key], (-2,))
            for key in target:
                if isinstance(target[key], list):
                    target[key] = [torch.flip(targ, (-2,)) for targ in target[key]]
                else:
                    target[key] = torch.flip(target[key], (-2,))

        # Flip horizontally
        if not self.validation and self.rng.random() > 0.5:
            for key in inpt:
                if not key.endswith("mask"):
                    inpt[key] = torch.flip(inpt[key], (-1,))
            for key in target:
                if isinstance(target[key], list):
                    target[key] = [torch.flip(targ, (-1,)) for targ in target[key]]
                else:
                    target[key] = torch.flip(target[key], (-1,))

        # Transpose
        if not self.validation and self.rng.random() > 0.5:
            for key in inpt:
                if not key.endswith("mask"):
                    inpt[key] = torch.transpose(inpt[key], -2, -1)
            for key in target:
                if isinstance(target[key], list):
                    target[key] = [torch.transpose(targ, -2, -1) for targ in target[key]]
                else:
                    target[key] = torch.transpose(target[key], -2, -1)

        return inpt, target




class GPROFNNHRDataset:
    """
    Dataset for loading GPROF-NN HR training data from CloudSat and GPM CMB collocations.
    """
    def __init__(
        self,
        paths: Union[Path, List[Path]],
        seq_len_in: int = 13,
        validation: bool = False,
        channel_dropout: float = 0.1,
        augment: bool = True,
        subsample: float = 1.0,
        precip_threshold: float = 0.5,
        extra_targets: Optional[List[str]] = None
    ):
        """
        Args:
            paths: A single path or a list of paths pointing to the directories containing
                the training files.
            seq_len_in: The maximum number of input observations.
            validation: If 'True' will ensure reproducibility between invocations.
            channel_dropout: Probability with which to dropout input channels.
            augment: Whether or not to augment the training data with random flips.
            subsample: Whether or not to subsample the training data.
            precip_threshold: The threshold between which to switch between CloudSat and Combined precipitation
                to load.
            extra_targets: List of additional variables to load as targets. Can be used to load, e.g.,
                latitude and longitude coordinates into the target dictionary.
        """
        if isinstance(paths, list):
            input_files = sum(
                [sorted(list(Path(path).glob("*.nc"))) for path in paths],
                []
            )
        else:
            input_files = sorted(list(Path(paths).glob("*.nc")))

        self.cloudsat_stats = None
        cloudsat_files = [path for path in input_files if "_cloudsat_" in path.name]
        if len(cloudsat_files) > 0:
            self.cloudsat_stats = xr.load_dataset(cloudsat_files[0].parent / ".stats.nc")
            self.cloudsat_stats.longitude.data[0] = -180.0
            self.cloudsat_stats.longitude.data[-1] = 180.0
        self.cmb_stats = None
        cmb_files = [path for path in input_files if "_cmb_" in path.name]
        if len(cmb_files) > 0:
            self.cmb_stats = xr.load_dataset(cmb_files[0].parent / ".stats.nc")
            self.cmb_stats.longitude.data[0] = -180.0
            self.cmb_stats.longitude.data[-1] = 180.0

        if self.cloudsat_stats and self.cmb_stats:
            scene_weights_cloudsat = 1.0 / self.cloudsat_stats.scene_counts.data
            scene_weights_cloudsat[~np.isfinite(scene_weights_cloudsat)] = np.nan
            self.scene_weights_cloudsat = scene_weights_cloudsat / np.nanmean(scene_weights_cloudsat)
            scene_weights_cmb = 1.0 / self.cmb_stats.scene_counts.data
            self.scene_weights_cmb = scene_weights_cmb / np.nanmean(scene_weights_cmb)
            self.global_weights = self.cloudsat_stats.counts / self.cmb_stats.counts
        else:
            self.global_weights = None
            self.scene_weights_cmb = None
            self.scene_weights_cloudsat = None


        self.validation = validation
        self.init_rng()
        self.input_files = input_files
        self.drop_inputs = 1
        self.seq_len_in = seq_len_in
        self.channel_dropout = channel_dropout
        self.augment = augment
        self.precip_threshold = precip_threshold
        self.subsample = subsample
        self.extra_targets = extra_targets

        self.oversampling_factor = len(cmb_files) / len(cloudsat_files)
        self.cloudsat_files = cloudsat_files
        self.cmb_files = cmb_files


    def load_resampling_weights(self, input_files: np.ndarray) -> xr.Dataset:
        """
        Load training file resampling weights from cached file or recompute if necessary.

        Args:
            input_files:

        Return:
            An xarray.Dataset containing the input files and corresponding resample weights to
            achieve uniform coverage.
        """

        weight_file = Path(".") / "resampling_weights.nc"
        if not weight_file.exists():
            weights = self.calculate_resampling_weights(input_files)
            weights.to_netcdf(weight_file)
            return weights

        weights = xr.load_dataset(weight_file)
        input_files_cached = weights.input_files.data
        if not (input_files_cached == input_files).all():
            weights = self.calculate_resampling_weights(input_files)
            weights.to_netcdf(weight_file)
            return weights

        return weights


    def calculate_resampling_weights(self, input_files: np.ndarray):
        """
        Calculate resampling weights.

        Args:
            input_files: An xarra.ndarray containing all input file names

        Return:
            An xarray.Dataset containing the resampling weights for all given input files.
        """
        weights = []

        global_weights = self.cloudsat_stats.counts / self.cmb_stats.counts
        invalid = global_weights.data > 1e4
        global_weights.data[invalid] = np.nan

        for path in track(input_files):
            with xr.open_dataset(path) as data:
                lons = data.longitude.data
                lats = data.latitude.data
                mean_lon = lons.mean()
                mean_lat = lats.mean()
                weight = global_weights.interp(
                    longitude=mean_lon,
                    latitude=mean_lat,
                    method="nearest"
                ).data
            if "_cmb_" in path:
                weights.append(weight)
            else:
                if weight < 1e4:
                    weights.append(1.0)
                else:
                    weights.append(0.0)

        weights = xr.Dataset({
            "weights": (("samples",), np.array(weights)),
            "input_files": (("samples",), input_files)
        })
        return weights


    def init_rng(self, w_id=0):
        """
        Initialize random number generator.

        Args:
            w_id: The worker ID which of the worker process..
        """
        if self.validation:
            seed = 42
        else:
            seed = int.from_bytes(os.urandom(4), "big") + w_id
        self.rng = np.random.default_rng(seed)

    def worker_init_fn(self, w_id: int):
        """
        Pytorch retrieve interface.
        """
        self.init_rng(w_id)
        winfo = torch.utils.data.get_worker_info()
        n_workers = winfo.num_workers


    def __len__(self):
        return 2 * len(self.cmb_files)

    def __getitem__(self, ind: int):


        if ind % 2 == 1:
            input_file = self.cloudsat_files[(ind // 2) % len(self.cloudsat_files)]
        else:
            input_file = self.cmb_files[ind // 2]

        try:
            scene = xr.open_dataset(input_file)
        except Exception:
            return self[self.rng.integers(0, len(self))]

        if "total_precip" in scene:
            surface_precip = scene["total_precip"].data.copy()
        else:
            surface_precip = scene["surface_precip"].data.copy()

        if "cmb" in input_file.name:
            land_frac = scene.land_fraction.data
            valid_ref = (
                (surface_precip > self.precip_threshold) +
                ((land_frac > 0) * (surface_precip > 0))
            )
            if self.global_weights is not None:
                weights = self.global_weights.interp(latitude=scene.latitude, longitude=scene.longitude).data
                weights *= self.oversampling_factor
                weights *= self.scene_weights_cmb
                weights = torch.tensor(weights.astype(np.float32))

            else:
                weights = torch.ones((96, 96))
        else:
            valid_ref = surface_precip <= self.precip_threshold
            if self.global_weights is not None:
                weights = torch.tensor(self.scene_weights_cloudsat.astype(np.float32)[None, None])
            else:
                weights = torch.ones((96, 96))[None, None]


        surface_precip[~valid_ref] = np.nan

        valid = np.isfinite(scene.longitude.data) * np.isfinite(scene.latitude.data)

        obs_in = []
        meta_in = []

        input_observations = scene.input_observations.data
        input_meta = scene.input_meta_data.data
        chans_in = np.random.permutation(scene.input_observations.all_channels.size)

        if self.rng.random() > 0.5:
            width = self.rng.integers(1, 25)
        else:
            width = 0
        left = self.rng.random()

        missing_obs = np.any(np.isnan(input_observations), 0)
        surface_precip[missing_obs] = np.nan

        for input_ind in range(self.seq_len_in):
            rand = self.rng.random()
            if (rand >= self.channel_dropout) and input_ind < len(chans_in):
                obs = input_observations[chans_in[input_ind]]
                meta = input_meta[chans_in[input_ind]]
                if width > 0 and chans_in[input_ind] >= 9:
                    if left > 0.5:
                        obs[..., :width] = np.nan
                    else:
                        obs[..., -width:] = np.nan
                obs, meta = transform_observations_satformer(obs, meta)
                obs_in.append(torch.tensor(obs.astype(np.float32)))
                meta_in.append(torch.tensor(meta.astype(np.float32)))
            else:
                obs_in.append(torch.nan * torch.zeros((1, 96, 96)))
                meta_in.append(torch.nan * torch.zeros((8, 96, 96)))


        inpt = {
            "observations": torch.stack(obs_in, 1),
            "input_observation_props": torch.stack(meta_in, 1),
        }

        anc_vars = [
            "two_meter_temperature",
            "total_column_water_vapor",
            "leaf_area_index",
            "land_fraction",
            "ice_fraction",
            "elevation",
            "ir_observations",
        ]
        for anc_var in anc_vars:
            anc_data = torch.tensor(scene[anc_var].data).to(dtype=torch.float32)
            if self.rng.random() > 0.5:
                anc_data = np.nan * torch.tensor(scene[anc_var].data).to(dtype=torch.float32)
            if anc_data.ndim < 3:
                anc_data = anc_data[None]
            if anc_data.ndim < 4:
                anc_data = anc_data[:, None]

            valid = torch.isfinite(anc_data)

            n_chans, n_seq, n_y, n_x = anc_data.shape
            anc_mask = torch.isnan(anc_data).all()[None]
            inpt[anc_var] = anc_data
            inpt[anc_var + "_mask"] = anc_mask

        obs = inpt["observations"]
        if torch.isfinite(obs).to(dtype=torch.float32).mean() < 0.5:
            LOGGER.info(
                "Less than 10 percent valid input observations in sample %s. Falling back to another, "
                "randomly chosen sample.",
                input_file
            )
            return self[self.rng.integers(0, len(self))]


        mask = torch.isnan(inpt["observations"]).all(0).all(-1).all(-1)
        inpt["input_observation_mask"] = mask

        surface_precip = torch.tensor(surface_precip.astype(np.float32))
        weight_mask = ~torch.isfinite(weights) + (weights == 0.0)
        weights = torch.nan_to_num(weights, nan=0.0)
        surface_precip[weight_mask] = torch.nan
        target = {
            "surface_precip": [surface_precip],
            "surface_precip_weights": [weights],
        }
        if self.extra_targets is not None:
            if "surface_precip_cmb" in self.extra_targets:
                if "_cmb_" in input_file.name:
                    target["surface_precip_cmb"] = torch.tensor(scene.surface_precip.data)
                else:
                    target["surface_precip_cmb"] = torch.nan * torch.zeros((96, 96))

            if "surface_precip_cloudsat" in self.extra_targets:
                if "_cloudsat_" in input_file.name:
                    target["surface_precip_cloudsat"] = torch.tensor(scene.total_precip.data)
                else:
                    target["surface_precip_cloudsat"] = torch.nan * torch.zeros((96, 96))

            remaining_targets = [targ for targ in self.extra_targets if targ not in ["surface_precip_cmn", "surface_precip_cloudsat"]]
            for targ in remaining_targets:
                target[targ] = torch.tensor(scene[targ].data.astype(np.float32))

        scene.close()

        valid = torch.isfinite(target["surface_precip"][0] * target["surface_precip_weights"][0])
        if valid.sum() == 0:
            new_ind = self.rng.integers(0, len(self))
            LOGGER.warning(
                "No valid pixels in file %s. Falling back to another "
                " randomly-chosen sample.",
                input_file
            )
            return self[new_ind]

        # Flip vertically
        if not self.validation and self.rng.random() > 0.5:
            for key in inpt:
                if not key.endswith("mask"):
                    inpt[key] = torch.flip(inpt[key], (-2,))
            for key in target:
                if isinstance(target[key], list):
                    target[key] = [torch.flip(targ, (-2,)) for targ in target[key]]
                else:
                    target[key] = torch.flip(target[key], (-2,))

        # Flip horizontally
        if not self.validation and self.rng.random() > 0.5:
            for key in inpt:
                if not key.endswith("mask"):
                    inpt[key] = torch.flip(inpt[key], (-1,))
            for key in target:
                if isinstance(target[key], list):
                    target[key] = [torch.flip(targ, (-1,)) for targ in target[key]]
                else:
                    target[key] = torch.flip(target[key], (-1,))

        # Transpose
        if not self.validation and self.rng.random() > 0.5:
            for key in inpt:
                if not key.endswith("mask"):
                    inpt[key] = torch.transpose(inpt[key], -2, -1)
            for key in target:
                if isinstance(target[key], list):
                    target[key] = [torch.transpose(targ, -2, -1) for targ in target[key]]
                else:
                    target[key] = torch.transpose(target[key], -2, -1)

        return inpt, target
