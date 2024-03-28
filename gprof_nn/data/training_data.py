"""
===========================
gprof_nn.data.training_data
===========================

This module defines the dataset classes that provide access to
the training data for the GPROF-NN retrievals.
"""
import io
import itertools
import math
import logging
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import rotate
import torch
from torch.utils.data import Dataset, IterableDataset
import xarray as xr

from quantnn.normalizer import MinMaxNormalizer

from gprof_nn import sensors
from gprof_nn.utils import (
    calculate_interpolation_weights,
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
from gprof_nn.definitions import (
    ANCILLARY_VARIABLES,
    MASKED_OUTPUT,
    LAT_BINS,
    TIME_BINS,
    LIMITS,
    ALL_TARGETS,
    PROFILE_TARGETS
)
from gprof_nn.data.preprocessor import PreprocessorFile
from gprof_nn.augmentation import (get_transformation_coordinates,
                                   extract_domain)

LOGGER = logging.getLogger(__name__)


_THRESHOLDS = {
    "surface_precip": 1e-4,
    "convective_precip": 1e-4,
    "rain_water_path": 1e-4,
    "ice_water_path": 1e-4,
    "cloud_water_path": 1e-4,
    "total_column_water_vapor": 1e0,
    "rain_water_content": 1e-5,
    "cloud_water_content": 1e-5,
    "snow_water_content": 1e-5,
    "latent_heat": -99999,
    "snow": 1e-4,
    "snow3": 1e-4,
    "snow4": 1e-4
}

_INPUT_DIMENSIONS = {
    "GMI": (96, 128),
    "TMIPR": (96, 128),
    "TMIPO": (96, 128),
    "SSMI": (96, 128),
    "SSMIS": (32, 128),
    "AMSR2": (32, 128),
    "MHS": (32, 128),
    "ATMS": (32, 128),
}


EIA_GMI = np.array([
    [52.98] * 10 + [49.16] * 5
])


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


def write_preprocessor_file_xtrack(input_data, output_file):
    """
    Handle the special case of writing preprocessor files for cross
    track scanning sensors. The difficulty here is that GPROF expects
    pixels to be organized into pixel positions according to their
    viewing angle.
    """
    if not isinstance(input_data, xr.Dataset):
        data = xr.open_dataset(input_data)
    else:
        data = input_data

    sensor = data.attrs["sensor"]
    platform = data.attrs["platform"].replace("-", "")
    sensor = sensors.get_sensor(sensor, platform)
    sensor_name = sensor.name
    platform_name = sensor.platform.name

    eia = input_data.earth_incidence_angle.data
    bins = sensor.viewing_geometry.get_earth_incidence_angles()
    bins = 0.5 * (bins[1:] + bins[:-1])
    indices = np.digitize(eia, bins)
    cts, _ = np.histogram(indices, bins=np.arange(bins.size + 2) - 0.5)

    n_scans = cts.max()
    n_pixels = sensor.viewing_geometry.pixels_per_scan
    n_chans = sensor.n_chans

    if "pixels" not in data.dims or "scans" not in data.dims:
        dim_offset = -1
    else:
        if hasattr(data, "samples"):
            dim_offset = 1
        else:
            dim_offset = 0

    new_dataset = {
        "scans": np.arange(n_scans),
        "pixels": np.arange(n_pixels),
        "channels": np.arange(n_chans),
    }

    dims = ("scans", "pixels", "channels")
    for k in data:
        da = data[k]
        if k == "scan_time":
            t = da.data.ravel()[0]
            new_dataset[k] = (("scans",), np.repeat(t, n_scans))
        else:
            new_shape = (n_scans, n_pixels) + da.shape[(2 + dim_offset) :]
            new_shape = new_shape[: len(da.data.shape) - dim_offset]
            dims = ("scans", "pixels") + da.dims[2 + dim_offset :]
            if "pixels_center" in dims:
                continue

            new_data = -9999.9 * np.ones(new_shape, da.data.dtype)
            for i in range(n_pixels):
                mask = indices == i
                n_elems = mask.sum()
                new_data[:n_elems, i] = da.data[mask]

            if new_data.dtype in [np.float32, np.float64]:
                new_data = np.nan_to_num(new_data, nan=-9999.9)

            # if k == "airimass_type":
            #    new_data[new_data <= 0] = 0
            new_dataset[k] = (dims, new_data)

    if "nominal_eia" in data.attrs:
        new_dataset["earth_incidence_angle"] = (
            ("scans", "pixels", "channels"),
            np.broadcast_to(
                data.attrs["nominal_eia"].reshape(1, 1, -1), (n_scans, n_pixels, 15)
            ),
        )

    if "sunglint_angle" not in new_dataset:
        new_dataset["sunglint_angle"] = (
            ("scans", "pixels"),
            np.zeros_like(new_dataset["surface_type"][1]),
        )
    if "quality_flag" not in new_dataset:
        new_dataset["quality_flag"] = (
            ("scans", "pixels"),
            np.zeros_like(new_dataset["surface_type"][1]),
        )
    if "latitude" not in new_dataset:
        new_dataset["latitude"] = (
            ("scans", "pixels"),
            np.zeros_like(new_dataset["surface_type"][1]),
        )
    if "longitude" not in new_dataset:
        new_dataset["longitude"] = (
            ("scans", "pixels"),
            np.zeros_like(new_dataset["surface_type"][1]),
        )
    new_data = xr.Dataset(new_dataset)

    template_path = Path(__file__).parent / ".." / "files"
    template_file = template_path / f"{sensor_name.lower()}_{platform_name.lower()}.pp"
    if template_file.exists():
        template = PreprocessorFile(template_file)
    else:
        template = PreprocessorFile(template_path / "preprocessor_template.pp")
    PreprocessorFile.write(output_file, new_data, sensor, template=template)
    return new_data


def write_preprocessor_file(input_data, output_file):
    """
    Extract samples from training dataset and write to a preprocessor
    file.

    This functions serves as an interface between the GPROF-NN training
    data and the GPROF legacy algorithm as it can be used to create
    preprocessor files with the observations. These can then be used
    as input for the legacy GPROF.

    Note: If the input isn't organized into scenes with dimensions
    'scans' and 'pixels' the number of samples that will be written to
    the file will be the largest multiple of 256 that is smaller than
    or equal to the original number of samples. This means that up to
    255 samples may be lost when writing them to a preprocessor file.

    Args:
        input_data: Path to a NetCDF4 file containing the training or test
            data or 'xarray.Dataset containing the data to write to a
            preprocessor file.
        output_file: Path of the file to write the output to.
        template: Template preprocessor file use to determine the orbit header
             information. If not provided this data will be filled with dummy
             values.
    """
    if not isinstance(input_data, xr.Dataset):
        data = xr.open_dataset(input_data)
    else:
        data = input_data

    sensor = data.attrs["sensor"]
    platform = data.attrs["platform"].replace("-", "")
    sensor = sensors.get_sensor(sensor, platform)
    sensor_name = sensor.name
    platform_name = sensor.platform.name

    if "earth_incidence_angle" in input_data:
        return write_preprocessor_file_xtrack(input_data, output_file)

    if "pixels" not in data.dims or "scans" not in data.dims:
        if data.samples.size < 256:
            n_pixels = data.samples.size
            n_scans = 1
        else:
            n_pixels = 256
            n_scans = data.samples.size // n_pixels
        n_scenes = 1
        dim_offset = -1
    else:
        n_pixels = data.pixels.size
        n_scans = data.scans.size
        if hasattr(data, "samples"):
            n_scenes = data.samples.size
            dim_offset = 1
        else:
            n_scenes = 1
            dim_offset = 0

    c = math.ceil(n_scenes / (n_pixels * 256))
    if c > 256:
        raise ValueError(
            "The dataset contains too many observations to be savely "
            " converted to a preprocessor file."
        )
    n_scans_r = n_scans * n_scenes
    n_pixels_r = n_pixels

    n_chans = input_data.channels.size

    new_dataset = {
        "scans": np.arange(n_scans_r),
        "pixels": np.arange(n_pixels_r),
        "channels": np.arange(n_chans),
    }
    dims = ("scans", "pixels", "channels")
    for k in data:
        da = data[k]
        if k == "scan_time":
            new_dataset[k] = (("scans",), da.data.ravel()[:n_scans_r])
        else:
            new_shape = (n_scans_r, n_pixels_r) + da.shape[(2 + dim_offset) :]
            new_shape = new_shape[: len(da.data.shape) - dim_offset]
            dims = ("scans", "pixels") + da.dims[2 + dim_offset :]
            if "pixels_center" in dims:
                continue
            n_elems = np.prod(new_shape)
            elements = da.data.ravel()[:n_elems]
            if elements.dtype in [np.float32, np.float64]:
                elements = np.nan_to_num(elements, nan=-9999.9)
            # if k == "airmass_type":
            #    elements[elements <= 0] = 1
            new_dataset[k] = (dims, elements.reshape(new_shape))

    if "nominal_eia" in data.attrs:
        new_dataset["earth_incidence_angle"] = (
            ("scans", "pixels", "channels"),
            np.broadcast_to(
                data.attrs["nominal_eia"].reshape(1, 1, -1), (n_scans_r, n_pixels_r, 15)
            ),
        )

    if "sunglint_angle" not in new_dataset:
        new_dataset["sunglint_angle"] = (
            ("scans", "pixels"),
            np.zeros_like(new_dataset["surface_type"][1]),
        )
    if "quality_flag" not in new_dataset:
        new_dataset["quality_flag"] = (
            ("scans", "pixels"),
            np.zeros_like(new_dataset["surface_type"][1]),
        )
    if "latitude" not in new_dataset:
        new_dataset["latitude"] = (
            ("scans", "pixels"),
            np.zeros_like(new_dataset["surface_type"][1]),
        )
    if "longitude" not in new_dataset:
        new_dataset["longitude"] = (
            ("scans", "pixels"),
            np.zeros_like(new_dataset["surface_type"][1]),
        )
    new_data = xr.Dataset(new_dataset)

    template_path = Path(__file__).parent / ".." / "files"
    template_file = template_path / f"{sensor_name.lower()}_{platform_name.lower()}.pp"
    if template_file.exists():
        template = PreprocessorFile(template_file)
    else:
        template = PreprocessorFile(template_path / "preprocessor_template.pp")
    PreprocessorFile.write(output_file, new_data, sensor, template=template)



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
        sensor: sensors.Sensor
) -> torch.Tensor:
    """
    Load brightness temperatures for cross-track scanning sensors from simulator
    collocations.

    Args:
        training_data: An xarray.Dataset containing training data extracted from
            GPROF simulator files.
        angles: A np.ndarray cotaining the viewing angle of the tbs to load.
        sensor: The sensor from which the TBs are loaded

    Return:
        A torch tensor containing the loaded brightness temperatures.

    """
    samples = np.arange(training_data.samples.size)
    samples = xr.DataArray(samples, dims="samples")
    angles = xr.DataArray(np.abs(angles), dims="samples")

    training_data = training_data[
        ["simulated_brightness_temperatures", "brightness_temperature_biases"]
    ]
    training_data = training_data.interp(samples=samples, angles=angles)
    tbs = training_data.simulated_brightness_temperatures.data

    tbs_full = np.nan * np.zeros((tbs.shape[0], 15), dtype=np.float32)
    tbs_full[:, sensor.gmi_channels] = tbs

    biases = training_data.brightness_temperature_biases.data
    biases_full = np.nan * np.zeros((tbs.shape[0], 15), dtype=np.float32)
    biases_full[:, sensor.gmi_channels] = biases

    biases = (
        biases_full /
        np.cos(np.deg2rad(EIA_GMI))[None] *
        np.cos(np.deg2rad(angles.data[..., None]))
    )

    return torch.tensor(tbs_full - biases)


def load_tbs_1d_conical_sim(
        training_data: xr.Dataset,
        sensor: sensors.Sensor
) -> torch.Tensor:
    """
    Load brightness temperatures for cross-track scanning sensors from simulator
    collocations.

    Args:
        training_data: An xarray.Dataset containing training data extracted from
            GPROF simulator files.
        angles: A np.ndarray cotaining the viewing angle of the tbs to load.
        sensor: The sensor from which the TBs are loaded

    Return:
        A torch tensor containing the loaded brightness temperatures.

    """
    training_data = training_data[
        [
            "simulated_brightness_temperatures",
            "brightness_temperature_biases",
        ]
    ]
    tbs = training_data.simulated_brightness_temperatures.data
    biases = training_data.brightness_temperature_biases.data

    tbs = tbs - biases
    return torch.tensor(tbs)


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
    tbs_full[:, sensor.gmi_channels] = tbs
    angles = training_data["earth_incidence_angle"].data
    angles_full = np.broadcast_to(angles[..., None], tbs_full.shape)

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
    tbs_full[:, sensor.gmi_channels] = tbs
    angles = training_data["earth_incidence_angle"].data
    angles_full = np.nan * np.ones(tbs.shape[:-1] + (15,), dtype="float32")
    angles_full[:, sensor.gmi_channels] = angles
    tbs = torch.tensor(tbs_full.astype("float32"))
    angles = torch.tensor(angles_full.astype("float32"))
    return tbs, angles



def load_ancillary_data_1d(training_data: xr.Dataset,) -> torch.Tensor:
    """
    Load brightness temperatures for GMI training data.

    Args:
        training_data: The xarray.Dataset containing the training data.

    Return:
        A torch tensor containign the ancillary data concatenated along the
        last dimension.
    """
    data = []
    for var in ANCILLARY_VARIABLES:
        data.append(training_data[var].data)
    data = np.stack(data, -1)
    return torch.tensor(data)


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
                shape = (n_samples, 1, 28)
            else:
                shape = (n_samples, 1, 28)
            data_t = np.zeros(shape, dtype=np.float32)

        if data_t.ndim == 1:
            data_t = data_t[..., None]
        targs[var] = torch.tensor(data_t.astype("float32"))
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

    training_data = training_data[targets]
    training_data = training_data.interp(samples=samples, angles=angles)

    targs = {}
    for var in targets:
        data_t = training_data[var].data
        if data_t.ndim == 1:
            data_t = data_t[..., None]
        targs[var] = torch.tensor(data_t.astype("float32"))
    return targs


class GPROFNN1DDataset(IterableDataset):
    """
    Dataset class for loading the training data for GPROF-NN 1D retrieval.
    """
    combine_files = 4

    def __init__(
        self,
        path: Path,
        targets: Optional[List[str]] = None,
        transform_zeros: bool = True,
        augment: bool = True,
        validation: bool = False,
    ):
        """
        Create GPROF-NN 1D dataset.

        The GPROF-NN 1D data is split up into separate files by orbit. This
        dataset loads the training data from all available files. And provides
        an iterable over the samples in the dataset.

        Args:
            path: The path containing the training data files.
            targets: A list of the target variables to load.
            transform_zeros: Whether or not to replace zeros in the output
                with small random values.
            augment: Whether or not to apply data augmentation to the loaded
                data.
            validation: If set to 'True', data  loaded in consecutive iterations
                over the dataset will be identical.
        """
        super().__init__()

        if targets is None:
            targets = ALL_TARGETS

        self.targets = targets
        self.transform_zeros = transform_zeros
        self.validation = validation
        self.augment = augment

        self.path = Path(path)
        if not self.path.exists():
            raise RuntimeError(
                "The provided path does not exists."
            )

        files = sorted(list(self.path.glob("*_*_*.nc")))
        if len(files) == 0:
            raise RuntimeError(
                "Could not find any GPROF-NN 1D training data files "
                f"in {self.path}."
            )
        self.files = files

        self.init_rng()
        self.files = self.rng.permutation(self.files)


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

    def worker_init_fn(self, w_id: int) -> None:
        """
        Initializes the worker state for parallel data loading.

        Args:
            w_id: The ID of the worker.
        """
        self.init_rng(w_id)
        winfo = torch.utils.data.get_worker_info()
        n_workers = winfo.num_workers

        self.files = self.files[w_id::n_workers]

    def load_training_data(self, dataset: xr.Dataset) -> Dict[str, torch.Tensor]:

        sensor = sensors.get_sensor(dataset.attrs["sensor"])
        targets = self.targets
        ref_target = targets[0]

        if sensor == sensors.GMI:
            tbs = dataset["brightness_temperatures"].data
            y_t = dataset[ref_target].data
            valid_input = np.any(tbs > 0, -1)
            valid_target = np.isfinite(y_t).any(tuple(range(1, y_t.ndim)))
            mask = valid_input * valid_target
            dataset = dataset[{"samples": mask}]

            tbs = load_tbs_1d_gmi(dataset)
            anc = load_ancillary_data_1d(dataset)
            targets = load_targets_1d(dataset, self.targets)
            angs = torch.tensor(np.broadcast_to(EIA_GMI.astype("float32"), tbs.shape))

        elif isinstance(sensor, sensors.CrossTrackScanner):

            if dataset.attrs["source"] == "sim":
                tbs = dataset["brightness_temperatures"].data
                y_t = dataset[ref_target].data
                valid_input = np.any(tbs > 0, -1)
                valid_target = np.isfinite(y_t).any(tuple(range(1, y_t.ndim)))
                mask = valid_input * valid_target
                dataset = dataset[{"samples": mask}]
                angles = dataset["angles"].data
                angs = self.rng.uniform(
                    angles.min(),
                    angles.max(),
                    size=dataset.samples.size,
                ).astype(np.float32)
                tbs = load_tbs_1d_xtrack_sim(dataset, angs, sensor)
                angs = torch.tensor(angs)
            else:
                tbs = dataset["brightness_temperatures"].data
                y_t = dataset[ref_target].data
                valid_input = np.any(tbs > 0, -1)
                valid_target = np.isfinite(y_t).any(tuple(range(1, y_t.ndim)))
                mask = valid_input * valid_target
                dataset = dataset[{"samples": mask}]
                tbs, angs = load_tbs_1d_xtrack_other(dataset, sensor)

            anc = load_ancillary_data_1d(dataset)
            targets = load_targets_1d(dataset, self.targets)

        elif isinstance(sensor, sensors.ConicalScanner):

            if dataset.source == "sim":
                tbs = load_tbs_1d_conical_sim(dataset, sensor)
                angs = torch.tensor(np.broadcast_to(EIA_GMI.astype("float32"), tbs.shape))
            else:
                tbs, angs = load_tbs_1d_conical_other(dataset, sensor)
            anc = load_ancillary_data_1d(dataset)
            targets = load_targets_1d(dataset, self.targets)

        x = {
            "brightness_temperatures": tbs,
            "ancillary_data": anc,
            "viewing_angles": angs
        }
        return x, targets


    def __repr__(self):
        return f"GPROFNN1DDataset(path={self.path}, targets={self.targets})"

    def __iter__(self):

        all_files = self.rng.permutation(self.files)
        for ind in range(0, len(self.files), self.combine_files):
            files = all_files[ind:ind + self.combine_files]

            inputs = {}
            targets = {}

            for path in files:
                with xr.open_dataset(path) as input_file:

                    inputs_f, targets_f = self.load_training_data(input_file)
                    for name, tensor in inputs_f.items():
                        inputs.setdefault(name, []).append(tensor)
                    for name, tensor in targets_f.items():
                        targets.setdefault(name, []).append(tensor)


            inputs = {name: torch.cat(data, 0) for name, data in inputs.items()}
            targets = {name: torch.cat(data, 0) for name, data in targets.items()}

            n_samples = inputs["brightness_temperatures"].shape[0]
            for ind in self.rng.permutation(n_samples):
                yield (
                    {name: data[ind] for name, data in inputs.items()},
                    {name: data[ind] for name, data in targets.items()}
                )


def load_training_data_3d_gmi(
        scene: xr.Dataset,
        targets: List[str],
        augment: bool = False,
        rng: np.random.Generator = None,
) -> Tuple[Dict[str, torch.Tensor]]:
    """
    Load GPROF-NN 3D training scene for GMI.

    Args:
        scene: An xarray.Dataset containing the scene from which to load
            the training data.
        targets: A list containing a list of the targets to load.
        augment: Whether or not to augment the input data.
        rng: A numpy random number generator to use for the augmentation.

    Return:
        A tuple ``(x, y)`` of dictionaries ``x`` and ``y`` containing the
        training input data in ``x`` and the training reference data in ``y``.
    """
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
    coords = get_transformation_coordinates(
        lats, lons, sensors.GMI.viewing_geometry, 64, 128, p_x_i, p_x_o, p_y
    )
    scene = remap_scene(scene, coords, variables)

    tbs = torch.tensor(scene.brightness_temperatures.data)
    angs = torch.tensor(np.broadcast_to(EIA_GMI.astype("float32"), tbs.shape))
    anc = torch.tensor(np.stack(
        [scene[anc_var].data.astype("float32") for anc_var in ANCILLARY_VARIABLES]
    ))
    tbs = torch.permute(tbs, (2, 0, 1))
    angs = torch.permute(angs, (2, 0, 1))

    x = {
        "brightness_temperatures": tbs,
        "viewing_angles": angs,
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
            y[target] = empty
            continue

        data = torch.tensor(scene[target].data.astype("float32"))
        dims = tuple(range(data.ndim))
        data = torch.permute(data, dims[-2:] + dims[:-2])
        y[target] = data

    return x, y


def load_training_data_3d_xtrack_sim(
        sensor: sensors.Sensor,
        scene: xr.Dataset,
        targets: List[str],
        augment: bool = False,
        rng: np.random.Generator = None,
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

    Return:
        A tuple ``(x, y)`` of dictionaries ``x`` and ``y`` containing the
        training input data in ``x`` and the training reference data in ``y``.
    """
    required = [
        "latitude",
        "longitude",
        "simulated_brightness_temperatures",
        "brightness_temperature_biases"
    ]
    variables = [
        name for name in targets + required
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

    width = 64
    height = 128

    lats = scene.latitude.data
    lons = scene.longitude.data
    coords = get_transformation_coordinates(
        lats, lons, sensor.viewing_geometry, width, height, p_x_i, p_x_o, p_y
    )
    scene = remap_scene(scene, coords, variables)

    center = sensor.viewing_geometry.get_window_center(p_x_o, width)
    j_start = int(center[1, 0, 0] - width // 2)
    j_end = int(center[1, 0, 0] + width // 2)
    angs = sensor.viewing_geometry.get_earth_incidence_angles()
    angs = angs[j_start:j_end]
    angs = np.repeat(angs.reshape(1, -1), height, axis=0)
    weights = calculate_interpolation_weights(np.abs(angs), sensor.angles)
    weights = np.repeat(weights.reshape(1, -1, sensor.n_angles), height, axis=0)
    weights = calculate_interpolation_weights(np.abs(angs), scene.angles.data)

    # Calculate brightness temperatures
    tbs_sim = scene.simulated_brightness_temperatures.data
    tbs_sim = interpolate(tbs_sim, weights)
    tb_biases = scene.brightness_temperature_biases.data
    tbs = tbs_sim - tb_biases

    full_shape = tbs_sim.shape[:2] + (15,)
    tbs_full = np.nan * np.ones(full_shape, dtype="float32")
    tbs_full[:, :, sensor.gmi_channels] = tbs
    tbs_full = torch.permute(torch.tensor(tbs_full), (2, 0, 1))

    angs_full = np.nan * np.ones(full_shape, dtype="float32")
    angs_full[:, :, sensor.gmi_channels] = angs[..., None]
    angs_full = torch.permute(torch.tensor(angs_full), (2, 0, 1))

    anc = torch.tensor(np.stack(
        [scene[anc_var].data.astype("float32") for anc_var in ANCILLARY_VARIABLES]
    ))

    x = {
        "brightness_temperatures": tbs_full,
        "viewing_angles": angs_full,
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
            y[target] = empty
            continue


        data = scene[target].data.astype("float32")

        if "angles" in scene[target].dims:
            data = interpolate(data, weights)

        data = torch.tensor(data)
        dims = tuple(range(data.ndim))
        data = torch.permute(data, dims[-2:] + dims[:-2])
        y[target] = data

    return x, y


def load_training_data_3d_conical_sim(
        sensor: sensors.Sensor,
        scene: xr.Dataset,
        targets: List[str],
        augment: bool = False,
        rng: np.random.Generator = None,
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

    Return:
        A tuple ``(x, y)`` of dictionaries ``x`` and ``y`` containing the
        training input data in ``x`` and the training reference data in ``y``.
    """
    required = [
        "latitude",
        "longitude",
        "simulated_brightness_temperatures",
        "brightness_temperature_biases"
    ]
    variables = [
        name for name in targets + required
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

    width = 64
    height = 128

    lats = scene.latitude.data
    lons = scene.longitude.data
    coords = get_transformation_coordinates(
        lats, lons, sensor.viewing_geometry, width, height, p_x_i, p_x_o, p_y
    )
    scene = remap_scene(scene, coords, variables)

    # Calculate brightness temperatures
    tbs_sim = scene.simulated_brightness_temperatures.data
    tb_biases = scene.brightness_temperature_biases.data
    tbs = torch.tensor(tbs_sim - tb_biases, dtype=torch.float32)
    tbs = torch.permute(tbs, (2, 0, 1))

    angs_full = torch.tensor(
        np.broadcast_to(EIA_GMI.astype("float32")[0][..., None, None], tbs.shape)
    )
    for ind in range(15):
        if ind not in sensor.gmi_channels:
            angs_full[ind] = np.nan
    angs_full = torch.tensor(angs_full)

    anc = torch.tensor(np.stack(
        [scene[anc_var].data.astype("float32") for anc_var in ANCILLARY_VARIABLES]
    ))

    x = {
        "brightness_temperatures": tbs,
        "viewing_angles": angs_full,
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
            y[target] = empty
            continue


        data = scene[target].data.astype("float32")

        data = torch.tensor(data)
        dims = tuple(range(data.ndim))
        data = torch.permute(data, dims[-2:] + dims[:-2])
        y[target] = data

    return x, y


def load_training_data_3d_other(
        sensor: sensors.Sensor,
        scene: xr.Dataset,
        targets: List[str],
        augment: bool = False,
        rng: np.random.Generator = None,
) -> Tuple[Dict[str, torch.Tensor]]:
    """
    Load training data for non-GMI sensors that are training scenes extracted
    from actualy observations, i.e., not .sim-file derived.

    Args:
        sensor: The sensor object from which the training data was extracted.
        scene: An xarray.Dataset containing the scene from which to load
            the training data.
        targets: A list containing a list of the targets to load.
        augment: Whether or not to augment the input data.
        rng: A numpy random number generator to use for the augmentation.

    Return:
        A tuple ``(x, y)`` of dictionaries ``x`` and ``y`` containing the
        training input data in ``x`` and the training reference data in ``y``.
    """
    required = [
        "latitude",
        "longitude",
        "simulated_brightness_temperatures",
        "brightness_temperature_biases"
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
        scn_start = (scene.scns.size - height) // 2
    pix_end = pix_start + width
    scn_end = scn_start + height
    scene = scene[{"pixels": slice(pix_start, pix_end), "scans": slice(scn_start, scn_end)}]


    # Calculate brightness temperatures
    tbs = scene.brightness_temperatures.data
    full_shape = tbs.shape[:2] + (15,)
    if tbs.shape != full_shape:
        tbs_full = np.nan * np.ones(full_shape, dtype="float32")
        tbs_full[:, :, sensor.gmi_channels] = tbs
    else:
        tbs_full = tbs
    tbs_full = torch.permute(torch.tensor(tbs_full), (2, 0, 1))

    angs = scene.earth_incidence_angle.data
    if angs.ndim == 2:
        angs = angs[..., None]
    if tbs.shape != full_shape:
        angs_full = np.nan * np.ones(full_shape, dtype="float32")
        angs_full[:, :, sensor.gmi_channels] = angs
    else:
        angs_full = angs
    angs_full = torch.permute(torch.tensor(angs_full), (2, 0, 1))

    anc = torch.tensor(np.stack(
        [scene[anc_var].data.astype("float32") for anc_var in ANCILLARY_VARIABLES]
    ))

    x = {
        "brightness_temperatures": tbs_full,
        "viewing_angles": angs_full,
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
            y[target] = empty
            continue


        data = scene[target].data.astype("float32")

        data = torch.tensor(data)
        dims = tuple(range(data.ndim))
        data = torch.permute(data, dims[-2:] + dims[:-2])
        y[target] = data

    return x, y


class GPROFNN3DDataset(Dataset):
    """
    Dataset class for loading the training data for GPROF-NN 3D retrieval.
    """

    def __init__(
        self,
        path: Path,
        targets: Optional[List[str]] = None,
        transform_zeros: bool = True,
        augment: bool = True,
        validation: bool = False
    ):
        """
        Create GPROF-NN 3D dataset.

        The training data for the GPROF-NN 3D retrieval consists of 2D scenes
        in separate files.

        Args:
            path: The path containing the training data files.
            targets: A list of the target variables to load.
            transform_zeros: Whether or not to replace zeros in the output
                with small random values.
            augment: Whether or not to apply data augmentation to the loaded
                data.
            validation: If set to 'True', data  loaded in consecutive iterations
                over the dataset will be identical.
        """
        super().__init__()

        if targets is None:
            targets = ALL_TARGETS
        self.targets = targets
        self.transform_zeros = transform_zeros
        self.validation = validation
        self.augment = augment and not validation
        self.validation = validation

        self.path = Path(path)
        if not self.path.exists():
            raise RuntimeError(
                "The provided path does not exists."
            )

        files = sorted(list(self.path.glob("*_*_*.nc")))
        if len(files) == 0:
            raise RuntimeError(
                "Could not find any GPROF-NN 3D training data files "
                f"in {self.path}."
            )
        self.files = files


        self.init_rng()
        self.files = self.rng.permutation(self.files)


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
        return len(self.files)

    def __getitem__(self, ind):
        with xr.open_dataset(self.files[ind]) as scene:
            sensor = scene.attrs["sensor"]
            sensor = getattr(sensors, sensor)

            if sensor == sensors.GMI:
                return load_training_data_3d_gmi(
                    scene,
                    targets=self.targets,
                    augment=self.augment,
                    rng=self.rng
                )
            elif isinstance(sensor, sensors.CrossTrackScanner):
                if scene.source == "sim":
                    return load_training_data_3d_xtrack_sim(
                        sensor,
                        scene,
                        targets=self.targets,
                        augment=self.augment,
                        rng=self.rng
                    )
                return load_training_data_3d_other(
                    sensor,
                    scene,
                    targets=self.targets,
                    augment=self.augment,
                    rng=self.rng
                )
            elif isinstance(sensor, sensors.ConicalScanner):
                if scene.source == "sim":
                    return load_training_data_3d_conical_sim(
                        sensor,
                        scene,
                        targets=self.targets,
                        augment=self.augment,
                        rng=self.rng
                    )
                return load_training_data_3d_other(
                    sensor,
                    scene,
                    targets=self.targets,
                    augment=self.augment,
                    rng=self.rng
                )
        raise RuntimeError(
            "Invalid sensor/scene combination in training file %s.",
            self.files[ind]
        )


class SimulatorDataset(Dataset):
    """
    Dataset class for loading the training data for training GPROF-NN observation
    emulators.
    """
    def __init__(
        self,
        path: Path,
        transform_zeros: bool = True,
        augment: bool = True,
        validation: bool = False
    ):
        """
        Create GPROF-NN 3D dataset.

        The training data for the GPROF-NN 3D retrieval consists of 2D scenes
        in separate files.

        Args:
            path: The path containing the training data files.
            transform_zeros: Whether or not to replace zeros in the output
                with small random values.
            augment: Whether or not to apply data augmentation to the loaded
                data.
            validation: If set to 'True', data  loaded in consecutive iterations
                over the dataset will be identical.
        """
        super().__init__()

        self.transform_zeros = transform_zeros
        self.validation = validation
        self.augment = augment and not validation
        self.validation = validation

        self.path = Path(path)
        if not self.path.exists():
            raise RuntimeError(
                "The provided path does not exists."
            )

        files = sorted(list(self.path.glob("sim_*_*.nc")))
        if len(files) == 0:
            raise RuntimeError(
                "Could not find any GPROF-NN Simulator training data files "
                f"in {self.path}."
            )
        self.files = files

        self.init_rng()
        self.files = self.rng.permutation(self.files)


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
        return f"SimulatorDataset(path={self.path})"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, ind):
        with xr.open_dataset(self.files[ind]) as scene:
            sensor = scene.attrs["sensor"]
            sensor = getattr(sensors, sensor)

            return load_training_data_3d_gmi(
                scene,
                targets=[
                    "simulated_brightness_temperatures",
                    "brightness_temperature_biases"
                ],
                augment=self.augment,
                rng=self.rng
            )
