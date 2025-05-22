"""
gprof_nn.config
===============

Provides objects for managing the 'gprof_nn' system configuration.
"""
from configparser import ConfigParser, SectionProxy
from dataclasses import dataclass, asdict
import logging
import os
from pathlib import Path
from typing import Optional
import sys

from appdirs import (
    user_config_dir,
    user_data_dir
)

import click


LOGGER = logging.getLogger(__name__)


class ConfigBase:
    """
    Base class for config classes implementing a generic parse function to parse
    config classes from .ini files.
    """
    @classmethod
    def parse(cls, section: SectionProxy):
        """
        Parse settings from corresponding section of the config.ini
        file.
        """
        settings = asdict(cls())
        config = cls()
        for key, value in settings.items():
            value = Path(section.get(key, value))
            setattr(config, key, value)
        return config


@dataclass
class PreprocessorConfig(ConfigBase):
    """
    Dataclass holding the preprocessor executables.
    """
    gmi: Path = Path("gprof2024pp_GMI_L1C")
    mhs: Path = Path("gprof2024pp_MHS_L1C")
    tmi: Path = Path("gprof2024pp_TMI_L1C")
    ssmi: Path = Path("gprof2024pp_SSMI_L1C")
    ssmis: Path = Path("gprof2024pp_SSMIS_L1C")
    amsr2: Path = Path("gprof2024pp_AMSR2_L1C")
    amsre: Path = Path("gprof2024pp_AMSRE_L1C")
    atms: Path = Path("gprof2024pp_ATMS_L1C")

    def print(self):
        txt = "[preprocessor]\n"
        txt += f"amsre = {self.amsre}\n"
        txt += f"amsr2 = {self.amsr2}\n"
        txt += f"atms  = {self.atms}\n"
        txt += f"gmi   = {self.gmi}\n"
        txt += f"mhs   = {self.mhs}\n"
        txt += f"ssmi  = {self.ssmi}\n"
        txt += f"ssmis = {self.ssmis}\n"
        txt += f"tmi = {self.tmi}\n"
        return txt


@dataclass
class DataConfig(ConfigBase):
    """
    Dataclass holding the paths of required data.
    """
    era5_path : Path = Path("/qdata2/archive/ERA5")
    model_path : Path = Path(user_data_dir("gprof_nn", "gprof_nn")) / "models"
    mrms_path : Path = Path("/pdata4/veljko/")

    def print(self):
        txt = "[data]\n"
        txt += f"era5_path:  {self.era5_path}\n"
        txt += f"model_path: {self.model_path}\n"
        txt += f"mrms_path:  {self.mrms_path}\n"
        return txt


@dataclass
class Config:
    """
    Dataclass bundling all configuration categories.
    """
    data: DataConfig
    preprocessor: PreprocessorConfig

    def print(self):
        txt = ""
        txt += self.data.print()
        txt += "\n"
        txt += self.preprocessor.print()
        return txt


def parse_config_file(path: Optional[Path] = None):
    """
    Parse config file.

    Parses a 'gprof_nn.config' file and returns a Config object representing
    the configuration.

    Args:
        path: If given, the configuration will be read from the given file.
    """
    preprocessor_config = PreprocessorConfig()
    data_config = DataConfig()

    if path is None:
        path = Path(user_config_dir("gprof_nn", "gprof_nn")) / "config.ini"

    if path.exists():
        parser = ConfigParser()
        parser.read(path)
        for section_name in parser.sections():
            sec = parser[section_name]
            if section_name == "preprocessor":
                preprocessor_config = PreprocessorConfig.parse(sec)
            elif section_name == "data":
                data_config = DataConfig.parse(sec)
            else:
                raise ValueError(
                    f"Config file contains an unknown section "
                    "'{section_Name}'."
                )

    return Config(data_config, preprocessor_config)


CONFIG = parse_config_file()


def get_config_file() -> Path:
    """
    Determine path of config file.
    """
    if "GPROF_NN_CONFIG" in os.environ:
        config_file = Path(os.environ["GPROF_NN_CONFIG"])
        return config_file
    return Path(user_config_dir("gprof_nn", "gprof_nn")) / "config.ini"


def show_config() -> None:
    """
    Show the current gprof_nn system configuration.

    """
    print("\n" + CONFIG.print())


def file() -> None:
    """
    Print the file from which the config is read.
    """
    print(get_config_file())


@click.argument("config_type", type=str)
@click.argument("property_name", type=str)
@click.argument("value", type=str)
def set_config(config_type, property_name, value) -> None:
    """
    Set configuration PROPERTY_NAME of CONFIGU_TYPE to VALUE.
    """

    if not config_type.lower() in ["data", "preprocessor"]:
        LOGGER.error(
            "'config_type' should be one of ['data', 'preprocessor'] "
            f" not '{config_type}'."
        )
        sys.exit(1)

    config = getattr(CONFIG, config_type.lower())

    prop_name = property_name.lower()
    if not hasattr(config, prop_name):
        LOGGER.error(
            f"'property_name' should be one of {list(asdict(config).keys())} "
            f"not '{property_name}'."
        )
        sys.exit(1)

    setattr(config, property_name.lower(), value)

    path = get_config_file()
    if not path.parent.exists():
        path.parent.mkdir(exist_ok=True, parents=True)
    with open(get_config_file(), "w") as out:
        out.write(CONFIG.print())
