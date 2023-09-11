from configparser import ConfigParser, SectionProxy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from appdirs import (
    user_config_dir,
    user_data_dir
)

class ConfigBase:
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
    GMI: Path = Path("gprof2023pp_GMI_L1C")
    MHS: Path = Path("gprof2021pp_MHS_L1C")
    TMIPR: Path = Path("gprof2021pp_TMI_L1C")
    TMIPO: Path = Path("gprof2021pp_TMI_L1C")
    SSMI: Path = Path("gprof2021pp_SSMI_L1C")
    SSMIS: Path = Path("gprof2021pp_SSMIS_L1C")
    AMSR2: Path = Path("gprof2023pp_AMSR2_L1C")
    AMSRE: Path = Path("gprof2021pp_AMSRE_L1C")
    ATMS: Path = Path("gprof2021pp_ATMS_L1C")


    def print(self):
        txt = ""
        txt += f"AMSRE: {self.AMSRE}\n"
        txt += f"AMSR2: {self.ASMR2}\n"
        txt += f"ATMS: {self.ATMS}\n"
        txt += f"GMI: {self.GMI}\n"
        txt += f"MHS: {self.MHS}\n"
        txt += f"SSMI: {self.SSMI}\n"
        txt += f"SSMIS: {self.SSMIS}\n"
        txt += f"TMIPR: {self.TMIPR}\n"
        txt += f"TMIPO: {self.TMIPO}\n"
        return txt


@dataclass
class DataConfig:
    """
    Dataclass holding the paths of required data.
    """
    model_path : Path = Path(user_data_dir("gprof_nn", "gprof_nn")) / "models"
    era5_path : Path = Path("/qdata2/archive/ERA5")

    def print(self):
        txt = ""
        txt += f"era5_path: {self.era5_path}\n"
        txt += f"model_path: {self.model_path}\n"
        return txt

@dataclass
class Config:
    data: DataConfig
    preprocessor: PreprocessorConfig


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
