"""
gprof_nn.training
=================

Interface for training of the GPROF-NN retrievals.
"""

def init(path: Path,
         configuration: str,
         training_data_path: Path,
         validation_data_path: Optional[Path] = None,
         targets: List[str] = ALL_TARGETS,
         ancillary_data: bool = True,
) -> None:

    config_path = Path(__file__).parent / "config_files"

    training_config = config_path / f"gprof_nn_{config.lower()}_training.toml"
    training_config = open(training_config, "r").read()
    training.config = training_config.format({
        "training_dataset_args": f"{{path = '{training_data_path}'}}",
        "validation_dataset_args": f"{{path = '{validation_data_path}'}}",
    })
    with open(path / "training.toml") as output:
        output.write(training_config)

    model_config = config_path / f"gprof_nn_{config.lower()}_training.toml"
    model_config = open(training_config, "r").read()
    for target in targets:
