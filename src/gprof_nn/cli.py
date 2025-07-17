"""
gprof_nn.cli
============

This module implements the command line interfaces for the functionality
provided by the 'gprof_nn' package.
"""
import logging

import click
import gprof_nn.logging
import gprof_nn.config as conf
from gprof_nn import training
from gprof_nn import retrieval, testing
from gprof_nn import download as dl
from gprof_nn import sensors


LOGGER = logging.getLogger(__name__)


@click.group()
def gprof_nn():
    pass


# Make data extraction commands available only if pansat is installed.
try:
    from gprof_nn.data import (
        sim,
        pretraining,
        mrms,
        era5,
        finetuning,
        cloudsat,
        combined,
    )

    @gprof_nn.group(name="extract_training_data")
    def extract_training_data():
        """
        Extract GPROF-NN training data.
        """
        pass

    extract_training_data.command(name="sim")(sim.cli)
    extract_training_data.command(name="pre")(pretraining.cli)
    extract_training_data.command(name="mrms")(mrms.cli)
    extract_training_data.command(name="era5")(era5.cli)
    extract_training_data.command(name="finetuning")(finetuning.cli)
    extract_training_data.command(name="cloudsat")(cloudsat.cli)
    extract_training_data.command(name="combined")(combined.cli)
except ImportError as err:
    LOGGER.debug(
        "Disabling training data extraction because of missing dependencies."
    )
    pass


######################################################################
# gprof_nn config
######################################################################


@gprof_nn.group(help="Inspect and change the local GPROF-NN configuration.")
def config():
    pass


config.command(name="file", help="Show location of configuration file.")(conf.file)
config.command(name="show", help="Show current configuration.")(conf.show_config)
config.command(name="set", help="Modify the configuration.")(conf.set_config)

######################################################################
# gprof_nn download
######################################################################


@gprof_nn.group(help="Download retrieval models and test data.")
def download():
    pass


@download.command(name="model")
@click.argument("sensor", type=str)
def download_model(sensor: str):
    """
    Download GPROF-NN retrieval model for SENSOR. Use SENSOR=all to download the models for all sensors.
    """
    if sensor.lower() == "all":
        sensor_list = sensors.all_sensors()
    else:
        sensor_list = [sensors.get_sensor(sensor)]
    for sensor in sensor_list:
        dl.download_model(sensor)


@download.command(name="test_data")
@click.argument("sensor", type=str)
@click.argument("kind", type=str)
def download_test_data(sensor: str, kind: str):
    """
    Download GPROF-NN retrieval test data of type KIND for SENSOR.
    """
    dl.download_test_file(sensor, kind)


######################################################################
# gprof_nn update
######################################################################

gprof_nn.command(name="update")(dl.update_models)


######################################################################
# gprof_nn training
######################################################################


@gprof_nn.group(name="training", help="Train GPROF-NN retrieval models.")
def train():
    pass


train.command(name="init")(training.init_cli)
train.command(name="run")(training.run_cli)

######################################################################
# gprof_nn test
######################################################################

gprof_nn.command(name="test")(testing.cli)

######################################################################
# gprof_nn retrieve
######################################################################

gprof_nn.command(name="retrieve")(retrieval.cli)
