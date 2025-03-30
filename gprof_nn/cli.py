"""
gprof_nn.cli
============

This module implements the command line interfaces for the functionality
provided by the 'gprof_nn' package.
"""
import click
import gprof_nn.logging
import gprof_nn.config as conf
from gprof_nn import training
from gprof_nn import retrieval, testing

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
        combined
    )

    @gprof_nn.group(name="extract_training_data")
    def extract_training_data():
        pass

    extract_training_data.command(name="sim")(sim.cli)
    extract_training_data.command(name="pre")(pretraining.cli)
    extract_training_data.command(name="mrms")(mrms.cli)
    extract_training_data.command(name="era5")(era5.cli)
    extract_training_data.command(name="finetuning")(finetuning.cli)
    extract_training_data.command(name="cloudsat")(cloudsat.cli)
    extract_training_data.command(name="combined")(combined.cli)
except ImportError:
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
# gprof_nn training
######################################################################

@gprof_nn.group(name="training")
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
