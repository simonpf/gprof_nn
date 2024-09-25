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
from gprof_nn.data import sim, pretraining, mrms, era5, finetuning

@click.group()
def gprof_nn():
    pass

@gprof_nn.group(name="extract_training_data")
def extract_training_data():
    pass

extract_training_data.command(name="sim")(sim.cli)
extract_training_data.command(name="pre")(pretraining.cli)
extract_training_data.command(name="mrms")(mrms.cli)
extract_training_data.command(name="era5")(era5.cli)
extract_training_data.command(name="finetuning")(finetuning.cli)

######################################################################
# gprof_nn config
######################################################################

@gprof_nn.group()
def config():
    pass

config.command(name="file")(conf.file)
config.command(name="show")(conf.show_config)
config.command(name="set")(conf.set_config)


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
