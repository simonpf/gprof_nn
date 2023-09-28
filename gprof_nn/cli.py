"""
gprof_nn.cli
============

This module implements the command line interfaces for the functionality
provided by the 'gprof_nn' package.
"""
import click
import gprof_nn.config as conf
from gprof_nn.data import sim, pretraining, mrms

@click.group()
def gprof_nn():
    pass

@gprof_nn.group(name="extract_training_data")
def extract_training_data():
    pass

extract_training_data.command(name="sim")(sim.cli)
extract_training_data.command(name="pre")(pretraining.cli)
extract_training_data.command(name="mrms")(mrms.cli)

######################################################################
# gprof_nn config
######################################################################

@gprof_nn.group()
def config():
    pass

config.command(name="file")(conf.file)
config.command(name="show")(conf.show_config)
config.command(name="set")(conf.set_config)
