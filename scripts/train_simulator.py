"""
Training script to train a simulator that produces full-swath simulated Tbs
and biases.
"""
import argparse
from pathlib import Path

from torch import nn
from torch import optim
import numpy as np
from quantnn.data import DataFolder
from quantnn.normalizer import Normalizer
from quantnn.models.pytorch.logging import TensorBoardLogger
from quantnn.metrics import ScatterPlot

from gprof_nn import sensors
from gprof_nn.data.training_data import SimulatorDataset
from gprof_nn.models import Simulator

###############################################################################
# Command line arguments.
###############################################################################

parser = argparse.ArgumentParser(
    description='Training script for simulator networks.'
)

# Input and output data
parser.add_argument('sensor',
                    metavar='sensor',
                    type=str,
                    help='The sensor for which to run the training.')
parser.add_argument('training_data',
                    metavar='training_data',
                    type=str,
                    nargs=1,
                    help='Path to training data.')
parser.add_argument('validation_data',
                    metavar='validation_data',
                    type=str,
                    nargs=1,
                    help='Path to validation data.')
parser.add_argument('model_path',
                    metavar='model_path',
                    type=str,
                    nargs=1,
                    help='Where to store the model.')

# Model configuration
parser.add_argument('--n_features_body',
                    metavar='n_features',
                    type=int,
                    nargs=1,
                    default=4,
                    help='How many layers in the body of network.')
parser.add_argument('--n_layers_head',
                    metavar='n_layers',
                    type=int,
                    nargs=1,
                    default=2,
                    help= ('How many layers in each head of the network.'))
parser.add_argument('--n_features_head',
                    metavar='n_features_body',
                    type=int,
                    nargs=1,
                    default=128,
                    help='How many neurons in each head.')

# Other
parser.add_argument('--device', metavar="device", type=str, nargs=1,
                    help="The name of the device on which to run the training")
parser.add_argument('--batch_size', metavar="n", type=int, nargs=1,
                    help="The batch size to use for training.")

args = parser.parse_args()

sensor = args.sensor
training_data = args.training_data[0]
validation_data = args.validation_data[0]

n_features_body = args.n_features_body[0]
n_layers_head = args.n_layers_head[0]
n_features_head = args.n_features_head[0]

device = args.device[0]
batch_size = args.batch_size[0]


#
# Determine sensor
#

sensor = sensor.strip().upper()
sensor = getattr(sensors, sensor, None)
if sensor is None:
    raise Exception(
        f"Could not find requested sensor '{args.sensor}'"
    )

#
# Prepare output
#

model_path = Path(args.model_path[0])
model_path.mkdir(parents=False, exist_ok=True)
network_name = (f"simulator_{sensor.name.lower()}.pckl")

#
# Load the data.
#

dataset_factory = SimulatorDataset
normalizer = Normalizer.load(f"../data/normalizer_gmi.pckl")
kwargs = {
    "batch_size": batch_size,
    "normalizer": normalizer,
}

training_data = DataFolder(
    training_data,
    dataset_factory,
    kwargs=kwargs,
    queue_size=16,
    n_workers=4)

kwargs = {
    "batch_size": 1 * batch_size,
    "normalizer": normalizer,
}
validation_data = DataFolder(
    validation_data,
    dataset_factory,
    kwargs=kwargs,
    n_workers=2
)

#
# Create neural network model
#

simulator = Simulator(sensor,
                      n_features_body,
                      n_layers_head,
                      n_features_head)
simulator.normalizer = normalizer
model = simulator.model

#
# Run training
#

n_epochs = 120
logger = TensorBoardLogger(n_epochs)
logger.set_attributes({
    "n_features_body": n_features_body,
    "n_layers_head": n_layers_head,
    "n_features_head": n_features_head,
    "type": "simulator",
    "optimizer": "adam"
    })
metrics = ["MeanSquaredError", "Bias", "CalibrationPlot", "CRPS"]
scatter_plot = ScatterPlot()
metrics.append(scatter_plot)

n_epochs = 10
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
simulator.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler, logger=logger,
           metrics=metrics,
           device=device,
           mask=-9999)
simulator.save(model_path / network_name)
n_epochs = 20
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
simulator.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
           logger=logger,
           metrics=metrics,
           device=device,
           mask=-9999)
simulator.save(model_path / network_name)
n_epochs = 30
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
simulator.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
           logger=logger,
           metrics=metrics,
           device=device,
           mask=-9999)
simulator.save(model_path / network_name)
n_epochs = 30
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
simulator.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
           logger=logger,
           metrics=metrics,
           device=device,
           mask=-9999)
simulator.save(model_path / network_name)
n_epochs = 30
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
simulator.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
           logger=logger,
           metrics=metrics,
           device=device,
           mask=-9999)
simulator.save(model_path / network_name)
