"""
Training script for the GPROF-NN-2D retrieval.
"""
import argparse
from pathlib import Path

from torch import nn
from torch import optim
import numpy as np
from quantnn import QRNN
from quantnn.drnn import DRNN
from quantnn.data import DataFolder
from quantnn.normalizer import Normalizer
from quantnn.models.pytorch.logging import TensorBoardLogger
from quantnn.metrics import ScatterPlot
from quantnn.transformations import LogLinear

from gprof_nn.data.training_data import GPROF2DDataset
from gprof_nn import sensors
from gprof_nn.models import GPROF_NN_2D_QRNN, GPROF_NN_2D_DRNN


###############################################################################
# Command line arguments.
###############################################################################

parser = argparse.ArgumentParser(
        description='Training script for the GPROF-NN-0D algorithm.')


# Input and output data
parser.add_argument('sensor',
                    metavar='sensor',
                    type=str,
                    help='The sensor for which to train a retrieval network.')
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
parser.add_argument('--type',
                    metavar="network_type",
                    type=str,
                    nargs=1,
                    help="The type of network: drnn, qrnn or qrnn_exp")
parser.add_argument('--n_blocks',
                    metavar='n_blocks',
                    type=int,
                    nargs=1,
                    default=4,
                    help=('How many blocks in each downsampling stage of the'
                          'body of the network.'))
parser.add_argument('--n_features_body',
                    metavar='n_features_body',
                    type=int,
                    nargs=1,
                    default=256,
                    help='How many features in the body of the network.')
parser.add_argument('--n_layers_head',
                    metavar='n_layers',
                    type=int,
                    nargs=1,
                    default=2,
                    help= ('How many layers in each head of the network.'))
parser.add_argument('--n_features_head',
                    metavar='n_features_head',
                    type=int,
                    nargs=1,
                    default=128,
                    help='How many neurons/features in each head.')
parser.add_argument('--activation',
                    metavar='activation',
                    type=str,
                    nargs=1,
                    default="ReLU",
                    help='The activation function to apply after each hidden layer.')
parser.add_argument('--residuals',
                    metavar='residuals',
                    type=str,
                    nargs=1,
                    default='simple',
                    help='The type of residual connections to apply.')

# Other
parser.add_argument('--device', metavar="device", type=str, nargs=1,
                    help="The name of the device on which to run the training")
parser.add_argument('--targets', metavar="target_1 target_2", type=str, nargs="+",
                    help="The target on which to train the network")
parser.add_argument('--batch_size', metavar="n", type=int, nargs=1,
                    help="The batch size to use for training.")

args = parser.parse_args()

sensor = getattr(sensors, args.sensor, None)
if sensor is None:
    raise ValueError(f"The sensor {args.sensor} isn't currently supported.")

training_data = args.training_data[0]
validation_data = args.validation_data[0]

n_blocks = args.n_blocks[0]
n_features_body = args.n_features_body[0]
n_layers_head = args.n_layers_head[0]
n_features_head = args.n_features_head[0]

activation = args.activation[0]
residuals = args.residuals[0]

device = args.device[0]
targets = args.targets
network_type = args.type[0]
batch_size = args.batch_size[0]

###############################################################################
# Prepare in- and output.
###############################################################################

dataset_factory = GPROF2DDataset
normalizer = Normalizer.load("../data/normalizer_gprof_0d_gmi.pckl")
kwargs = {
    "batch_size": batch_size,
    "normalizer": normalizer,
    "target": targets,
    "augment": True
}
training_data = DataFolder(
    training_data,
    dataset_factory,
    kwargs=kwargs,
    n_workers=6)

kwargs = {
    "batch_size": 8 * batch_size,
    "normalizer": normalizer,
    "target": targets,
    "augment": False
}
validation_data = DataFolder(
    validation_data,
    dataset_factory,
    kwargs=kwargs,
    n_workers=2
)

model_path = Path(args.model_path[0])
model_path.mkdir(parents=False, exist_ok=True)
network_name = (f"gprof_nn_2d_gmi_{network_type}_{n_blocks}_{n_features_body}"
                "_{n_layers_head}_{n_features_head}.pt")

###############################################################################
# Prepare in- and output.
###############################################################################

#
# Create neural network model
#

if network_type == "drnn":
    xrnn = GPROF_NN_2D_DRNN(sensor,
                            n_blocks,
                            n_features_body,
                            n_layers_head,
                            n_features_head,
                            target=targets)
elif network_type == "qrnn_exp":
    transformation = {t: LogLinear() for t in targets}
    transformation["latent_heat"] = None
    xrnn = GPROF_NN_2D_QRNN(sensor,
                            n_blocks,
                            n_features_body,
                            n_layers_head,
                            n_features_head,
                            transformation=transformation,
                            targets=targets)
else:
    xrnn = GPROF_NN_2D_QRNN(sensor,
                            n_blocks,
                            n_features_body,
                            n_layers_head,
                            n_features_head,
                            targets=targets)
model = xrnn.model

###############################################################################
# Run the training.
###############################################################################

n_epochs = 80
logger = TensorBoardLogger(n_epochs)
logger.set_attributes({
    "n_blocks": n_blocks,
    "n_features_body": n_features_body,
    "n_layers_head": n_layers_head,
    "n_features_head": n_features_head,
    "targets": ", ".join(targets),
    "type": network_type,
    "optimizer": "adam"
    })

metrics = ["MeanSquaredError", "Bias", "CalibrationPlot", "CRPS"]
scatter_plot = ScatterPlot(log_scale=True)
metrics.append(scatter_plot)

n_epochs = 20
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
xrnn.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
           logger=logger,
           metrics=metrics,
           device=device,
           mask=-9999)
xrnn.save(model_path / network_name)
n_epochs = 20
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
xrnn.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
           logger=logger,
           metrics=metrics,
           device=device,
           mask=-9999)
xrnn.save(model_path / network_name)
n_epochs = 20
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
xrnn.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
           logger=logger,
           metrics=metrics,
           device=device,
           mask=-9999)
xrnn.save(model_path / network_name)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
xrnn.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
           logger=logger,
           metrics=metrics,
           device=device,
           mask=-9999)
xrnn.save(model_path / network_name)
