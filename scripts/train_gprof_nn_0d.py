"""
Training script for the GPROF-NN-0D retrieval.
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

from gprof_nn.data.training_data import GPROF0DDataset
from gprof_nn.models import GPROF_NN_0D_QRNN, GPROF_NN_0D_DRNN

###############################################################################
# Command line arguments.
###############################################################################

parser = argparse.ArgumentParser(
        description='Training script for the GPROF-NN-0D algorithm.')


# Input and output data
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
parser.add_argument('--n_layers_body',
                    metavar='n_layers',
                    type=int,
                    nargs=1,
                    default=4,
                    help='How many layers in the body of network.')
parser.add_argument('--n_neurons_body',
                    metavar='n_neurons_body',
                    type=int,
                    nargs=1,
                    default=256,
                    help='How many neurons in each hidden layer of the body.')
parser.add_argument('--n_layers_head',
                    metavar='n_layers',
                    type=int,
                    nargs=1,
                    default=2,
                    help= ('How many layers in each head of the network.'))
parser.add_argument('--n_neurons_head',
                    metavar='n_neurons_body',
                    type=int,
                    nargs=1,
                    default=128,
                    help='How many neurons in each head.')
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

training_data = args.training_data[0]
validation_data = args.validation_data[0]

n_layers_body = args.n_layers_body[0]
n_neurons_body = args.n_neurons_body[0]
n_layers_head = args.n_layers_head[0]
n_neurons_head = args.n_neurons_head[0]

print(n_layers_body, n_neurons_body, n_layers_head, n_neurons_head)
activation = args.activation[0]
residuals = args.residuals[0]

device = args.device[0]
targets = args.targets
network_type = args.type[0]
batch_size = args.batch_size[0]

#
# Prepare output
#

model_path = Path(args.model_path[0])
model_path.mkdir(parents=False, exist_ok=True)
network_name = (f"gprof_nn_0d_gmi_{network_type}_{n_layers_body}_"
                f"{n_neurons_body}_{n_layers_head}_{n_neurons_head}"
                f"_{activation}_{residuals}.pckl")

#
# Load the data.
#

dataset_factory = GPROF0DDataset
normalizer = Normalizer.load("../data/normalizer_gprof_0d_gmi.pckl")
kwargs = {
    "batch_size": batch_size,
    "normalizer": normalizer,
    "target": targets,
    "augment": False
}
training_data = DataFolder(
    training_data,
    dataset_factory,
    kwargs=kwargs,
    aggregate=2,
    n_workers=4)

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

#
# Create neural network model
#

if network_type == "drnn":
    xrnn = GPROF_NN_0D_DRNN(n_layers_body,
                            n_neurons_body,
                            n_layers_head,
                            n_neurons_head,
                            activation=activation,
                            residuals=residuals,
                            targets=targets)
elif network_type == "qrnn_exp":
    transformation = {t: LogLinear() for t in targets}
    transformation["latent_heat"] = None
    xrnn = GPROF_NN_0D_QRNN(n_layers_body,
                            n_neurons_body,
                            n_layers_head,
                            n_neurons_head,
                            activation=activation,
                            residuals=residuals,
                            transformation=transformation,
                            targets=targets)
else:
    xrnn = GPROF_NN_0D_QRNN(n_layers_body,
                            n_neurons_body,
                            n_layers_head,
                            n_neurons_head,
                            activation=activation,
                            residuals=residuals,
                            targets=targets)
model = xrnn.model

#
# Run training
#

n_epochs = 60
logger = TensorBoardLogger(n_epochs)
logger.set_attributes({
    "n_layers_body": n_layers_body,
    "n_neurons_body": n_neurons_body,
    "n_layers_head": n_layers_head,
    "n_neurons_head": n_neurons_head,
    "activation": activation,
    "residuals": residuals,
    "targets": ", ".join(targets),
    "type": network_type,
    "optimizer": "adam"
    })

metrics = ["MeanSquaredError", "Bias", "CalibrationPlot", "CRPS"]
scatter_plot = ScatterPlot(log_scale=True)
metrics.append(scatter_plot)

n_epochs = 10
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
