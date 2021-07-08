"""
Training script for noise estimators.
"""
import argparse
from pathlib import Path

import torch
from torch.optim import Adam
from quantnn.normalizer import Normalizer
from quantnn.data import DataFolder

from gprof_nn import sensors
from gprof_nn.data.training_data import TrainingObsDataset0D
from gprof_nn.noise_estimation import (ConditionalDiscriminator,
                                       CategoricalGaussianNoiseGenerator,
                                       ObservationDataset0D,
                                       train)

###############################################################################
# Command line arguments.
###############################################################################

parser = argparse.ArgumentParser(
        description='Training script for observation uncertainty estimation.'
)


# Input and output data
parser.add_argument('sensor',
                    metavar='sensor',
                    type=str,
                    help='The sensor for which to run the training.')
parser.add_argument('training_data_source',
                    metavar='training_data_source',
                    type=str,
                    nargs=1,
                    help='Path to simulated observations to use for training.')
parser.add_argument('training_data_target',
                    metavar='training_data_target',
                    type=str,
                    nargs=1,
                    help='Path to real observations to use for training.')
parser.add_argument('validation_data_source',
                    metavar='validation_data_source',
                    type=str,
                    nargs=1,
                    help='Path to simulated observations to use for validation.')
parser.add_argument('validation_data_target',
                    metavar='validation_data_source',
                    type=str,
                    nargs=1,
                    help='Path to real observations to use for validation.')
parser.add_argument('model_path',
                    metavar='model_path',
                    type=str,
                    nargs=1,
                    help='Where to store the model.')

# Other
parser.add_argument('--device', metavar="device", type=str, nargs=1,
                    help="The name of the device on which to run the training")
parser.add_argument('--batch_size', metavar="n", type=int, nargs=1,
                    help="The batch size to use for training.")

args = parser.parse_args()
sensor = args.sensor
training_data_source = args.training_data_source[0]
training_data_target = args.training_data_target[0]
validation_data_source = args.validation_data_source[0]
validation_data_target = args.validation_data_target[0]

model_path = Path(args.model_path[0])

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
network_name = (f"noise_generator_{sensor.name.lower()}.pt")

#
# Training and validation data
#

normalizer_x = Normalizer.load(
    f"../data/obs_normalizer_{sensor.name.lower()}_x.pckl"
)
normalizer_y = Normalizer.load(
    f"../data/obs_normalizer_{sensor.name.lower()}_y.pckl"
)

kwargs = {
    "sensor": sensor,
    "batch_size": batch_size,
    "normalizer_x": normalizer_x,
    "normalizer_y": normalizer_y,
}

training_data_source = DataFolder(
    training_data_source,
    TrainingObsDataset0D,
    kwargs=kwargs,
    queue_size=16,
    n_workers=4
)

training_data_target = DataFolder(
    training_data_target,
    ObservationDataset0D,
    kwargs=kwargs,
    queue_size=16,
    n_workers=4
)

kwargs = {
    "sensor": sensor,
    "batch_size": 4 * batch_size,
    "normalizer_x": normalizer_x,
    "normalizer_y": normalizer_y,
}

validation_data_source = DataFolder(
    validation_data_source,
    TrainingObsDataset0D,
    kwargs=kwargs,
    queue_size=16,
    n_workers=4
)

validation_data_target = DataFolder(
    validation_data_target,
    ObservationDataset0D,
    kwargs=kwargs,
    queue_size=16,
    n_workers=4
)

#
# Create neural network models
#

discriminator = ConditionalDiscriminator(19, 5, 6, 256)
generator = CategoricalGaussianNoiseGenerator(5, 18)
optimizer_gen = Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_disc = Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

loss_gen, loss_disc = train(training_data_source,
                            generator,
                            optimizer_gen,
                            training_data_target,
                            discriminator,
                            optimizer_disc,
                            100,
                            device="cuda:0",
                            iter_gen=10,
                            validation_data_source=validation_data_source,
                            validation_data_target=validation_data_target)
torch.save(generator, model_path / network_name)
