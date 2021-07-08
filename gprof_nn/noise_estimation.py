"""
=========================
gprof_nn.noise_estimation
=========================

This module implements all required functionality for the adversarial
noise estimation that is used to estimate the uncertainty affecting
the simulated observations used during training of the GPROF-NN algorithm.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm
from torch.utils.tensorboard import SummaryWriter
from quantnn.normalizer import MinMaxNormalizer
import xarray as xr

from gprof_nn.data.training_data import Dataset0DBase

###############################################################################
# PyTorch modules.
###############################################################################


class Discriminator(nn.Sequential):
    """
    Simple discriminator using only observations.

    Discriminator is regularized using spectral norm to improve stability
    of adversarial training.
    """
    def __init__(self,
                 n_inputs,
                 n_layers,
                 n_neurons):
        """
        Args:
            n_inputs: The number of input features.
            n_layers: The number of hidden layers of the discriminator.
            n_neurons: The number of neurons in each hidden layer.
        """

        layers = []
        n_in = n_inputs
        for i in range(n_layers - 1):
            layers += [spectral_norm(nn.Linear(n_in, n_neurons))]
            layers += [nn.GELU()]
            n_in = n_neurons

        layers += [spectral_norm(nn.Linear(n_neurons, 1))]
        layers += [nn.Sigmoid()]

        super().__init__(*layers)

    def forward(self, x, y):
        """
        Discards x and forwards y through discriminator.
        """
        return nn.Sequential.forward(self, y)


class ConditionalDiscriminator(nn.Sequential):
    """
    Conditional discriminator for 0D input data normalized using spectral
    norm.

    Discriminator is regularized using spectral norm to improve stability
    of adversarial training.
    """
    def __init__(self,
                 n_inputs_x,
                 n_inputs_y,
                 n_layers,
                 n_neurons):
        """
        Args:
            n_inputs_x: The number of ancillary data values.
            n_inputs_y: The number of observations.
            n_layers: The number of hidden layers of the discriminator.
            n_neurons: The number of neurons in each hidden layer.
        """

        layers = []
        n_in = n_inputs_x + n_inputs_y
        for i in range(n_layers - 1):
            layers += [spectral_norm(nn.Linear(n_in, n_neurons))]
            layers += [nn.GELU()]
            n_in = n_neurons

        layers += [spectral_norm(nn.Linear(n_neurons, 1))]
        layers += [nn.Sigmoid()]

        super().__init__(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        return nn.Sequential.forward(self, x)


class GaussianNoiseGenerator(nn.Module):
    """
    Simple gaussian noise generator using global bias and std. dev. values.
    """
    def __init__(self,
                 n_features):
        """
        Args:
            n_features: The number of observations (channels).
        """
        super().__init__()
        self.n_features = n_features

        self.mean = torch.nn.Parameter(torch.zeros(1, n_features))
        self.sigma = torch.nn.Parameter(torch.zeros(1, n_features))

    def forward(self, x, y):
        z = torch.randn(y.shape[0], self.n_features, device=y.device)
        z = z * torch.exp(self.sigma) + self.mean
        return y + z

class CategoricalGaussianNoiseGenerator(nn.Module):
    """
    Gaussian noise generator using bias and std. dev. values conditioned on
    surface type.
    """
    def __init__(self,
                 n_features,
                 n_classes):
        """
        Args:
            n_features:
            n_classes:
        """
        super().__init__()
        self.n_features = n_features

        self.mean = torch.nn.Parameter(torch.zeros(1, n_classes, n_features))
        self.sigma = torch.nn.Parameter(-1.0 * torch.ones(1, n_classes, n_features))

    def forward(self, x, y):

        mask = torch.unsqueeze(x[:, 1:], 2) > 0
        means = torch.masked_select(self.mean.to(mask.device), mask)
        means = means.reshape(-1, self.n_features)
        sigmas = torch.masked_select(self.sigma.to(mask.device), mask)
        sigmas = sigmas.reshape(-1, self.n_features)

        z = torch.randn(y.shape[0], self.n_features, device=y.device)
        z = z * torch.exp(sigmas) + means
        return y + z


class ConditionalGaussianNoiseGenerator(nn.Module):
    """
    Discriminator for 0D input data normalized using spectral
    norm.
    """
    def __init__(self,
                 n_features):
        """
        Args:
            n_inputs: The number of input features.
            n_layers: The number of hidden layers of the discriminator.
            n_neurons: The number of neurons in each hidden layer.
        """
        super().__init__()
        self.n_features = n_features

        self.model = nn.Sequential(*[
            nn.Linear(24, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 2 * n_features)
        ])

    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        z = self.model(x)
        means = z[:, :self.n_features]
        sigmas = z[:, self.n_features:]
        z = torch.randn(y.shape[0], self.n_features, device=y.device)
        z = z * torch.exp(sigmas) + means
        return y + z


class Generator(nn.Module):
    """
    CGAN-type parameter-free generator.
    """
    def __init__(self,
                 n_features):
        """
        Args:
            n_inputs: The number of input features.
            n_layers: The number of hidden layers of the discriminator.
            n_neurons: The number of neurons in each hidden layer.
        """
        super().__init__()
        self.n_features = n_features

        self.model = nn.Sequential(*[
            nn.Linear(24 + n_features, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, n_features)
        ])

    def forward(self, x, y):
        z = torch.randn(y.shape[0], self.n_features, device=y.device)
        x = torch.cat([x, y, z], 1)
        d_y = self.model(x)
        return y + d_y

FAKE = 0
REAL = 1

def loss(y, target):
    """
    GAN loss function.
    """
    if target == FAKE:
        return nn.functional.binary_cross_entropy(y, torch.zeros_like(y))
    return nn.functional.binary_cross_entropy(y, torch.ones_like(y))



def train_on_batch(x_source,
                   y_source,
                   generator,
                   optimizer_gen,
                   x_target,
                   y_target,
                   discriminator,
                   optimizer_disc,
                   iter,
                   iter_gen=1):
    """
    Train given generator and discriminator for a given batch of
    data.
    Based on https://beckham.nz/2021/06/28/training-gans.html.

    Args:
        x: Ancillary input, provided to both generator and discriminator.
        y_source: The source observations which should be fitted to
            target observations.
        generator: The generator module.
        optimizer_gen: The optimizer of the generator.
        y_target: The target observations.
        discriminator: The discriminator module.
        optimizer_disc: The optimizer of the discriminator.

    Return:
        Tuple ``(loss_gen, loss_disc)`` containing the loss of the
        generator and discriminator respectively.
    """

    optimizer_gen.zero_grad()
    optimizer_disc.zero_grad()

    #
    # Train discriminator.
    #

    y_corr = generator(x_source, y_source)
    loss_disc = (loss(discriminator(x_source, y_corr.detach()), FAKE) +
                 loss(discriminator(x_target, y_target), REAL))
    loss_disc.backward()
    optimizer_disc.step()

    #
    # Train generator.
    #

    if iter % iter_gen == 0:
        optimizer_disc.zero_grad()
        loss_gen = loss(discriminator(x_source, y_corr), REAL)

        d_y = y_corr - y_source
        #loss_gen += 0.01 * torch.linalg.norm(d_y)

        loss_gen.backward()
        optimizer_gen.step()
    else:
        loss_gen = torch.zeros_like(loss_disc)

    return loss_gen.detach(), loss_disc.detach()


def evaluate(iteration,
             input_data,
             generator,
             target_data,
             discriminator,
             writer,
             device):
    """
    Evaluate training progress by logging histograms of observation
    distributions to tensorflow.

    Args:
        iteration: The iteration number.
        input_data: Iterable providing the source data.
        generator: The generator module.
        target_data: Iterable providing the target data.
        discriminator: The discriminator module
        writer: Pytorch tensorboard summary writer.
        device: String identifying the device on which to perform
            the evaulation.
    """

    with torch.no_grad():
        x_s = []
        y_s = []
        y_c = []
        x_t = []
        y_t = []
        y_d = []

        for j, (batch_s, batch_t) in enumerate(zip(input_data, target_data)):

            x_source, y_source = batch_s
            x_source = x_source.to(device)
            y_source = y_source.to(device)

            x_target, y_target = batch_t
            x_target = x_target.to(device)
            y_target = y_target.to(device)

            y_corrected = generator(x_source, y_source)
            y_delta = y_corrected - y_source

            x_s += [x_source.cpu().numpy()]
            y_s += [y_source.cpu().numpy()]
            y_c += [y_corrected.cpu().numpy()]
            x_t += [x_target.cpu().numpy()]
            y_t += [y_target.cpu().numpy()]
            y_d += [y_delta.cpu().numpy()]

        x_s = np.concatenate(x_s)
        y_s = np.concatenate(y_s)
        y_c = np.concatenate(y_c)
        x_t = np.concatenate(x_t)
        y_t = np.concatenate(y_t)
        y_d = np.concatenate(y_d)

        #
        # Generate plots
        #

        bins = np.linspace(-1.2, 1.2, 101)

        for surface_type in range(18):

            inds = x_t[:, 1 + surface_type] > 0
            y_target = y_t[inds]

            x = 0.5 * (bins[1:] + bins[:-1])
            inds = x_s[:, 1 + surface_type] > 0
            y_source = y_s[inds]
            y_corrected = y_c[inds]
            y_delta = y_d[inds]

            # Distribution
            f, ax = plt.subplots(1, 1, figsize=(8, 6))
            for j in range(5):
                y, _ = np.histogram(y_target[:, j], bins=bins, density=True)
                ax.fill_between(x, y, 0, label=f"Feature {j + 1}", facecolor=f"C{j}",
                                edgecolor=None, alpha=0.5)
                y, _ = np.histogram(y_source[:, j], bins=bins, density=True)
                ax.plot(x, y, c=f"C{j}", ls="--")
                y, _ = np.histogram(y_corrected[:, j], bins=bins, density=True)
                ax.plot(x, y, c=f"C{j}", ls="-")
            ax.set_xlabel("Feature value")
            ax.set_ylabel("Frequency")
            ax.legend()

            writer.add_figure(
                f"Feature distributions (surface type {surface_type + 1})",
                f,
                global_step=iteration
            )
            plt.close(f)

            f, ax = plt.subplots(1, 1, figsize=(8, 6))
            means = y_delta.mean(axis=0)
            sigmas = y_delta.std(axis=0)
            n = means.size

            x = np.arange(n)
            width = 0.3

            ax.bar(x - 0.5 * width, means, width=width, label="Bias")
            ax.bar(x + 0.5 * width, sigmas, width=width, label="Std. dev.")

            ax.set_xlabel("Feature")
            ax.legend()

            writer.add_figure(
                f"Bias and std. dev. (surface type {surface_type + 1})",
                f,
                global_step=iteration
            )
            plt.close(f)


def train(input_data,
          generator,
          optimizer_gen,
          target_data,
          discriminator,
          optimizer_disc,
          n_epochs,
          device="cpu",
          iter_gen=1,
          validation_data_source=None,
          validation_data_target=None):
    """
    Run adversarial training.

    Args:
        input_data: Data loader provided tuples ``(x, y_source)`` of ancillary
            input and corresponding source observations.
        generator: The generator module to train.
        optimizer_gen: The generator to use to train the generator.
        target_data: Data loader providing tuple ``(x, y_target)`` of ancillary
            input and corresponding target observations.
        discriminator: The discriminator module to train.
        optimizer_disc: The optimizer to use to train the discriminator.
        n_epochs: For how many numbers of epochs to train generator and
            discriminator.
        device: String defining the device on which to perform the training.
    """
    loss_generator = []
    loss_discriminator = []

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    writer = SummaryWriter()

    step = 0
    for i in range(n_epochs):

        sum_gen = 0.0
        sum_disc = 0.0

        l_g_p = 0.0

        for j, (batch_s, batch_t) in enumerate(zip(input_data, target_data)):

            x_source, y_source = batch_s
            x_source = x_source.to(device)
            y_source = y_source.to(device)

            x_target, y_target = batch_t
            x_target = x_target.to(device)
            y_target = y_target.to(device)

            l_g, l_d = train_on_batch(x_source,
                                      y_source,
                                      generator,
                                      optimizer_gen,
                                      x_target,
                                      y_target,
                                      discriminator,
                                      optimizer_disc,
                                      j,
                                      iter_gen)
            if l_g == 0.0:
                l_g = l_g_p
            l_g_p = l_g

            loss_generator.append(l_g.item())
            loss_discriminator.append(l_d.item())

            j = j + 1
            sum_gen += l_g
            sum_disc += l_d

            if step % 10 == 0:
                writer.add_scalars(
                    'GAN losses',
                    {
                        "Generator": l_g,
                        "Discriminator": 0.5 * l_d,
                    },
                    global_step=step
                )
            step += 1

        if ((validation_data_source is not None) and
            (validation_data_target is not None)):
            evaluate(i,
                     validation_data_source, generator,
                     validation_data_target, discriminator,
                     writer,
                     device)

    return (np.array(loss_generator),
            np.array(loss_discriminator))


class ObservationDataset0D(Dataset0DBase):
    """
    Dataset class to load target, i.e. real, observations to fit simulated
    observations to.
    """
    def __init__(self,
                 filename,
                 batch_size=512,
                 shuffle=True,
                 normalize=True,
                 normalizer_x=None,
                 normalizer_y=None,
                 sensor=None):
        """
        Args:
            filename: Filename of the NetCDF file containing the extracted
                obesrvations.
            batch_size: The size of the mini-batches.
            shuffle: Whether or not to shuffle the data each time the batch
                at index 0 is accessed.
            normalizer_x: Normalizer to use for ancillary data.
            normalizer_y: Normalizer to use for observation data.
        """
        super().__init__()
        self.transform_zeros = False

        self.filename = Path(filename)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalize = normalize
        self.normalizer_x = normalizer_x
        self.normalizer_y = normalizer_y

        self._load_data()

        # Normalize ancillary data and observations
        # independently.
        if self.normalize:
            if self.normalizer_x is None:
                indices_1h = list(range(1, 19))
                self.normalizer_x = MinMaxNormalizer(
                    self.x,
                    exclude_indices=indices_1h
                )
            self.x = self.normalizer_x(self.x)
            if self.normalizer_y is None:
                self.normalizer_y = MinMaxNormalizer(self.y)
            self.y = self.normalizer_y(self.y)

        self._shuffled = False
        if self.shuffle:
            self._shuffle()

    def __repr__(self):
        return f"ObservationDataset0D({self.filename.name}, n_batches={len(self)})"

    def __str__(self):
        return f"ObservationDataset0D({self.filename.name}, n_batches={len(self)})"

    def _load_data(self):
        """
        Loads the data from the file into the ``x`` and ``y`` attributes.
        """
        with xr.open_dataset(self.filename) as dataset:
            tbs = dataset["brightness_temperatures"].data
            valid = np.all((tbs > 0) * (tbs < 500), axis=-1)

            # Ancillary data
            vas = dataset["earth_incidence_angle"].data[valid]
            st = dataset["surface_type"].data[valid]
            n_types = 18
            st_1h = np.zeros((st.shape[0], n_types), dtype=np.float32)
            for i in range(n_types):
                mask = st == i + 1
                st_1h[mask, i] = 1.0

            self.x = np.concatenate([vas[:, np.newaxis], st_1h], axis=1)

            # Observations
            tbs = tbs[valid]
            self.y = tbs
