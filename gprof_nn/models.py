"""
===============
gprof_nn.models
===============

This module defines the neural network models that are used
for the implementation of the GPROF-NN algorithms.
"""
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn.functional import softplus
from quantnn.qrnn import QRNN
from quantnn.drnn import DRNN
from quantnn.mrnn import MRNN, Mean, Quantiles, Density
from quantnn.models.pytorch.xception import (SeparableConv3x3,
                                             SymmetricPadding)

from gprof_nn.definitions import ALL_TARGETS, PROFILE_NAMES
from gprof_nn.retrieval import (
    NetcdfLoader1D,
    NetcdfLoader3D,
    PreprocessorLoader1D,
    PreprocessorLoader3D,
    L1CLoader1D,
    L1CLoader3D,
    SimulatorLoader,
)
from gprof_nn.data.training_data import (GPROF_NN_1D_Dataset,
                                         GPROF_NN_3D_Dataset,
                                         _INPUT_DIMENSIONS)


# Define bins for DRNN models.
BINS = {
    "surface_precip": np.logspace(-5.0, 2.5, 129),
    "convective_precip": np.logspace(-5.0, 2.5, 129),
    "rain_water_path": np.logspace(-5.0, 2.5, 129),
    "ice_water_path": np.logspace(-5.0, 2.5, 129),
    "cloud_water_path": np.logspace(-5.0, 2.5, 129),
    "total_column_water_vapor": np.logspace(-4.5, 2.5, 129),
    "cloud_water_content": np.logspace(-6.0, 1.5, 129),
    "snow_water_content": np.logspace(-6.0, 1.5, 129),
    "rain_water_content": np.logspace(-6.0, 1.5, 129),
    "latent_heat": np.concatenate(
        [-np.logspace(-2, 2.5, 64)[::-1], np.array([0.0]), np.logspace(-2, 3.0, 64)]
    ),
}

for k in BINS:
    if k != "latent_heat":
        BINS[k][0] == 0.0

# Quantiles for scalar variables
QUANTILES = np.linspace(0.0, 1.0, 66)[1:-1]
# Quantiles for profile variables
PROFILE_QUANTILES = np.linspace(0.0, 1.0, 18)[1:-1]
# Define types of residuals for MLP models.
RESIDUALS = ["none", "simple", "hyper"]


###############################################################################
# GPROF-NN 1D
###############################################################################


class MLP(nn.Module):
    """
    Pytorch 'Module' implementating a fully-connected feed-forward
    neural network.
    """

    def __init__(
        self,
        n_inputs,
        n_neurons,
        n_outputs,
        n_layers,
        activation="ReLU",
        internal=False,
    ):
        """
        Create MLP object.

        Args:
            n_inputs: Number of features in input.
            n_outputs: Number of output values.
            n_neurons: Number of neurons in hidden layers.
            n_layers: Number of layers including the output layer.
            activation: The activation function to use in each layer.
            internal: Whether or not activation layer norm and activation
                should be applied to output from last layer.
        """
        super().__init__()
        self.n_inputs = n_inputs
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.layers = nn.ModuleList()
        self.internal = internal
        self.activation = getattr(nn, activation)
        for i in range(n_layers - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(n_inputs, n_neurons, bias=False),
                    nn.LayerNorm(n_neurons),
                    self.activation(),
                )
            )
            n_inputs = n_neurons
        if n_layers > 0:
            if self.internal:
                self.output_layer = nn.Sequential(
                    nn.Linear(n_inputs, n_neurons, bias=False),
                    nn.LayerNorm(n_neurons),
                    self.activation(),
                )
            else:
                self.output_layer = nn.Linear(n_inputs, n_outputs, bias=False)

    def __repr__(self):
        try:
            return (
                f"MLP(n_inputs={self.n_inputs}, n_layers={self.n_layers}, "
                f"n_neurons={self.n_neurons})"
            )
        except:
            return super().__repr__()

    def forward(self, x, *args, **kwargs):
        """
        Forward input through network.

        Args:
            x: The 2D input tensor to propagate through the network.

        Return:
            Tuple ``(y, None)`` consisting of the network output ``y``
            and ``None``.
        """
        if self.n_layers == 0:
            return x, None

        for l in self.layers:
            y = l(x)
            x = y

        y = self.output_layer(x)
        return y, None


class ResidualMLP(MLP):
    """
    Fully-connected feed-forward neural network with residual
    connections between adjacent layers.
    """

    def __init__(
        self,
        n_inputs,
        n_neurons,
        n_outputs,
        n_layers,
        activation="ReLU",
        internal=False,
    ):
        """
        Create MLP with residual connection.

        Args:
            n_inputs: Number of input features in first layer.
            n_outputs: Number of output values.
            n_neurons: Number of neurons in hidden layers.
            n_layers: Number of layers including the output layer.
            activation: The activation function to use in each layer.
            internal: Whether or not activation layer norm and activation
                are applied to output from last layer.
        """
        super().__init__(
            n_inputs,
            n_neurons,
            n_outputs,
            n_layers,
            activation=activation,
            internal=internal,
        )
        if n_inputs != n_neurons:
            self.projection = nn.Linear(n_inputs, n_neurons)
        else:
            self.projection = None

    def __repr__(self):
        try:
            return (
                f"ResidualMLP(n_inputs={self.n_inputs}, n_layers={self.n_layers}, "
                f"n_neurons={self.n_neurons})"
            )
        except:
            return super().__repr__()

    def forward(self, x, *args, **kwargs):
        """
        Forward input through network.

        Args:
            x: The 2D input tensor to propagate through the network.

        Return:
            Tuple ``(y, None)`` consisting of the network output ``y``
            and ``None``.
        """
        if self.n_layers == 0:
            return x, None

        if self.layers:
            l = self.layers[0]
            y = l(x)
            if self.projection is not None:
                y = y + self.projection(x)
            else:
                y = y + x
            x = y

        for l in self.layers[1:]:
            y = l(x)
            y += x
            x = y

        y = self.output_layer(x)
        if self.internal:
            y += x
        return y, None


class HyperResidualMLP(ResidualMLP):
    """
    Fully-connected feed-forward neural network with residual
    connections between adjacent layers and hyper-residual
    connections between all layers.
    """

    def __init__(
        self,
        n_inputs,
        n_neurons,
        n_outputs,
        n_layers,
        activation="ReLU",
        internal=False,
    ):
        """
        Create network object.

        Args:
            n_inputs: Number of features in the input vector.
            n_outputs: Number of output values.
            n_neurons: Number of neurons in hidden layers.
            n_layers: Number of layers including the output layer.
            output_activation: Layer class to use as output activation.
            internal: If set to true, residuals will be applied to output
                from last layer.
        """
        super().__init__(
            n_inputs,
            n_neurons,
            n_outputs,
            n_layers,
            activation=activation,
            internal=internal,
        )

    def __repr__(self):
        try:
            return (
                f"HyperResidualMLP(n_inputs={self.n_inputs}, n_layers={self.n_layers}, "
                f"n_neurons={self.n_neurons})"
            )
        except:
            return super().__repr__()

    def forward(self, x, acc_in=None, li=1):
        """
        Forward input through network.

        Args:
            x: The 2D input tensor to propagate through the network.
            acc_in: Accumulator tensor containing accumulated activations
                 from previous layers.
            li: Layer index used to normalize accumulated activations.

        Return:
            Tuple ``(y, acc_out)`` consisting of the network output ``y``
            and and the accumulator tensor ``acc_out``.
        """
        if self.n_layers == 0:
            return x, None

        acc = torch.zeros((x.shape[0], self.n_neurons), dtype=x.dtype, device=x.device)
        if acc_in is not None:
            if self.projection is not None:
                acc += self.projection(acc_in)
            else:
                acc += acc_in

        if self.layers:
            l = self.layers[0]
            y = l(x)
            if self.projection is not None:
                x_p = self.projection(x)
            else:
                x_p = x
            y = y + x_p

            if li > 1:
                y += (1.0 / (li - 1)) * acc
            acc += x_p
            li += 1
            x = y

        for l in self.layers[1:]:
            y = l(x)
            y += x
            if li > 1:
                y += (1.0 / (li - 1)) * acc
            acc += x
            li += 1
            x = y

        y = self.output_layer(x)
        if self.internal:
            y += x
            if li > 1:
                y += (1.0 / (li - 1)) * acc
            acc += x
        return y, acc


class MultiHeadMLP(nn.Module):
    """
    Pytorch neural network model for the GPROF-NN 1D retrieval.

    The model is a fully-connected residual network model with a separate
    head for each retrieval target.

    Attributes:
        n_layers_body: The number of layers in the body of the network.
        n_neurons_body: The number of neurons in the hidden layers of the
            body of the network.
        n_layers_head: The number of layers in each head of the network.
        n_neurons_head: The number of neurons in the hidden layers of the
            head of the network.
        n_outputs: How many quantiles to predict for each retrieval target.
        target: Single string or list containing the retrieval targets
              to predict.
    """

    def __init__(
        self,
        n_inputs,
        n_layers_body,
        n_neurons_body,
        n_layers_head,
        n_neurons_head,
        n_outputs,
        residuals="none",
        targets=None,
        activation="ReLU",
        ancillary=True,
    ):
        if targets is None:
            self.targets = ["surface_precip"]
        else:
            self.targets = targets
        self.n_inputs = n_inputs
        self.n_layers_body = n_layers_body
        self.n_neurons_body = n_neurons_body
        self.n_layers_head = n_layers_head
        self.n_neurons_head = n_neurons_head

        residuals = residuals.lower()
        if residuals not in RESIDUALS:
            raise ValueError(f"'residuals' argument should be one of f{RESIDUALS}.")
        if residuals == "none":
            module_class = MLP
        elif residuals == "hyper":
            module_class = HyperResidualMLP
        else:
            module_class = ResidualMLP

        self.ancillary = ancillary
        if not ancillary:
            n_inputs = n_inputs - 24

        super().__init__()
        self.body = module_class(
            n_inputs,
            n_neurons_body,
            n_neurons_body,
            n_layers_body,
            activation=activation,
            internal=True,
        )
        self.n_inputs_body = n_inputs

        targets = self.targets
        if n_layers_body > 0:
            n_in = n_neurons_body
        else:
            n_in = n_inputs

        self.heads = nn.ModuleDict()
        for t in targets:
            if t in PROFILE_NAMES:
                self.heads[t] = module_class(
                    n_in, n_neurons_head, 16 * 28, n_layers_head, activation=activation
                )
            else:
                self.heads[t] = module_class(
                    n_in,
                    n_neurons_head,
                    n_outputs,
                    n_layers_head,
                    activation=activation,
                )

    def __repr__(self):
        try:
            return (
                f"MultiHeadMLP(n_inputs={self.n_inputs}, n_layers_body={self.n_layers_body}, "
                f", n_neurons_body={self.n_neurons_body}, n_layers_head={self.n_layers_head}, "
                f" n_neurons_head={self.n_neurons_head})"
            )
        except:
            return super().__repr__()

    def forward(self, x):
        """
        Forward the input x through the network.

        Args:
             x: Rank-2 tensor with the 39 input elements along the
                 second dimension and the batch sample along the first.

        Return:
            In the case of a single-target network a single tensor. In
            the case of a multi-target network a dictionary of tensors.
        """
        targets = self.targets
        if not self.ancillary:
            x = x[..., : self.n_inputs_body]

        y, acc = self.body(x, None)
        results = {}
        profile_shape = (x.shape[0], 16, 28)
        for k in targets:
            results[k], _ = self.heads[k](y, acc, self.n_layers_body + 1)
            if k in PROFILE_NAMES:
                y_k = results[k]
                results[k] = y_k.reshape(profile_shape)
        return results


class GPROF_NN_1D_QRNN(MRNN):
    """
    Neural network for the GPROF-NN 1D algorithm based on 'quantnn's
    quantile regression neural networks (QRNN).

    Attributes:
        sensor: The sensor that the network is compatible with.
        targets: The retrieval targets that the network can handle.
        preprocessor_class: Interface class that reads CSU preprocessor
            and transforms their content into the input format expected
            by the network.
        netcdf_class: Interface class that reads NetCDF training data
            and transforms their content into the input format expected
            by the network.
        normalizer: Normalizer object to use to normalize the network
            inputs. Note that this will be 'None' before the network
            has been trained.
        configuration: The configuration ('GANAL' or 'ERA5' for which
            the network was trained. Note that this will be 'None' before
            the network has been trained.
    """

    def __init__(
        self,
        sensor,
        n_layers_body,
        n_neurons_body,
        n_layers_head,
        n_neurons_head,
        activation="ReLU",
        residuals="simple",
        targets=None,
        transformation=None,
        ancillary=True,
    ):
        self.sensor = sensor
        residuals = residuals.lower()
        if residuals not in RESIDUALS:
            raise ValueError(f"'residuals' argument should be one of {RESIDUALS}.")

        if targets is None:
            targets = ALL_TARGETS
        self.targets = targets

        if transformation is not None and not isinstance(transformation, dict):
            if type(transformation) is type:
                transformation = {t: transformation() for t in targets}
            else:
                transformation = {t: transformation for t in targets}
            if "latent_heat" in targets:
                transformation["latent_heat"] = None

        model = MultiHeadMLP(
            sensor.n_inputs,
            n_layers_body,
            n_neurons_body,
            n_layers_head,
            n_neurons_head,
            64,
            targets=targets,
            residuals=residuals,
            activation=activation,
            ancillary=ancillary,
        )

        losses = {}
        for target in targets:
            if target in PROFILE_NAMES:
                losses[target] = Quantiles(PROFILE_QUANTILES)
            else:
                losses[target] = Quantiles(QUANTILES)

        super().__init__(
            n_inputs=sensor.n_inputs,
            losses=losses,
            model=model,
            transformation=transformation,
        )

        if ancillary:
            self.preprocessor_class = PreprocessorLoader1D
        else:
            self.preprocessor_class = L1CLoader1D
        self.netcdf_class = NetcdfLoader1D

        # Initialize attributes that will be set during training.
        self.normalizer = None
        self.configuration = None

    @property
    def suffix(self):
        return "1D"

    def __repr__(self):
        return (f"GPROF_NN_1D_QRNN(targets={self.targets})")
        trained = getattr(self, "configuration", None) is not None
        if trained:
            return (f"GPROF_NN_1D_QRNN(sensor={self.sensor}, "
                    f"configuration={self.configuration}, "
                    f"targets={self.targets})")
        else:
            return (f"GPROF_NN_1D_QRNN(targets={self.targets})")

    def set_targets(self, targets):
        """
        Set targets for retrival.

        This function can be used to reduce the number of retrieved
        targets during inference to speed up the retrieval.

        Args:
            targets: List of the targets to retrieve. Note: These must
                be a subset of the targets the network was trained to
                to retriev.
        """
        if not all([t in self.targets for t in targets]):
            raise ValueError("'targets' must be a sub-set of the models targets.")
        self.targets = targets
        self.model.target = targets


class GPROF_NN_1D_DRNN(MRNN):
    """
    Neural network for the GPROF-NN 1D algorithm based on 'quantnn's
    density regression neural networks (DRNN).

    Attributes:
        sensor: The sensor that the network is compatible with.
        targets: The retrieval targets that the network can handle.
        preprocessor_class: Interface class that reads CSU preprocessor
            and transforms their content into the input format expected
            by the network.
        netcdf_class: Interface class that reads NetCDF training data
            and transforms their content into the input format expected
            by the network.
        normalizer: Normalizer object to use to normalize the network
            inputs. Note that this will be 'None' before the network
            has been trained.
        configuration: The configuration ('GANAL' or 'ERA5' for which
            the network was trained. Note that this will be 'None' before
            the network has been trained.
    """
    def __init__(
        self,
        sensor,
        n_layers_body,
        n_neurons_body,
        n_layers_head,
        n_neurons_head,
        activation="ReLU",
        residuals="simple",
        targets=None,
        ancillary=True,
    ):
        self.sensor = sensor
        residuals = residuals.lower()
        if residuals not in RESIDUALS:
            raise ValueError(f"'residuals' argument should be one of {RESIDUALS}.")

        if targets is None:
            targets = ALL_TARGETS
        self.targets = targets

        model = MultiHeadMLP(
            sensor.n_inputs,
            n_layers_body,
            n_neurons_body,
            n_layers_head,
            n_neurons_head,
            128,
            targets=targets,
            residuals=residuals,
            activation=activation,
            ancillary=ancillary,
        )

        losses = {}
        for target in targets:
            if target in PROFILE_NAMES:
                losses[target] = Mean()
            else:
                losses[target] = Density(bins=BINS[target])

        super().__init__(n_inputs=sensor.n_inputs, losses=losses, model=model)

        if ancillary:
            self.preprocessor_class = PreprocessorLoader1D
        else:
            self.preprocessor_class = L1CLoader1D
        self.netcdf_class = NetcdfLoader1D

        # Initialize attributes that will be set during training.
        self.normalizer = None
        self.configuration = None

    @property
    def suffix(self):
        return "1D"

    def __repr__(self):
        trained = getattr(self, "configuration", None) is not None
        if trained:
            return (f"GPROF_NN_1D_DRNN(sensor={self.sensor}, "
                    f"configuration={self.configuration}, "
                    f"targets={self.targets})")
        else:
            return (f"GPROF_NN_1D_DRNN(sensor={self.sensor}, "
                    f"targets={self.targets})")

    def set_targets(self, targets):
        """
        Set targets for retrival.

        This function can be used to reduce the number of retrieved
        targets during inference to speed up the retrieval.

        Args:
            targets: List of the targets to retrieve. Note: These must
                be a subset of the targets the network was trained to
                to retriev.
        """
        if not all([t in self.targets for t in targets]):
            raise ValueError("'targets' must be a sub-set of the models targets.")
        self.targets = targets
        self.model.targets = targets


GPROF_NN_0D_QRNN = GPROF_NN_1D_QRNN
GPROF_NN_0D_DRNN = GPROF_NN_1D_DRNN


def adapt_normalizer(gmi_normalizer, sensor):
    """
    Adapt GMI normalizer to another sensor.

    Args:
        gmi_normalizer: Normalizer used to train GMI model.
        sensor: The sensor to which to adapt the normalizer.

    Return:
        A normalizer that is equivalent to the provided GMI normalizer
        adapted to the channels of the new sensor.
    """
    new_normalizer = deepcopy(gmi_normalizer)
    new_stats = {}
    for key in gmi_normalizer.stats:
        if key < 15:
            if key in sensor.gmi_channels:
                new_key = sensor.gmi_channels.index(key)
                print(key, new_key)
                new_stats[new_key] = gmi_normalizer.stats[key]
        else:
            d = 15 - sensor.n_chans
            new_stats[key - d] = gmi_normalizer.stats[key]
    new_normalizer.stats = new_stats
    return new_normalizer


def prepare_model(gmi_model, sensor, normalizer=None):
    """
    Create a new model for a given sensor using the weights from a
    pre-trained model for GMI.

    Args:
        gmi_model: The model trained on GMI data.
        sensor: The sensor for which this new model will be trained.
        normalizer: If provided, a new normalizer specific for the sensor.
            Otherwise the normalizer from the GMI model will be reused.

    Return:
        A new model for the given sensor, which reuses the weights from
        the GMI model that match the GMI input channels.
    """
    new_model = deepcopy(gmi_model)
    new_model.model = deepcopy(gmi_model.model)

    # Modify input layer
    old_layer = gmi_model.model.body.layers[0][0]
    out_features = old_layer.out_features
    bias = old_layer.bias is not None
    new_model.model.body.layers[0][0] = nn.Linear(
        sensor.n_inputs,
        out_features,
        bias=bias
    )
    new_layer = new_model.model.body.layers[0][0]
    for i, c in enumerate(sensor.gmi_channels):
        new_layer.weight.data[:, i] = old_layer.weight.data[:, c]

    # Modify projection layer
    old_layer = gmi_model.model.body.projection
    out_features = old_layer.out_features
    bias = old_layer.bias is not None
    new_model.model.body.projection = nn.Linear(
        sensor.n_inputs,
        out_features,
        bias=bias
    )
    new_layer = new_model.model.body.projection
    for i, c in enumerate(sensor.gmi_channels):
        new_layer.weight.data[:, i] = old_layer.weight.data[:, c]

    new_model.sensor = sensor
    new_model.n_inputs = sensor.n_inputs
    if normalizer is None:
        new_model.normalizer = adapt_normalizer(gmi_model.normalizer, sensor)
    else:
        new_model.normalizer = normalizer

    del new_model.training_history

    return new_model


###############################################################################
# GPROF-NN 3D
###############################################################################

class XceptionBlock(nn.Module):
    """
    Xception block consisting of two depth-wise separable convolutions
    each folowed by batch-norm and GELU activations.
    """

    def __init__(
            self,
            channels_in,
            channels_out,
            downsample=False,
            across_track=True
    ):
        """
        Args:
            channels_in: The number of incoming channels.
            channels_out: The number of outgoing channels.
            downsample: Whether or not to insert 3x3 max pooling block
                after the first convolution.
            across_track: Whether to downsample in across track direction.
        """
        super().__init__()
        if downsample:
            if across_track:
                stride = (2, 2)
            else:
                stride = (2, 1)

            self.block_1 = nn.Sequential(
            SeparableConv3x3(channels_in, channels_out),
            nn.GroupNorm(32, channels_out),
            SymmetricPadding(1),
            nn.MaxPool2d(kernel_size=3, stride=stride),
            nn.GELU())
        else:
            self.block_1 = nn.Sequential(
                SeparableConv3x3(channels_in, channels_out),
                nn.GroupNorm(32, channels_out),
                nn.GELU(),
            )

        self.block_2 = nn.Sequential(
            SeparableConv3x3(channels_out, channels_out),
            nn.GroupNorm(32, channels_out),
            nn.GELU(),
        )

        if channels_in != channels_out or downsample:
            if downsample:
                self.projection = nn.Conv2d(channels_in, channels_out, 1, stride=stride)
            else:
                self.projection = nn.Conv2d(channels_in, channels_out, 1)
        else:
            self.projection = None

    def forward(self, x):
        """
        Propagate input through block.
        """
        if self.projection is None:
            x_proj = x
        else:
            x_proj = self.projection(x)
        y = self.block_2(self.block_1(x))
        return x_proj + y

    def forward_no_activation(self, x):
        """
        Forward input through network but omit last activation.
        """
        y = self.block_1(x)
        for layer in self.block_2[:-1]:
            y = layer(y)
        return y



class DownsamplingStage(nn.Sequential):
    """
    Downsampling stage consisting of Xception blocks.
    """
    def __init__(self, n_channels, n_blocks, across_track=True):
        """
        Args:
            n_channels: The number of elements along the dimension 1 of the
                input_tensor.
            n_blocks: The number of blocks in the stage.
            across_track: If 'False' downsampling will not be performed along
                the across track direction.
        """
        blocks = [
            XceptionBlock(
                n_channels,
                n_channels,
                downsample=True,
                across_track=across_track
            )
        ]
        for i in range(n_blocks):
            blocks.append(XceptionBlock(n_channels, n_channels))
        super().__init__(*blocks)


    def forward_no_activation(self, x):
        """
        Forward input through stage but omit last activation.
        """
        y = x
        layers = list(self)
        for layer in layers[:-1]:
            y = layer(y)
        return layers[-1].forward_no_activation(y)


class UpsamplingStage(nn.Module):
    """
    Upsampling stage consisting of Xception blocks.
    """
    def __init__(self, n_channels, across_track=True):
        """
        Args:
            n_channels: The number of incoming and outgoing channels.
            across_track: Whether to upsample also along ``across_track``
                dimension.
        """
        super().__init__()
        if across_track:
            scale_factor = (2, 2)
        else:
            scale_factor = (2, 1)
        self.upsample = nn.Upsample(mode="bilinear",
                                    scale_factor=scale_factor,
                                    align_corners=False)
        self.block = nn.Sequential(
            SeparableConv3x3(n_channels * 2, n_channels),
            nn.GroupNorm(32, n_channels),
            nn.GELU(),
        )

    def forward(self, x, x_skip):
        """
        Propagate input through block.
        """
        x_up = self.upsample(x)
        x_merged = torch.cat([x_up, x_skip], 1)
        return self.block(x_merged)


class MLPHead(nn.Module):
    """
    Fully-convolutional implementation of a fully-connected network
    with residual connections.

    This module is used as network heads for different retrieval
    variables after the encode-decoder stage of the GPROF-NN 3D
    network.
    """
    def __init__(self, n_inputs, n_hidden, n_outputs, n_layers):
        """
        Args:
            n_inputs: The number of input channels to the layer.
            n_hidden: The number of channels in the 'hidden' layers
                of the module.
            n_outputs: The number of channels
            n_layers: The number of hidden layers.
        """
        super().__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        for i in range(n_layers - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(n_inputs, n_hidden, 1),
                    nn.GroupNorm(n_hidden, n_hidden),
                    nn.GELU(),
                )
            )
            n_inputs = n_hidden
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(n_hidden, n_outputs, 1),
            )
        )

    def __repr__(self):
        try:
            return (f"MLPHead(n_inputs={self.n_inputs}, n_hidden={self.n_hidden},"
                    f"n_outputs={self.n_outputs}, n_layers={self.n_layers})")
        except:
            return super().__repr__()

    def forward(self, x):
        """
        Propagate input through head.

        Args:
            x: 4D ``torch.Tensor`` containing the input.

        Return:
            4D ``torch.Tensor`` containing the nerwork output.
        """
        for l in self.layers[:-1]:
            y = l(x)
            n = min(x.shape[1], y.shape[1])
            y[:, :n] += x[:, :n]
            x = y
        return self.layers[-1](y)


class XceptionFPN(nn.Module):
    """
    This class implements the fully-convolutional neural network for the
    GPROF-NN 3D algorithm. The network body consists of an asymmetric
    encoder-decoder structure with skip connections between different
    stages. Each stage consists of a given number of Xception blocks.
    The head of the network consists of a separate MLP for all retrieval
    targets.
    """
    def __init__(
        self,
        sensor,
        n_outputs,
        n_blocks,
        n_features_body,
        n_layers_head,
        n_features_head,
        ancillary=True,
        targets=None,
    ):
        """
        Args:
            sensor: The sensor whose observations the network should process.
            n_outputs: The number of output features for the scalar retrieval
                variables.
            n_blocks: The number of blocks in each stage of the encoder. They
                may be given as a 5-element list providing the number of
                blocks in each of the 5 downsampling stages or as a single
                'int' if the number of blocks in each stage should be the
                same.
            n_features_body: The number of features/channels in the network
                body.
            n_layers_head: The number of layers in each network head.
            n_features_head: The number of features in each layer of each head.
            ancillary: Whether or not to make use of ancillary data.
            target: List of target variables.
        """
        super().__init__()

        self.sensor = sensor
        n_channels = sensor.n_chans
        n_anc = sensor.n_inputs - n_channels

        self.ancillary = ancillary
        if targets is None:
            self.targets = ["surface_precip"]
        else:
            self.targets = targets
        self.n_outputs = n_outputs

        if isinstance(n_blocks, int):
            n_blocks = [n_blocks] * 5

        self.in_block = nn.Conv2d(n_channels, n_features_body, 1)

        width = _INPUT_DIMENSIONS[sensor.sensor_id][0]

        self.down_block_2 = DownsamplingStage(
            n_features_body,
            n_blocks[0],
            across_track=width>4)
        width //= 2
        self.down_block_4 = DownsamplingStage(
            n_features_body,
            n_blocks[1],
            across_track=width>4)
        width //= 2
        self.down_block_8 = DownsamplingStage(
            n_features_body,
            n_blocks[2],
            across_track=width>4)
        width //= 2
        self.down_block_16 = DownsamplingStage(
            n_features_body,
            n_blocks[3],
            across_track=width>4
        )
        width //= 2
        self.down_block_32 = DownsamplingStage(
            n_features_body,
            n_blocks[4],
            across_track=width>4
        )

        self.up_block_16 = UpsamplingStage(
            n_features_body,
            across_track=width>4
        )
        width *= 2
        self.up_block_8 = UpsamplingStage(
            n_features_body,
            across_track=width>4
        )
        width *= 2
        self.up_block_4 = UpsamplingStage(
            n_features_body,
            across_track=width>4
        )
        width *= 2
        self.up_block_2 = UpsamplingStage(
            n_features_body,
            across_track=width>4
        )
        width *= 2
        self.up_block = UpsamplingStage(
            n_features_body,
            across_track=width>4)

        n_inputs = 2 * n_features_body
        if self.ancillary:
            n_inputs += n_anc

        targets = self.targets

        self.heads = nn.ModuleDict()
        for k in targets:
            if k in PROFILE_NAMES:
                self.heads[k] = MLPHead(n_inputs, n_features_head, 28, n_layers_head)
            else:
                self.heads[k] = MLPHead(
                    n_inputs, n_features_head, n_outputs, n_layers_head
                )

    def __repr__(self):
        return (f"XceptionFPN(sensor={self.sensor}, targets={self.targets}, "
                f"ancillary={self.ancillary})")

    def forward(self, x):
        """
        Propagate input through block.
        """
        n_chans = self.sensor.n_chans
        x_in = self.in_block(x[:, :n_chans])
        x_in[:, :n_chans] += x[:, :n_chans]

        x_2 = self.down_block_2(x_in)
        x_4 = self.down_block_4(x_2)
        x_8 = self.down_block_8(x_4)
        x_16 = self.down_block_16(x_8)
        x_32 = self.down_block_32(x_16)

        x_16_u = self.up_block_16(x_32, x_16)
        x_8_u = self.up_block_8(x_16_u, x_8)
        x_4_u = self.up_block_4(x_8_u, x_4)
        x_2_u = self.up_block_2(x_4_u, x_2)
        x_u = self.up_block(x_2_u, x_in)

        if self.ancillary:
            x = torch.cat([x_in, x_u, x[:, n_chans:]], 1)
        else:
            x = torch.cat([x_in, x_u], 1)

        targets = self.targets

        results = {}
        for k in targets:
            y = self.heads[k](x)
            results[k] = y
        return results

    def forward_until(self, x, n=1):
        """
        Forward input through network and return output activations from
        the nth downsampling block.
        """
        n_chans = self.sensor.n_chans
        x_in = self.in_block(x[:, :n_chans])
        x_in[:, :n_chans] += x[:, :n_chans]

        if n == 1:
            return self.down_block_2.forward_no_activation(x_in)
        x_2 = self.down_block_2(x_in)

        if n == 2:
            return self.down_block_4.forward_no_activation(x_2)
        x_4 = self.down_block_4(x_2)

        if n == 3:
            return self.down_block_8.forward_no_activation(x_4)
        x_8 = self.down_block_8(x_4)

        if n == 4:
            return self.down_block_16.forward_no_activation(x_8)
        x_16 = self.down_block_16(x_8)

        return self.down_block_32.forward_no_activation(x_16)


    def copy_model(self, sensor):
        """
        Create a new model for a different model and copy  weights
        from this model.

        Note: Only GMI based convolutional model can be used to derive
        models for other sensors.

        Args:
            sensor: The sensor for which to create a new model.

        Return:
            A new model for the given sensor.
        """
        n_blocks = [
            len(self.down_block_2) - 1,
            len(self.down_block_4) - 1,
            len(self.down_block_8) - 1,
            len(self.down_block_16) - 1,
            len(self.down_block_32) - 1
        ]

        n_features_body = self.in_block.out_channels
        n_layers_head = len(self.heads["surface_precip"].layers)
        n_features_head = self.heads["surface_precip"].layers[0][0].out_channels
        ancillary = self.ancillary
        targets = self.targets

        new_model = XceptionFPN(
            sensor,
            self.n_outputs,
            n_blocks,
            n_features_body,
            n_layers_head,
            n_features_head,
            targets)

        #
        # Copy relevant weights
        #

        for i, c in enumerate(sensor.gmi_channels):
            new_model.in_block.weight[:, i] = self.in_block.weight[:, c]
        new_model.in_block.bias[:] = self.in_block.bias[:]

        blocks = [self.down_block_2,
                  self.down_block_4,
                  self.down_block_8,
                  self.down_block_16,
                  self.down_block_32,
                  self.up_block_16,
                  self.up_block_8,
                  self.up_block_4,
                  self.up_block_2]
        new_blocks = [new_model.down_block_2,
                      new_model.down_block_4,
                      new_model.down_block_8,
                      new_model.down_block_16,
                      new_model.down_block_32,
                      new_model.up_block_16,
                      new_model.up_block_8,
                      new_model.up_block_4,
                      new_model.up_block_2]
        for b, b_new in zip(blocks, new_blocks):
            for p, p_new in zip(b.parameters(), b_new.parameters()):
                p_new.data[:] = p.data[:]

        for h, h_new in zip(self.heads.values(), new_model.heads.values()):
            for p, p_new in zip(h.parameters(), h_new.parameters()):
                if p.data.shape != p_new.data.shape:
                    n_body = 2 * n_features_body
                    p_new.data[:, :n_body] = p.data[:, :n_body]
                    p_new.data[:, n_body + 1:] = p.data[:, n_body:]
                else:
                    p_new.data[:] = p.data

        return new_model



class GPROF_NN_3D_QRNN(MRNN):
    """
    Neural network for the GPROF-NN 3D algorithm using quantnn's mixed
    regression neural network (MRNN) class to combine quantile regression
    for scalar variables and least-squares regression for profile targets.
    """
    def __init__(
        self,
        sensor,
        n_blocks,
        n_features_body,
        n_layers_head,
        n_features_head,
        activation="ReLU",
        targets=None,
        transformation=None,
        ancillary=True,
    ):
        """
        Args:
            sensor: The sensor for which this network is meant to retrieve
                rain rates.
            n_blocks: The number of blocks in each downsampling stage.
            n_features_body: The number of features in the network body.
            n_layers_head: The number of hidden layers in each head.
            n_features_head: The number of features in each head.
            activation: The activation to use in the network.
            targets: List of retrieval targets to retrieve.
            transformation: Transformation to apply to outputs.
            ancillary: Whether to use ancillary data in head.
        """
        if targets is None:
            targets = ALL_TARGETS
        self.targets = targets

        if transformation is not None and not isinstance(transformation, dict):
            if type(transformation) is type:
                transformation = {t: transformation() for t in targets}
            else:
                transformation = {t: transformation for t in targets}
            if "latent_heat" in targets:
                transformation["latent_heat"] = None

        model = XceptionFPN(
            sensor,
            64,
            n_blocks,
            n_features_body,
            n_layers_head,
            n_features_head,
            targets=targets,
            ancillary=ancillary,
        )

        losses = {}
        for target in targets:
            if target in PROFILE_NAMES:
                losses[target] = Mean()
            else:
                losses[target] = Quantiles(QUANTILES)

        super().__init__(
            n_inputs=sensor.n_inputs,
            losses=losses,
            model=model,
            transformation=transformation,
        )

        if ancillary:
            self.preprocessor_class = PreprocessorLoader3D
        else:
            self.preprocessor_class = L1CLoader3D
        self.netcdf_class = NetcdfLoader3D

        # Initialize attributes that will be set during training.
        self.normalizer = None
        self.configuration = None

    @property
    def suffix(self):
        return "3D"

    def __repr__(self):
        trained = getattr(self, "configuration", None) is not None
        return (f"GPROF_NN_3D_QRNN(targets={self.targets})")
        if trained:
            return (f"GPROF_NN_3D_QRNN(sensor={self.sensor}, "
                    f"configuration={self.configuration}, "
                    f"targets={self.targets})")
        else:
            return (f"GPROF_NN_3D_QRNN(sensor={self.sensor}, "
                    f"targets={self.targets})")

    def set_targets(self, targets):
        """
        Set target list.

        This function can be used to reduc
        """
        if not all([t in self.targets for t in targets]):
            raise ValueError("'targets' must be a sub-set of the models targets.")
        self.targets = targets
        self.model.targets = targets


class GPROF_NN_3D_DRNN(MRNN):
    """
    Neural network for the GPROF-NN 3D algorithm using quantnn's mixed
    regression neural network (MRNN) class to combine density regression
    for scalar variables and least-squares regression for profile targets.
    """
    def __init__(
        self,
        sensor,
        n_blocks,
        n_features_body,
        n_layers_head,
        n_features_head,
        activation="ReLU",
        targets=None,
        ancillary=True,
    ):
        """
        Args:
            sensor: The sensor for which this network is meant to retrieve
                rain rates.
            n_blocks: The number of blocks in each downsampling stage.
            n_features_body: The number of features in the network body.
            n_layers_head: The number of hidden layers in each head.
            n_features_head: The number of features in each head.
            activation: The activation to use in the network.
            targets: List of retrieval targets to retrieve.
            ancillary: Whether to use ancillary data in head.
        """
        self.sensor = sensor
        if targets is None:
            targets = ALL_TARGETS
        self.targets = targets

        model = XceptionFPN(
            sensor,
            128,
            n_blocks,
            n_features_body,
            n_layers_head,
            n_features_head,
            targets=targets,
            ancillary=True,
        )

        losses = {}
        for target in targets:
            if target in PROFILE_NAMES:
                losses[target] = Mean()
            else:
                losses[target] = Density(BINS[target])

        super().__init__(n_inputs=sensor.n_inputs, losses=losses, model=model)

        if ancillary:
            self.preprocessor_class = PreprocessorLoader3D
        else:
            self.preprocessor_class = L1CLoader3D

        self.netcdf_class = NetcdfLoader3D

    @property
    def suffix(self):
        return "3D"

    def set_targets(self, targets):
        """
        Set targets for retrival.

        This function can be used to reduce the number of retrieved
        targets during inference to speed up the retrieval.

        Args:
            targets: List of the targets to retrieve. Note: These must
                be a subset of the targets the network was trained to
                to retriev.
        """
        if not all([t in self.targets for t in targets]):
            raise ValueError("'targets' must be a sub-set of the models targets.")
        self.targets = targets
        self.model.targets = targets


# Included here for temporary compatibility with legacy naming.
GPROF_NN_2D_QRNN = GPROF_NN_3D_QRNN
GPROF_NN_2D_DRNN = GPROF_NN_3D_DRNN


###############################################################################
# Simulator
###############################################################################


class SimulatorNet(nn.Module):
    """
    Special version of the Xception FPN network for simulating brightness
    temperatures and biases.
    """
    def __init__(self, sensor, n_features_body, n_layers_head, n_features_head):
        """
        Args:
            n_features_body: The number of features/channels in the network
                body.
            n_layers_head: The number of layers in each network head.
            n_features_head: The number of features in each layer of each head.
        """
        super().__init__()
        self.sensor = sensor

        if sensor.n_angles > 1:
            n_chans_sim = sensor.n_chans
            n_biases = sensor.n_chans
            n_angs = sensor.n_angles
        else:
            n_chans_sim = sensor.n_chans
            n_biases = sensor.n_chans
            n_angs = 1

        self.in_block = nn.Conv2d(15, n_features_body, 1)

        self.down_block_2 = DownsamplingStage(n_features_body, 2)
        self.down_block_4 = DownsamplingStage(n_features_body, 2)
        self.down_block_8 = DownsamplingStage(n_features_body, 2)
        self.down_block_16 = DownsamplingStage(n_features_body, 2)
        self.down_block_32 = DownsamplingStage(n_features_body, 2)

        self.up_block_16 = UpsamplingStage(n_features_body)
        self.up_block_8 = UpsamplingStage(n_features_body)
        self.up_block_4 = UpsamplingStage(n_features_body)
        self.up_block_2 = UpsamplingStage(n_features_body)
        self.up_block = UpsamplingStage(n_features_body)

        n_inputs = 2 * n_features_body + 24
        self.bias_heads = nn.ModuleList()
        for i in range(n_biases):
            self.bias_heads.append(MLPHead(n_inputs, n_features_head, 1, n_layers_head))

        self.sim_heads = nn.ModuleList()
        for i in range(n_biases):
            self.sim_heads.append(
                MLPHead(n_inputs, n_features_head, n_angs, n_layers_head)
            )

    def forward(self, x):
        """
        Propagate input through block.
        """
        x_in = self.in_block(x[:, :15])
        x_in[:, :15] += x[:, :15]

        x_2 = self.down_block_2(x_in)
        x_4 = self.down_block_4(x_2)
        x_8 = self.down_block_8(x_4)
        x_16 = self.down_block_16(x_8)
        x_32 = self.down_block_32(x_16)

        x_16_u = self.up_block_16(x_32, x_16)
        x_8_u = self.up_block_8(x_16_u, x_8)
        x_4_u = self.up_block_4(x_8_u, x_4)
        x_2_u = self.up_block_2(x_4_u, x_2)
        x_u = self.up_block(x_2_u, x_in)

        x = torch.cat([x_in, x_u, x[:, 15:]], 1)

        n_chans = self.sensor.n_chans
        if self.sensor.n_angles > 1:
            n_angles = self.sensor.n_angles
            sim_shape = x.shape[:1] + (n_angles, n_chans) + x.shape[2:4]
        else:
            sim_shape = x.shape[:1] + (n_chans,) + x.shape[2:4]
        bias_shape = x.shape[:1] + (n_chans,) + x.shape[2:4]
        biases = [h(x) for h in self.bias_heads]
        sims = [h(x).unsqueeze(2) for h in self.sim_heads]

        results = {
            f"brightness_temperature_biases_{i}": b
            for i, b in enumerate(biases)
        }
        for i, s in enumerate(sims):
            key = f"simulated_brightness_temperatures_{i}"
            results[key] = s
        return results


class Simulator(MRNN):
    """
    Simulator QRNN to learn to predict simulated brightness temperatures
    from GMI observations.
    """
    def __init__(self, sensor, n_features_body, n_layers_head, n_features_head):
        """
        Args:
            sensor: Sensor object defining the object for which the simulations
                should be performed.
            n_features_body: The number of features in the body of the Xception
                FPN.
            n_layers_head: The number of layers in each head of the FPN.
            n_features_header: The number of features in each head.
        """
        model = SimulatorNet(sensor, n_features_body, n_layers_head, n_features_head)

        losses = {}
        for i in range(sensor.n_chans):
            losses[f"simulated_brightness_temperatures_{i}"] = Mean()
            losses[f"brightness_temperature_biases_{i}"] = Mean()

        super().__init__(losses, model=model)
        self.preprocessor_class = None
        self.netcdf_class = SimulatorLoader

    def set_targets(self, *args):
        """
        This function does nothing. It's a dummy function provided
        for compatibilty with retrieval driver interface.
        """
