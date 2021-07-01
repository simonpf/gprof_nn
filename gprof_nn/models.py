"""
gprof_nn.models
===========

Neural network models used for the GPROF-NN algorithms.
"""
import numpy as np
import torch
from torch import nn
from torch.nn.functional import softplus
from quantnn.qrnn import QRNN
from quantnn.drnn import DRNN
from quantnn.models.pytorch.xception import (UpsamplingBlock,
                                             DownsamplingBlock)


from gprof_nn.definitions import ALL_TARGETS, PROFILE_NAMES
from gprof_nn.retrieval import (NetcdfLoader0D,
                                NetcdfLoader2D,
                                PreprocessorLoader0D,
                                PreprocessorLoader2D)
from gprof_nn.data.training_data import (GPROF0DDataset,
                                         GPROF2DDataset)


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
    "latent_heat": np.concatenate([
        -np.logspace(-2, 2.5, 64)[::-1],
        np.array([0.0]),
        np.logspace(-2, 3.0, 64)
        ])
}

for k in BINS:
    if k != "latent_heat":
        BINS[k][0] == 0.0

QUANTILES = np.linspace(0.0, 1.0, 66)[1:-1]

RESIDUALS = ["none", "simple", "hyper"]

###############################################################################
# GPROF-NN 0D
###############################################################################

class MLP(nn.Module):
    """
    Vanilla Fully-connected feed-forward neural network.
    """
    def __init__(
        self,
        n_inputs,
        n_neurons,
        n_outputs,
        n_layers,
        activation="ReLU",
        internal=False
    ):
        """
        Create MLP object.

        Args:
            n_inputs: Number of input features in first layer.
            n_outputs: Number of output values.
            n_neurons: Number of neurons in hidden layers.
            n_layers: Number of layers including the output layer.
            activation: The activation function to use in each layer.
            internal: Whether or not activation layer norm and activation
                are applied to output from last layer.
        """
        super().__init__()
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

    def forward(self, x, *args, **kwargs):
        """
        Forward input through network.

        Args:
            x: The 2D input tensor to propagate through the network.
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
        internal=False
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
            internal=internal
        )
        if n_inputs != n_neurons:
            self.projection = nn.Linear(n_inputs, n_neurons)
        else:
            self.projection = None


    def forward(self, x, *args, **kwargs):
        """
        Forward input through network.

        Args:
            x: The 2D input tensor to propagate through the network.
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
        internal=False
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
            internal=internal
        )

    def forward(self, x, acc_in=None, li=1):
        """
        Forward input through network.

        Args:
            x: The 2D input tensor to propagate through the network.
            acc_in: Accumulator tensor containing accumulated activations
                 from previous layers.
            li: Layer index used to normalize accumulated activations.
        """
        if self.n_layers == 0:
            return x, None

        acc = torch.zeros((x.shape[0], self.n_neurons),
                          dtype=x.dtype,
                          device=x.device)
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
    Pytorch neural network model for the GPROF-NN 0D retrieval.

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
        target="surface_precip",
        activation="ReLU"
    ):
        self.target = target
        self.profile_shape = (-1, n_outputs, 28)
        self.n_layers_body = n_layers_body
        self.n_neurons_body = n_neurons_body

        residuals = residuals.lower()
        if residuals not in RESIDUALS:
            raise ValueError(
                f"'residuals' argument should be one of f{RESIDUALS}.")
        if residuals == "none":
            module_class = MLP
        elif residuals == "hyper":
            module_class = HyperResidualMLP
        else:
            module_class = ResidualMLP

        super().__init__()
        self.body = module_class(
            n_inputs,
            n_neurons_body,
            n_neurons_body,
            n_layers_body,
            activation=activation,
            internal=True
        )

        if not isinstance(self.target, list):
            targets = [self.target]
        else:
            targets = self.target
        if n_layers_body > 0:
            n_in = n_neurons_body
        else:
            n_in = 39

        self.heads = nn.ModuleDict()
        for t in targets:
            if t in PROFILE_NAMES:
                self.heads[t] = module_class(
                    n_in,
                    n_neurons_head,
                    28 * n_outputs,
                    n_layers_head,
                    activation=activation
                )
            else:
                self.heads[t] = module_class(
                    n_in,
                    n_neurons_head,
                    n_outputs,
                    n_layers_head,
                    activation=activation
                )

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
        if not isinstance(self.target, list):
            targets = [self.target]
        else:
            targets = self.target

        y, acc = self.body(x, None)
        results = {}
        for k in targets:
            results[k], _ = self.heads[k](y, acc, self.n_layers_body + 1)
            if k in PROFILE_NAMES:
                results[k] = results[k].reshape(self.profile_shape)
        if not isinstance(self.target, list):
            return results[self.target]
        return results


class GPROF_NN_0D_QRNN(QRNN):
    """
    DRNN-based version of the GPROF-NN 0D algorithm.
    """
    def __init__(self,
                 sensor,
                 n_layers_body,
                 n_neurons_body,
                 n_layers_head,
                 n_neurons_head,
                 activation="ReLU",
                 residuals="simple",
                 targets=None,
                 transformation=None
    ):
        self.sensor = sensor
        residuals = residuals.lower()
        if residuals not in RESIDUALS:
            raise ValueError(
                f"'residuals' argument should be one of {RESIDUALS}."
            )

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

        model = MultiHeadMLP(39,
                             n_layers_body,
                             n_neurons_body,
                             n_layers_head,
                             n_neurons_head,
                             64,
                             target=targets,
                             residuals=residuals,
                             activation=activation)

        super().__init__(n_inputs=sensor.n_inputs,
                         quantiles=QUANTILES,
                         model=model,
                         transformation=transformation)

        self.preprocessor_class = PreprocessorLoader0D
        self.netcdf_class = NetcdfLoader0D


class GPROF_NN_0D_DRNN(DRNN):
    """
    DRNN-based version of the GPROF-NN 0D algorithm.
    """
    def __init__(self,
                 sensor,
                 n_layers_body,
                 n_neurons_body,
                 n_layers_head,
                 n_neurons_head,
                 activation="ReLU",
                 residuals="simple",
                 targets=None
    ):
        self.sensor = sensor
        residuals = residuals.lower()
        if residuals not in RESIDUALS:
            raise ValueError(
                f"'residuals' argument should be one of {RESIDUALS}."
            )

        if targets is None:
            targets = ALL_TARGETS
        self.targets = targets

        model = MultiHeadMLP(sensor.n_inputs,
                             n_layers_body,
                             n_neurons_body,
                             n_layers_head,
                             n_neurons_head,
                             128,
                             target=targets,
                             residuals=residuals,
                             activation=activation)

        super().__init__(n_inputs=39,
                         bins=BINS,
                         model=model)

        self.preprocessor_class = PreprocessorLoader0D
        self.training_data_class = GPROF0DDataset

        self.preprocessor_class = PreprocessorLoader0D
        self.netcdf_class = NetcdfLoader0D


###############################################################################
# GPROF-NN 2D
###############################################################################


class MLPHead(nn.Module):
    """
    MLP-type head for convolutional network.
    """
    def __init__(self,
                 n_inputs,
                 n_hidden,
                 n_outputs,
                 n_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers - 1):
            self.layers.append(nn.Sequential(
                nn.Conv2d(n_inputs, n_hidden, 1),
                nn.GroupNorm(1, n_hidden),
                nn.ReLU()
            ))
            n_inputs = n_hidden
        self.layers.append(nn.Sequential(
            nn.Conv2d(n_hidden, n_outputs, 1),
        ))

    def forward(self, x):
        "Propagate input through head."
        for l in self.layers[:-1]:
            y = l(x)
            n = min(x.shape[1], y.shape[1])
            y[:, :n] += x[:, :n]
            x = y
        return self.layers[-1](y)


class XceptionFPN(nn.Module):
    """
    Feature pyramid network (FPN) with 5 stages based on xception
    architecture.
    """

    def __init__(self,
                 n_outputs,
                 n_blocks,
                 n_features_body,
                 n_layers_head,
                 n_features_head,
                 ancillary=True,
                 target=None):
        """
        Args:
            n_outputs: The number of output channels,
            n_blocks: The number of blocks in each stage of the encoder.
            n_features_body: The number of features/channels in the network
                body.
            n_layers_head: The number of layers in each network head.
            n_features_head: The number of features in each layer of each head.
            ancillary: Whether or not to make use of ancillary data.
            target: List of target variables.
        """
        super().__init__()
        self.ancillary = ancillary
        if target is None:
            self.target = ["surface_precip"]
        else:
            self.target = target
        self.n_outputs = n_outputs

        if isinstance(n_blocks, int):
            n_blocks = [n_blocks] * 5

        self.in_block = nn.Conv2d(15, n_features_body, 1)

        self.down_block_2 = DownsamplingBlock(n_features_body, n_blocks[0])
        self.down_block_4 = DownsamplingBlock(n_features_body, n_blocks[1])
        self.down_block_8 = DownsamplingBlock(n_features_body, n_blocks[2])
        self.down_block_16 = DownsamplingBlock(n_features_body, n_blocks[3])
        self.down_block_32 = DownsamplingBlock(n_features_body, n_blocks[4])

        self.up_block_16 = UpsamplingBlock(n_features_body)
        self.up_block_8 = UpsamplingBlock(n_features_body)
        self.up_block_4 = UpsamplingBlock(n_features_body)
        self.up_block_2 = UpsamplingBlock(n_features_body)
        self.up_block = UpsamplingBlock(n_features_body)

        n_inputs = 2 * n_features_body
        if self.ancillary:
            n_inputs += 24

        targets = self.target
        if not isinstance(targets, list):
            targets = [targets]
        self.heads = nn.ModuleDict()
        for k in targets:
            if k in PROFILE_NAMES:
                self.heads[k] = MLPHead(n_inputs,
                                        n_features_head,
                                        28 * n_outputs,
                                        n_layers_head)
            else:
                self.heads[k] = MLPHead(n_inputs,
                                        n_features_head,
                                        n_outputs,
                                        n_layers_head)

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

        if self.ancillary:
            x = torch.cat([x_in, x_u, x[:, 15:]], 1)
        else:
            x = torch.cat([x_in, x_u], 1)

        if not isinstance(self.target, list):
            targets = [self.target]
        else:
            targets = self.target

        results = {}
        for k in targets:
            y = self.heads[k](x)
            if k in PROFILE_NAMES:
                profile_shape = y.shape[:1] + (self.n_outputs, 28) + y.shape[2:4]
                results[k] = y.reshape(profile_shape)
            else:
                results[k] = y
        if not isinstance(self.target, list):
            return results[self.target]
        return results


class GPROF_NN_2D_QRNN(QRNN):
    """
    QRNN-based version of the GPROF-NN 2D algorithm.
    """
    def __init__(self,
                 n_blocks,
                 n_features_body,
                 n_layers_head,
                 n_features_head,
                 activation="ReLU",
                 targets=None,
                 transformation=None
    ):
        """
        Args:
            n_blocks: The number of blocks in each downsampling stage.
            n_features_body: The number of features in the network body.
            n_layers_head: The number of hidden layers in each head.
            n_features_head: The number of features in each head.
            activation: The activation to use in the network.
            targets: List of retrieval targets to retrieve.
            transformation: Transformation to apply to outputs.
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

        model = XceptionFPN(64,
                            n_blocks,
                            n_features_body,
                            n_layers_head,
                            n_features_head,
                            target=targets)

        super().__init__(n_inputs=39,
                         quantiles=QUANTILES,
                         model=model,
                         transformation=transformation)

        self.preprocessor_class = PreprocessorLoader2D
        self.netcdf_class = NetcdfLoader2D


class GPROF_NN_2D_DRNN(DRNN):
    """
    QRNN-based version of the GPROF-NN 2D algorithm.
    """
    def __init__(self,
                 n_blocks,
                 n_features_body,
                 n_layers_head,
                 n_features_head,
                 activation="ReLU",
                 targets=None,
    ):
        """
        Args:
            n_blocks: The number of blocks in each downsampling stage.
            n_features_body: The number of features in the network body.
            n_layers_head: The number of hidden layers in each head.
            n_features_head: The number of features in each head.
            activation: The activation to use in the network.
            targets: List of retrieval targets to retrieve.
        """
        if targets is None:
            targets = ALL_TARGETS
        self.targets = targets

        model = XceptionFPN(128,
                            n_blocks,
                            n_features_body,
                            n_layers_head,
                            n_features_head,
                            target=targets)

        super().__init__(n_inputs=39,
                         bins=BINS,
                         model=model)

        self.preprocessor_class = PreprocessorLoader2D
        self.netcdf_class = NetcdfLoader2D
