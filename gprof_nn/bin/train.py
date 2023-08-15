"""
==================
gprof_nn.bin.train
==================

This sub-module implements the 'train' sub-command of the
'gprof_nn' command line application, which trains
networks for the GPROF-NN retrieval algorithm.
"""
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from gprof_nn import sensors
from gprof_nn.normalizer import get_normalizer
import gprof_nn.logging
from gprof_nn.retrieval import RetrievalDriver, RetrievalGradientDriver
from gprof_nn.definitions import (
    ALL_TARGETS,
    PROFILE_NAMES,
    CONFIGURATIONS,
    GPROF_NN_DATA_PATH,
    ANCILLARY_DATA
)
from gprof_nn.data.training_data import (
    GPROF_NN_1D_Dataset,
    GPROF_NN_3D_Dataset,
    GPROF_NN_HR_Dataset,
    SimulatorDataset,
    HR_TARGETS,
)
from gprof_nn.models import (
    GPROF_NN_1D_DRNN,
    GPROF_NN_1D_QRNN,
    GPROF_NN_3D_DRNN,
    GPROF_NN_3D_QRNN,
    Simulator,
    GPROF_NN_HR,
)


LOGGER = logging.getLogger(__name__)


def add_parser(subparsers):
    """
    Add parser for 'train' command to top-level parser. This function
    is called from the top-level parser defined in 'gprof_nn.bin'.

    Args:
        subparsers: The subparsers object provided by the top-level parser.
    """
    parser = subparsers.add_parser(
        "train",
        help="Train a GPROF-NN model.",
        description=(
            """
            Trains a GPROF-NN 1D, 3D, simulator or HR retrieval network.

            By default, the training will proceed as described in the GPROF-NN
            paper and the network will be trained to retrieve all standard
            GPROF variables. The options below can be used to modify
            the training.

            Tensorboard logs are written into the 'runs' directory in the
            current working directory.

            If the 'model_output' argument points to an exisiting model, the
            training of this model will be continued.
            """
        ),
    )

    # Input and output data
    parser.add_argument(
        "variant",
        metavar="variant",
        type=str,
        help="The type of GPROF-NN model to train: '1D', '3D', 'SIM' or 'HR'",
    )
    parser.add_argument(
        "sensor",
        metavar="sensor",
        type=str,
        help=(
            "Name of the sensor for which to train the algorithm. "
            "The name should correspond to a sensor object defined in"
            " 'gprof_nn.sensors'."
        )
    )
    parser.add_argument(
        "configuration",
        metavar="[ERA5/GANAL]",
        type=str,
        help=(
            "The type of ancillary data that the model is trained for."
        ),
    )
    parser.add_argument(
        "training_data",
        metavar="training_data",
        type=str,
        help="Path containing the training data.",
    )
    parser.add_argument(
        "validation_data",
        metavar="validation_data",
        type=str,
        help="Path containing the validation data.",
    )
    parser.add_argument(
        "output",
        metavar="output",
        type=str,
        help="""
        Path or file to which to write the trained model. If 'output'
        points to an existing model, its training will be continued.
        """
    )

    # Model configuration
    parser.add_argument(
        "--type",
        metavar="network_type",
        type=str,
        help=(
        """
        The type of network: 'DRNN', 'QRNN' or 'QRNN_EXP'. 'DRNN' predicts
        the posterior PDF of the retrieval over a range of bins. 'QRNN'j
        predicts quantiles for the posterior, while 'QRNN_EXP' also
        applies a log-linear transform to all strictly positive retrieval
        variables. Defaults to 'QRNN_EXP'.
        """
        ),
        default="qrnn_exp",
    )

    parser.add_argument(
        "--n_layers_body",
        metavar="n",
        type=int,
        default=6,
        help=(
            "For GPROF-NN 1D: The number of hidden layers in the shared body"
            " of the network."
        ),
    )

    parser.add_argument(
        "--n_neurons_body",
        metavar="n",
        type=int,
        default=256,
        help=("For GPROF-NN 1D and 3D: The number of neurons in the body."),
    )
    parser.add_argument(
        "--n_layers_head",
        metavar="n",
        type=int,
        default=4,
        help="For GPROF-NN 1D: The number of layers in each head of the model.",
    )
    parser.add_argument(
        "--n_neurons_head",
        metavar="n",
        type=int,
        default=128,
        help=(
            """
            How many neurons/features in each head of the "network.
            """
        ),
    )
    parser.add_argument(
        "--n_blocks",
        metavar="N",
        type=int,
        nargs="+",
        default=[5],
        help=(
            """
            For GPROF-NN 3D, SIM and HR: The number of Xception blocks per
            downsampling stage of the model.
            """
        ),
    )
    parser.add_argument(
        "--activation",
        metavar="activation",
        type=str,
        default="GELU",
        help="The type of activation function to apply in the model.",
    )
    parser.add_argument(
        "--residuals",
        metavar="residuals",
        type=str,
        default="hyper",
        help=(
            """
            For GPROF-NN 1D: The type of residual connections to apply.
            'none', 'simple' or 'hyper'. Deaults to 'hyper'.
            """
        )
    )
    parser.add_argument(
        "--n_epochs",
        metavar="n",
        type=int,
        nargs="*",
        default=[10, 20, 30],
        help=(
            """
            For how many epochs to train the network. When multiple values
            are given the network is trained multiple times each time
            performing. a warm restart.
            """
        ),
    )
    parser.add_argument(
        "--learning_rate",
        metavar="lr",
        type=float,
        nargs="*",
        default=[0.0005, 0.0005, 0.0005],
        help=(
        """
        The learning rate for each training run.
        """
        )
    )
    parser.add_argument(
        "--no_lr_schedule", action="store_true", help="Disable learning rate schedule."
    )
    parser.add_argument(
        "--no_ancillary",
        action="store_false",
        help="Don't use acillary data in retrieval.",
    )
    parser.add_argument(
        "--no_validation",
        action="store_true",
        help="Disable performance monitoring on validation set",
    )
    parser.add_argument(
        "--normalizer",
        type=str,
        help="Path to a custom normalizer object to use."
    )

    # Other
    parser.add_argument(
        "--device",
        metavar="device",
        type=str,
        default="cuda",
        help="The name of the device on which to run the training",
    )
    parser.add_argument(
        "--targets",
        metavar="target",
        type=str,
        nargs="+",
        default=None,
        help=(
            "List of the targets to train the model on."
        )
    )
    parser.add_argument(
        "--batch_size",
        metavar="n",
        type=int,
        help="The batch size to use for training.",
        default=8,
    )
    parser.add_argument(
        "--permute",
        metavar="feature_index",
        type=int,
        help=(
            "If provided, the input feature with the given index " "will be permuted."
        ),
    )
    parser.set_defaults(func=run)
    parser.add_argument(
        "--n_processes_train",
        metavar="n",
        type=int,
        default=6,
        help=(
            "The number of processes to use to load the training data."
        ),
    )
    parser.add_argument(
        "--n_processes_val",
        metavar="n",
        type=int,
        default=4,
        help=(
            "The number of processes to use to load the validation data."
        ),
    )
    parser.add_argument(
        "--drop_inputs",
        metavar="index",
        type=int,
        nargs="*",
        default=None,
        help=(
            "A list of indices specifying inputs to drop from the GPROF input."
            "The inputs are ordered as follows: \n"
            " [1] Tbs\n (Number of inputs depends on sensor.)"
            " [2]. Earth incidence angle (Only cross-track scanning sensors)\n"
            " [3] TCWV\n"
            " [4] T2m\n"
            " [5] Surface types (16 inputs)\n"
            " [6] Airmass types (4 inputs)\n"
        ),
    )
    parser.add_argument(
        "--simulated_tbs",
        action="store_true",
        help=(
            "Train the retrieval using only simulated TBs."
        )
    )

def run(args):
    """
    Run the training.

    Args:
        args: The namespace object provided by the top-level parser.
    """

    sensor = args.sensor
    configuration = args.configuration
    training_data = args.training_data[0]
    validation_data = args.validation_data[0]

    #
    # Determine sensor
    #

    sensor = sensor.strip().upper()
    sensor = getattr(sensors, sensor, None)
    if sensor is None:
        LOGGER.error("Sensor '%s' is not supported.", args.sensor.strip().upper())
        return 1
    if args.simulated_tbs:
        sensor.use_simulated_tbs = True

    variant = args.variant
    if variant.upper() not in ["1D", "3D", "SIM", "HR"]:
        LOGGER.error("'variant' should be one of ['1D', '3D', 'SIM', 'HR']")
        return 1

    #
    # Configuration
    #

    if configuration.upper() not in CONFIGURATIONS:
        LOGGER.error("'configuration' should be one of $s.", CONFIGURATIONS)
        return 1

    # Check output path and define model name if necessary.
    output = Path(args.output)
    if output.is_dir() and not output.exists():
        LOGGER.error("The output path '%s' doesn't exist.", output)
        return 1
    if not output.is_dir() and not output.parent.exists():
        LOGGER.error("The output path '%s' doesn't exist.", output.parent)
        return 1
    if output.is_dir():
        network_name = (
            f"gprof_nn_{variant.lower()}_{sensor.name.lower()}_"
            f"{configuration.lower()}.pckl"
        )
        output = output / network_name

    training_data = args.training_data
    validation_data = args.validation_data

    if variant.upper() == "1D":
        run_training_1d(
            sensor, configuration, training_data, validation_data, output, args
        )
    elif variant.upper() == "3D":
        run_training_3d(
            sensor, configuration, training_data, validation_data, output, args
        )
    elif variant.upper() == "SIM":
        run_training_sim(
            sensor, configuration, training_data, validation_data, output, args
        )
    elif variant.upper() == "HR":
        run_training_hr(configuration, training_data, validation_data, output, args)
    else:
        raise ValueError("'variant' should be one of '1D', '3D', 'SIM'.")


def run_training_1d(
    sensor, configuration, training_data, validation_data, output, args
):
    """
    Run training for GPROF-NN 1D algorithm.

    Args:
        sensor: Sensor object representing the sensor for which to train
            an algorithm.
        configuration: String identifying the retrieval configuration.
        training_data: The path to the training data.
        validation_data: The path to the validation data.
        output: Path to which to write the resulting model.
        args: Namespace with the remaining command line arguments.
    """
    from quantnn.qrnn import QRNN
    from quantnn.normalizer import Normalizer
    from quantnn.data import DataFolder
    from quantnn.transformations import LogLinear
    from quantnn.models.pytorch.logging import TensorBoardLogger
    from quantnn.metrics import ScatterPlot
    import torch
    from torch import optim

    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_num_threads(1)

    n_layers_body = args.n_layers_body
    n_neurons_body = args.n_neurons_body
    n_layers_head = args.n_layers_head
    n_neurons_head = args.n_neurons_head

    activation = args.activation
    residuals = args.residuals

    device = args.device
    targets = args.targets
    if targets is None:
        targets = ALL_TARGETS

    network_type = args.type
    batch_size = args.batch_size
    permute = args.permute
    ancillary = args.no_ancillary

    n_epochs = args.n_epochs
    lr = args.learning_rate
    no_schedule = args.no_lr_schedule

    if len(n_epochs) == 1:
        n_epochs = n_epochs * len(lr)
    if len(lr) == 1:
        lr = lr * len(n_epochs)

    if args.normalizer is None:
        normalizer = get_normalizer(sensor)
    else:
        normalizer = Normalizer.load(args.normalizer)

    #
    # Load training data.
    #

    dataset_factory = GPROF_NN_1D_Dataset
    kwargs = {
        "sensor": sensor,
        "batch_size": batch_size,
        "normalizer": normalizer,
        "targets": targets,
        "augment": True,
        "permute": permute,
    }

    n_workers = args.n_processes_train
    training_data = DataFolder(
        training_data,
        dataset_factory,
        kwargs=kwargs,
        queue_size=1024,
        n_workers=n_workers
    )

    if args.no_validation:
        validation_data = None
    else:
        kwargs = {
            "sensor": sensor,
            "batch_size": 4 * batch_size,
            "normalizer": normalizer,
            "targets": targets,
            "augment": False,
            "permute": permute,
        }
        n_workers = args.n_processes_val
        validation_data = DataFolder(
            validation_data,
            dataset_factory,
            kwargs=kwargs,
            queue_size=64,
            n_workers=n_workers
        )

    #
    # Create neural network model
    #

    if Path(output).exists():
        try:
            xrnn = QRNN.load(output)
            LOGGER.info(f"Continuing training of existing model {output}.")
        except Exception:
            xrnn = None
    else:
        xrnn = None

    if xrnn is None:
        LOGGER.info(f"Creating new model of type {network_type}.")

        if args.drop_inputs is not None:
            input_names = {
                ind: f"channel {ind}" for ind in range(sensor.n_chans)
            }
            for ind, name in enumerate(ANCILLARY_DATA):
                input_names[sensor.n_chans + ind] = name
            input_names = [input_names[index] for index in args.drop_inputs]
            LOGGER.info(f"Dropped inputs: {input_names}.")

        if network_type == "drnn":
            xrnn = GPROF_NN_1D_DRNN(
                sensor,
                n_layers_body,
                n_neurons_body,
                n_layers_head,
                n_neurons_head,
                activation=activation,
                residuals=residuals,
                targets=targets,
                ancillary=ancillary,
                drop_inputs=args.drop_inputs,
            )
        elif network_type == "qrnn_exp":
            transformation = {t: LogLinear() for t in targets}
            transformation["latent_heat"] = None
            xrnn = GPROF_NN_1D_QRNN(
                sensor,
                n_layers_body,
                n_neurons_body,
                n_layers_head,
                n_neurons_head,
                activation=activation,
                residuals=residuals,
                transformation=transformation,
                targets=targets,
                ancillary=ancillary,
                drop_inputs=args.drop_inputs,
            )
        else:
            xrnn = GPROF_NN_1D_QRNN(
                sensor,
                n_layers_body,
                n_neurons_body,
                n_layers_head,
                n_neurons_head,
                activation=activation,
                residuals=residuals,
                targets=targets,
                ancillary=ancillary,
                drop_inputs=args.drop_inputs,
            )
    model = xrnn.model
    xrnn.normalizer = normalizer
    xrnn.configuration = configuration
    xrnn.sensor = sensor.full_name

    ###############################################################################
    # Run the training.
    ###############################################################################

    n_epochs_tot = sum(n_epochs)
    logger = TensorBoardLogger(n_epochs_tot)
    logger.set_attributes(
        {
            "sensor": sensor.name,
            "configuration": configuration,
            "n_layers_body": n_layers_body,
            "n_neurons_body": n_neurons_body,
            "n_layers_head": n_layers_head,
            "n_neurons_head": n_neurons_head,
            "activation": activation,
            "residuals": residuals,
            "targets": ", ".join(targets),
            "type": network_type,
            "optimizer": "adam",
        }
    )
    metrics = ["MeanSquaredError", "Bias", "CalibrationPlot", "CRPS"]
    scatter_plot = ScatterPlot(log_scale=True)
    metrics.append(scatter_plot)

    for n, r in zip(n_epochs, lr):
        LOGGER.info(f"Starting training for {n} epochs with learning rate {r}")
        optimizer = optim.Adam(model.parameters(), lr=r)
        if no_schedule:
            scheduler = None
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n)
        xrnn.train(
            training_data=training_data,
            validation_data=validation_data,
            n_epochs=n,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
            metrics=metrics,
            device=device,
            mask=-9999,
        )
        LOGGER.info(f"Saving training network to {output}.")
        xrnn.save(output)


def run_training_3d(
    sensor, configuration, training_data, validation_data, output, args
):
    """
    Run training for GPROF-NN 3D algorithm.

    Args:
        sensor: Sensor object representing the sensor for which to train
            an algorithm.
        configuration: String identifying the retrieval configuration.
        training_data: The path to the training data.
        validation_data: The path to the validation data.
        output: Path to which to write the resulting model.
        args: Namespace with the remaining command line arguments.
    """
    from quantnn.qrnn import QRNN
    from quantnn.normalizer import Normalizer
    from quantnn.data import DataFolder
    from quantnn.transformations import LogLinear
    from quantnn.models.pytorch.logging import TensorBoardLogger
    from quantnn.metrics import ScatterPlot
    import torch
    from torch import optim

    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_num_threads(1)

    n_blocks = args.n_blocks[0]
    n_neurons_body = args.n_neurons_body
    n_layers_head = args.n_layers_head
    n_neurons_head = args.n_neurons_head

    device = args.device
    targets = args.targets
    network_type = args.type
    batch_size = args.batch_size

    n_epochs = args.n_epochs
    lr = args.learning_rate
    no_schedule = args.no_lr_schedule
    ancillary = args.no_ancillary

    if len(n_epochs) == 1:
        n_epochs = n_epochs * len(lr)
    if len(lr) == 1:
        lr = lr * len(n_epochs)

    if args.normalizer is None:
        normalizer = get_normalizer(sensor)
    else:
        normalizer = Normalizer.load(args.normalizer)

    #
    # Load training data.
    #

    dataset_factory = GPROF_NN_3D_Dataset
    kwargs = {
        "sensor": sensor,
        "batch_size": batch_size,
        "normalizer": normalizer,
        "targets": targets,
        "augment": True,
    }
    n_workers = args.n_processes_train
    training_data = DataFolder(
        training_data,
        dataset_factory,
        queue_size=256,
        kwargs=kwargs,
        n_workers=n_workers
    )

    if args.no_validation:
        validation_data = None
    else:
        kwargs = {
            "sensor": sensor,
            "batch_size": 4 * batch_size,
            "normalizer": normalizer,
            "targets": targets,
            "augment": False,
        }
        n_workers = args.n_processes_val
        validation_data = DataFolder(
            validation_data,
            dataset_factory,
            queue_size=256,
            kwargs=kwargs,
            n_workers=n_workers
        )

    ###############################################################################
    # Prepare in- and output.
    ###############################################################################

    #
    # Create neural network model
    #

    if Path(output).exists():
        try:
            xrnn = QRNN.load(output)
            LOGGER.info(f"Continuing training of existing model {output}.")
        except Exception:
            xrnn = None
    else:
        xrnn = None

    if xrnn is None:
        LOGGER.info(f"Creating new model of type {network_type}.")

        if args.drop_inputs is not None:
            input_names = {
                ind: f"channel {ind}" for ind in range(sensor.n_chans)
            }
            for ind, name in enumerate(ANCILLARY_DATA):
                input_names[sensor.n_chans + ind] = name
            input_names = [input_names[index] for index in args.drop_inputs]
            LOGGER.info(f"Dropped inputs: ", input_names)

        if network_type == "drnn":
            xrnn = GPROF_NN_3D_DRNN(
                sensor,
                n_blocks,
                n_neurons_body,
                n_layers_head,
                n_neurons_head,
                targets=targets,
                ancillary=ancillary,
                drop_inputs=args.drop_inputs,
            )
        elif network_type == "qrnn_exp":
            transformation = {}
            for target in targets:
                if target in PROFILE_NAMES:
                    transformation[target] = None
                else:
                    transformation[target] = LogLinear()
            xrnn = GPROF_NN_3D_QRNN(
                sensor,
                n_blocks,
                n_neurons_body,
                n_layers_head,
                n_neurons_head,
                transformation=transformation,
                targets=targets,
                ancillary=ancillary,
                drop_inputs=args.drop_inputs,
            )
        else:
            xrnn = GPROF_NN_3D_QRNN(
                sensor,
                n_blocks,
                n_neurons_body,
                n_layers_head,
                n_neurons_head,
                targets=targets,
                ancillary=ancillary,
                drop_inputs=args.drop_inputs,
            )
    model = xrnn.model
    xrnn.normalizer = normalizer
    xrnn.configuration = configuration
    xrnn.sensor = sensor.full_name

    ###############################################################################
    # Run the training.
    ###############################################################################

    n_epochs_tot = sum(n_epochs)
    logger = TensorBoardLogger(n_epochs_tot)
    logger.set_attributes(
        {
            "n_blocks": n_blocks,
            "n_neurons_body": n_neurons_body,
            "n_layers_head": n_layers_head,
            "n_neurons_head": n_neurons_head,
            "targets": ", ".join(targets),
            "type": network_type,
            "optimizer": "adam",
        }
    )

    metrics = ["MeanSquaredError", "Bias", "CalibrationPlot", "CRPS"]
    scatter_plot = ScatterPlot(log_scale=True)
    metrics.append(scatter_plot)

    for n, r in zip(n_epochs, lr):
        LOGGER.info(f"Starting training for {n} epochs with learning rate {r}")
        optimizer = optim.Adam(model.parameters(), lr=r)
        if no_schedule:
            scheduler = None
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n)
        xrnn.model.train(False)
        xrnn.train(
            training_data=training_data,
            validation_data=validation_data,
            n_epochs=n,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
            metrics=metrics,
            device=device,
            mask=-9999,
        )
        LOGGER.info(f"Saving training network to {output}.")
        xrnn.save(output)


def run_training_sim(
    sensor, configuration, training_data, validation_data, output, args
):
    """
    Train simulator network for sensors other than GMI.

    Args:
        sensor: Sensor object representing the sensor for which to train
            the model.
        configuration: String identifying the retrieval configuration.
        training_data: The path to the training data.
        validation_data: The path to the validation data.
        output: Path to which to write the resulting model.
        args: Namespace with the remaining command line arguments.
    """
    from quantnn.qrnn import QRNN
    from quantnn.normalizer import Normalizer
    from quantnn.data import DataFolder
    from quantnn.transformations import LogLinear
    from quantnn.models.pytorch.logging import TensorBoardLogger
    from quantnn.metrics import ScatterPlot
    import torch
    from torch import optim

    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_num_threads(1)

    n_blocks = args.n_blocks[0]
    n_neurons_body = args.n_neurons_body
    n_layers_head = args.n_layers_head
    n_neurons_head = args.n_neurons_head

    device = args.device
    batch_size = args.batch_size

    n_epochs = args.n_epochs
    lr = args.learning_rate
    no_schedule = args.no_lr_schedule
    ancillary = args.no_ancillary

    if len(n_epochs) == 1:
        n_epochs = n_epochs * len(lr)
    if len(lr) == 1:
        lr = lr * len(n_epochs)

    if args.normalizer is None:
        normalizer = get_normalizer(sensors.GMI)
    else:
        normalizer = Normalizer.load(args.normalizer)

    #
    # Load training data.
    #

    dataset_factory = SimulatorDataset
    kwargs = {"batch_size": batch_size, "normalizer": normalizer, "augment": True}

    n_workers = args.n_processes_train
    training_data = DataFolder(
        training_data,
        dataset_factory,
        queue_size=256,
        kwargs=kwargs,
        n_workers=n_workers
    )

    if args.no_validation:
        validation_data = None
    else:
        kwargs = {
            "batch_size": 4 * batch_size,
            "normalizer": normalizer,
            "augment": False,
        }
        n_workers = args.n_processes_val
        validation_data = DataFolder(
            validation_data,
            dataset_factory,
            queue_size=256,
            kwargs=kwargs,
            n_workers=n_workers
        )

    ###############################################################################
    # Prepare in- and output.
    ###############################################################################

    #
    # Create neural network model
    #

    if Path(output).exists():
        try:
            xrnn = QRNN.load(output)
            LOGGER.info(f"Continuing training of existing model {output}.")
        except Exception:
            xrnn = None
    else:
        xrnn = None

    if xrnn is None:
        LOGGER.info(f"Creating new simulator model.")
        xrnn = Simulator(sensor, n_neurons_body, n_layers_head, n_neurons_head)

    model = xrnn.model
    xrnn.normalizer = normalizer
    xrnn.configuration = configuration
    xrnn.sensor = sensor.full_name

    ###############################################################################
    # Run the training.
    ###############################################################################

    n_epochs_tot = sum(n_epochs)
    logger = TensorBoardLogger(n_epochs_tot)
    logger.set_attributes(
        {
            "n_blocks": n_blocks,
            "n_neurons_body": n_neurons_body,
            "n_layers_head": n_layers_head,
            "n_neurons_head": n_neurons_head,
            "optimizer": "adam",
        }
    )

    metrics = ["MeanSquaredError", "Bias", "CalibrationPlot", "CRPS"]
    scatter_plot = ScatterPlot()
    metrics.append(scatter_plot)

    for n, r in zip(n_epochs, lr):
        LOGGER.info(f"Starting training for {n} epochs with learning rate {r}")
        optimizer = optim.Adam(model.parameters(), lr=r)
        if no_schedule:
            scheduler = None
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n)
        xrnn.train(
            training_data=training_data,
            validation_data=validation_data,
            n_epochs=n,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
            metrics=metrics,
            device=device,
            mask=-9999,
        )
        LOGGER.info(f"Saving training network to {output}.")
        xrnn.save(output)


def run_training_hr(configuration, training_data, validation_data, output, args):
    """
    Train simulator network for sensors other than GMI.

    Args:
        sensor: Sensor object representing the sensor for which to train
            the model.
        configuration: String identifying the retrieval configuration.
        training_data: The path to the training data.
        validation_data: The path to the validation data.
        output: Path to which to write the resulting model.
        args: Namespace with the remaining command line arguments.
    """
    from quantnn.qrnn import QRNN
    from quantnn.normalizer import Normalizer
    from quantnn.data import DataFolder
    from quantnn.transformations import LogLinear
    from quantnn.models.pytorch.logging import TensorBoardLogger
    from quantnn.metrics import ScatterPlot
    import torch
    from torch import optim

    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_num_threads(1)

    n_blocks = args.n_blocks[0]
    n_neurons_body = args.n_neurons_body
    n_layers_head = args.n_layers_head
    n_neurons_head = args.n_neurons_head

    device = args.device
    batch_size = args.batch_size

    n_epochs = args.n_epochs
    lr = args.learning_rate
    no_schedule = args.no_lr_schedule
    ancillary = args.no_ancillary

    if len(n_epochs) == 1:
        n_epochs = n_epochs * len(lr)
    if len(lr) == 1:
        lr = lr * len(n_epochs)

    if args.normalizer is None:
        normalizer = get_normalizer(sensors.GMI)
    else:
        normalizer = Normalizer.load(args.normalizer)

    #
    # Load training data.
    #

    dataset_factory = GPROF_NN_HR_Dataset
    normalizer_path = GPROF_NN_DATA_PATH / f"normalizer_gmi.pckl"
    kwargs = {"batch_size": batch_size, "normalizer": normalizer, "augment": True}

    n_workers = args.n_processes_train
    training_data = DataFolder(
        training_data,
        dataset_factory,
        queue_size=256,
        kwargs=kwargs,
        n_workers=n_workers
    )

    if args.no_validation:
        validation_data = None
    else:
        kwargs = {
            "batch_size": 4 * batch_size,
            "normalizer": normalizer,
            "augment": False,
        }
        n_workers = args.n_processes_val
        validation_data = DataFolder(
            validation_data,
            dataset_factory,
            queue_size=256,
            kwargs=kwargs,
            n_workers=n_workers
        )

    ###############################################################################
    # Prepare in- and output.
    ###############################################################################

    #
    # Create neural network model
    #

    if Path(output).exists():
        try:
            xrnn = QRNN.load(output)
            LOGGER.info(f"Continuing training of existing model {output}.")
        except Exception:
            xrnn = None
    else:
        xrnn = None

    if xrnn is None:
        transformation = {t: LogLinear() for t in HR_TARGETS}
        transformation["latent_heat"] = None
        xrnn = GPROF_NN_HR(
            n_blocks,
            n_neurons_body,
            n_layers_head,
            n_neurons_head,
            transformation=transformation,
        )

    model = xrnn.model
    xrnn.normalizer = normalizer
    xrnn.configuration = configuration
    xrnn.sensor = "GMI"

    ###############################################################################
    # Run the training.
    ###############################################################################

    n_epochs_tot = sum(n_epochs)
    logger = TensorBoardLogger(n_epochs_tot)
    logger.set_attributes(
        {
            "n_blocks": n_blocks,
            "n_neurons_body": n_neurons_body,
            "n_layers_head": n_layers_head,
            "n_neurons_head": n_neurons_head,
            "optimizer": "adam",
        }
    )

    metrics = ["MeanSquaredError", "Bias", "CalibrationPlot", "CRPS"]
    scatter_plot = ScatterPlot()
    metrics.append(scatter_plot)

    for n, r in zip(n_epochs, lr):
        LOGGER.info(f"Starting training for {n} epochs with learning rate {r}")
        optimizer = optim.Adam(model.parameters(), lr=r)
        if no_schedule:
            scheduler = None
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n)
        xrnn.train(
            training_data=training_data,
            validation_data=validation_data,
            n_epochs=n,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
            metrics=metrics,
            device=device,
            mask=-9999,
        )
        LOGGER.info(f"Saving training network to {output}.")
        xrnn.save(output)
