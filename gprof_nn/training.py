"""
gprof_nn.training
=================

Implements training routines for the different stages of the
training of the GPROF-NN retrievals.
"""
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional, Union, Tuple, List

import numpy as np
import pytorch_lightning as pl
from quantnn import metrics
from quantnn.mrnn import MRNN, Quantiles
import torch
from torch.utils.data import DataLoader

from gprof_nn.definitions import MASKED_OUTPUT
from gprof_nn.models import GPROF_NN_3D_CONFIGS, GPROFNet3D
from gprof_nn.data.training_data import PretrainingDataset


@dataclass
class TrainingConfig:
    """
    A description of a training regime.
    """
    name: str
    n_epochs: int
    optimizer: str
    optimizer_kwargs: Optional[dict] = None
    scheduler: str = None
    scheduler_kwargs: Optional[dict] = None
    precision: str = "16-mixed"
    batch_size: int = 8
    accelerator: str = "cuda"
    data_loader_workers: int = 4
    minimum_lr: Optional[float] = None
    reuse_optimizer: bool = False
    stepwise_scheduling: bool = False


def parse_training_config(path: Union[str, Path]):
    """
    Parse a training config file.

    Args:
        path: Path pointing to the training config file.

    Return:
        A list 'TrainingConfig' objects representing the training
        passes to perform.
    """
    path = Path(path)
    parser = ConfigParser()
    parser.read(path)

    training_configs = []

    for section_name in parser.sections():

        sec = parser[section_name]

        n_epochs = sec.getint("n_epochs", 1)
        optimizer = sec.get("optimizer", "SGD")
        optimizer_kwargs = eval(sec.get("optimizer_kwargs", "{}"))
        scheduler = sec.get("scheduler", None)
        scheduler_kwargs = eval(sec.get("scheduler_kwargs", "{}"))
        precision = sec.get("precision", "16-mixed")
        batch_size = sec.getint("batch_size", 8)
        data_loader_workers = sec.getint("data_loader_workers", 8)
        minimum_lr = sec.getfloat("minimum_lr", None)
        reuse_optimizer = sec.getboolean("reuse_optimizer", False)
        stepwise_scheduling = sec.getboolean("stepwise_scheduling", False)

        training_configs.append(TrainingConfig(
            name=section_name,
            n_epochs=n_epochs,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            precision=precision,
            sample_rate=sample_rate,
            batch_size=batch_size,
            data_loader_workers=data_loader_workers,
            minimum_lr=minimum_lr,
            reuse_optimizer=reuse_optimizer,
            stepwise_scheduling=stepwise_scheduling
        ))


def get_optimizer_and_scheduler(
        training_config,
        model,
        previous_optimizer=None
):
    """
    Return torch optimizer, learning-rate scheduler and callback objects
    corresponding to this configuration.

    Args:
        training_config: A TrainingConfig object specifying training
            settings for one training stage.
        model: The model to be trained as a torch.nn.Module object.
        previous_optimizer: Optimizer from the previous stage in case
            it is reused.

    Return:
        A tuple ``(optimizer, scheduler, callbacks)`` containing a PyTorch
        optimizer object ``optimizer``, the corresponding LR scheduler
        ``scheduler`` and a list of callbacks.

    Raises:
        Value error if training configuration specifies to reuse the optimizer
        but 'previous_optimizer' is none.

    """
    if training_config.reuse_optimizer:
        if previous_optimizer is None:
            raise RuntimeError(
                "Training stage '{training_config.name}' has 'reuse_optimizer' "
                "set to 'True' but no previous optimizer is available."
            )
        optimizer = previous_optimizer

    else:
        optimizer_cls = getattr(torch.optim, training_config.optimizer)
        optimizer = optimizer_cls(
            model.parameters(),
            **training_config.optimizer_kwargs
        )

    scheduler = training_config.scheduler
    if scheduler is None:
        return optimizer, None, []

    if scheduler == "lr_search":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=2.0
        )
        callbacks = [
            ResetParameters(),
        ]
        return optimizer, scheduler, callbacks

    scheduler = getattr(torch.optim.lr_scheduler, training_config.scheduler)
    scheduler_kwargs = training_config.scheduler_kwargs
    if scheduler_kwargs is None:
        scheduler_kwargs = {}
    scheduler = scheduler(
        optimizer=optimizer,
        **scheduler_kwargs,
    )
    scheduler.stepwise = training_config.stepwise_scheduling

    if training_config.minimum_lr is not None:
        callbacks = [
            EarlyStopping(
                f"Learning rate",
                stopping_threshold=training_config.minimum_lr * 1.001,
                patience=training_config.n_epochs,
                verbose=True,
                strict=True
            )
        ]
    else:
        callbacks = []

    return optimizer, scheduler, callbacks


def create_data_loaders_pretraining(
        training_config: TrainingConfig,
        training_data_path: Path,
        validation_data_path: Optional[Path]
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create pytorch Dataloaders for training and validation data.

    Args:
        training_config: Dataclass specifying the training configuration,
            which defines how many processes to use for the data loading.
        training_data_path: The path pointing to the folder containing
            the training data.
        validation_data_path: The path pointing to the folder containing
            the validation data.
    """
    training_data = PretrainingDataset(
        training_data_path,
        normalize=True,
        ancillary_data=True,
        channel_corruption=0.2
    )
    training_loader = DataLoader(
        training_data,
        shuffle=True,
        batch_size=training_config.batch_size,
        num_workers=training_config.data_loader_workers,
        worker_init_fn=training_data.seed,
        pin_memory=True,
    )
    if validation_data_path is None:
        return training_loader, None

    validation_data =  PretrainingDataset(
        validation_data_path,
        normalize=True,
        ancillary_data=True,
        channel_corruption=0.2
    )
    validation_loader = DataLoader(
        validation_data,
        shuffle=False,
        batch_size=training_config.batch_size,
        num_workers=training_config.data_loader_workers,
        worker_init_fn=validation_data.seed,
        pin_memory=True,
    )
    return training_loader, validation_loader


def find_most_recent_checkpoint(path: Path, model_name: str) -> Path:
    """
    Find most recente Pytorch lightning checkpoint files.

    Args:
        path: A pathlib.Path object pointing to the folder containing the
            checkpoints.
        model_name: The model name as defined by the user.

    Return:
        If a checkpoint was found, returns a object pointing to the
        checkpoint file with the highest version number. Otherwise
        returns 'None'.
    """
    path = Path(path)

    checkpoint_files = list(path.glob(f"{model_name}*.ckpt"))
    if len(checkpoint_files) == 0:
        return None
    if len(checkpoint_files) == 1:
        return checkpoint_files[0]

    checkpoint_regexp = re.compile(rf"{model_name}(-v\d*)?.ckpt")
    versions = []
    for checkpoint_file in checkpoint_files:
        match = checkpoint_regexp.match(checkpoint_file.name)
        if match is None:
            return None
        if match.group(1) is None:
            versions.append(-1)
        else:
            versions.append(int(match.group(1)[2:]))
    ind = np.argmax(versions)
    return checkpoint_files[ind]


def compile_model(
        model_config: str,
        kind: str = "pretraining",
        base_model: Optional[Path] = None
) -> MRNN:
    """
    Compile quantnn.mrnn.MRNN model for the training.

    Args:
        model_config: Name of the model config.
        kind: The kind of training to be performed.
        base_model: Path to a pretrained model to initialize the model
            with.
    """
    config = GPROF_NN_3D_CONFIGS[model_config]
    if kind.lower() in ["pretraining", "pretrain"]:
        targets = {
            f"tbs_{ind}": (32,) for ind in range(15)
        }
    else:
        targets = {
            name: (16, 28) if name in PROFILE_NAMES else (32,)
            for name in ALL_TARGETS
        }

    model = GPROFNet3D(
        config.n_channels,
        config.n_blocks,
        targets=targets,
        ancillary_data=config.ancillary_data
    )

    losses = {
        name: Quantiles(np.linspace(0, 1, shape[0] + 2)[1:-1])
        for name, shape in targets.items()
    }

    return MRNN(losses=losses, model=model)


def run_pretraining(
        output_path: Union[Path, str],
        model_config: str,
        training_configs: List[TrainingConfig],
        training_data_path: Path,
        validation_data_path: Optional[Path] = None,
        continue_training: bool = False
):
    """
    Run pretraining for GPROF-NN model.

    Args:
        output_path: The path to which to write the trained model.
        model_config: The configuration of the base model. Should be one
            ['small', 'small_no_ancillary', 'large', 'large_no_ancillary']
            of for 'small' order 'large' model capacity and including
            ancillary data or not.
        training_configs: List of training configs specifying the training settings
            for all training passes to perform.
        validation_data: Optional path pointing to the validation data.
    """
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    mrnn = compile_model(model_config, "pretrain")
    model_name = f"gprof_nn_3d_pre_{model_config}"

    mtrcs = [
        metrics.Bias(),
        metrics.Correlation(),
        metrics.CRPS(),
        metrics.MeanSquaredError(),
    ]
    lightning_module = mrnn.lightning(
        mask=MASKED_OUTPUT,
        metrics=mtrcs,
        name=model_name,
        log_dir=output_path / "logs"
    )

    ckpt_path = None
    if continue_training:
        ckpt_path = find_most_recent_checkpoint(output_path, model_name)
        ckpt_data = torch.load(ckpt_path)
        stage = ckpt_data["stage"]
        lightning_module.stage = stage

    devices = None
    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.ModelCheckpoint(
            dirpath=output_path,
            filename=f"gprof_nn_{model_name}",
            verbose=True
        )
    ]

    all_optimizers = []
    all_schedulers = []
    all_callbacks = []
    opt_prev = None
    for stage_ind, training_config in enumerate(training_configs):
        opt_s, sch_s, cback_s = get_optimizer_and_scheduler(
            training_config,
            mrnn.model,
            previous_optimizer=opt_prev
        )
        opt_prev = opt_s
        all_optimizers.append(opt_s)
        all_schedulers.append(sch_s)
        all_callbacks.append(cback_s)

    lightning_module.optimizer = all_optimizers
    lightning_module.scheduler = all_schedulers


    for stage_ind, training_config in enumerate(training_configs):
        if stage_ind < lightning_module.stage:
            continue

        # Restore LR if optimizer is reused.
        if training_config.reuse_optimizer:
            if "lr" in training_config.optimizer_kwargs:
                optim = lightning_module.optimizer[stage_ind]
                lr = training_config.optimizer_kwargs["lr"]
                for group in optim.param_groups:
                    group["lr"] = lr

        stage_callbacks = callbacks + all_callbacks[stage_ind]
        training_loader, validation_loader = create_data_loaders_pretraining(
            training_config,
            training_data_path,
            validation_data_path
        )

        if training_config.accelerator in ["cuda", "gpu"]:
            devices = -1
        else:
            devices = 1
        lightning_module.stage_name = training_config.name

        trainer = pl.Trainer(
            default_root_dir=output_path,
            max_epochs=training_config.n_epochs,
            accelerator=training_config.accelerator,
            devices=devices,
            precision=training_config.precision,
            logger=lightning_module.tensorboard,
            callbacks=stage_callbacks,
            num_sanity_val_steps=0,
            #strategy=pl.strategies.DDPStrategy(find_unused_parameters=True),
        )
        trainer.fit(
            model=lightning_module,
            train_dataloaders=training_loader,
            val_dataloaders=validation_loader,
            ckpt_path=ckpt_path
        )
        mrnn.save(output_path / f"cimr_{model_name}.pckl")
        ckpt_path=None
