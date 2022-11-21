"""
python src/train.py -cat carpet -config configs/carpet.yaml -train_dir ../training/carpet/
"""

import shutil
import warnings
import argparse
from pathlib import Path

import torch
import yaml
from ignite.contrib import metrics
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.utils import make_grid
from torchvision import transforms
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping

from src.iou import IoU
from src.model import UFlow
from src.nfa import compute_log_nfa_anomaly_score
from src.datamodule import MVTecLightningDatamodule, mvtec_un_normalize, get_debug_images_paths
from src.callbacks import MyPrintingCallback, ModelCheckpointByAuROC, ModelCheckpointByIoU, ModelCheckpointByInterval

warnings.filterwarnings("ignore", category=UserWarning, message="Your val_dataloader has `shuffle=True`")
warnings.filterwarnings("ignore", category=UserWarning, message="Checkpoint directory .* exists and is not empty")

LOG = 0


class UFlowTrainer(LightningModule):

    def __init__(
        self,
        flow_model,
        category,
        learning_rate=1e-3,
        weight_decay=1e-7,
        log_every_n_epochs=25,
        save_debug_images_every=25,
        log_predefined_debug_images=True,
        log_n_images=20
    ):
        """

        @param flow_model:
        @param learning_rate:
        @param weight_decay:
        @param log_every_n_epochs:
        @param save_debug_images_every:
        @param log_n_images:
        """
        super(UFlowTrainer, self).__init__()
        self.model = flow_model
        self.lr = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.log_every_n_epochs = log_every_n_epochs
        self.save_debug_images_every = save_debug_images_every
        self.log_predefined_debug_images = log_predefined_debug_images
        self.log_n_images = log_n_images
        self.debug_img_size = 256
        self.debug_img_resizer = transforms.Compose([transforms.Resize(self.debug_img_size)])
        self.debug_images = get_debug_images_paths(category)

        # Metrics
        self.pixel_auroc = metrics.ROC_AUC()
        self.image_auroc = metrics.ROC_AUC()
        self.iou_lnfa0 = IoU(thresholds=[0])

        # Debug images
        self.test_images = None
        self.test_targets = None

    def step(self, batch, batch_idx):
        z, ljd = self.model(batch)

        # Compute loss
        lpz = torch.sum(torch.stack([0.5 * torch.sum(z_i ** 2, dim=(1, 2, 3)) for z_i in z], dim=0))
        flow_loss = torch.mean(lpz - ljd)

        return {"01_Train/Loss": flow_loss.detach(), "loss": flow_loss}

    def training_step(self, batch, batch_idx):
        losses = self.step(batch, batch_idx)
        self.log_dict({"loss": losses['loss']}, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return losses

    def training_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("02_Train/Loss", avg_loss, self.current_epoch)

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']
        self.logger.experiment.add_scalar("04_LearningRate", get_lr(self.optimizers()), self.current_epoch)

    def validation_step(self, batch, batch_idx):
        images, targets, paths = batch

        if self.current_epoch == 0:
            debug_images = self.debug_images
            # Keep predefined images to show in all different trainings always the same ones
            if self.log_predefined_debug_images and (len(debug_images) > 0):
                if batch_idx == 0:
                    self.test_images = torch.Tensor(len(debug_images), *images.shape[1:]).to(images.device)
                    self.test_targets = torch.Tensor(len(debug_images), *targets.shape[1:]).to(targets.device)
                for i, img_path in enumerate(paths):
                    try:
                        idx = [i for i, s in enumerate(debug_images) if s in img_path][0]
                        self.test_images[idx] = images[i]
                        self.test_targets[idx] = targets[i]
                    except IndexError:
                        pass
            # Or randomly sample a different set for each training
            else:
                if batch_idx == 0:
                    self.test_images = torch.Tensor().to(images.device)
                    self.test_targets = torch.Tensor().to(targets.device)
                n_to_keep = self.log_n_images - batch_idx * images.shape[0]
                if n_to_keep > 0:
                    self.test_images = torch.cat([self.test_images, images[:n_to_keep, ...]], dim=0)
                    self.test_targets = torch.cat([self.test_targets, targets[:n_to_keep, ...]], dim=0)

        # Update metrics
        if self.current_epoch % self.log_every_n_epochs == 0:
            images, targets, paths = batch
            z, _ = self.model(images)
            z = [zz.detach() for zz in z]

            # Pixel level metrics
            anomaly_score = 1 - self.model.get_probability(z, self.debug_img_size)
            lnfa = compute_log_nfa_anomaly_score(z, win_size=5, binomial_probability_thr=0.9, high_precision=False)
            resized_targets = 1 * (self.debug_img_resizer(targets) > 0.5)
            self.pixel_auroc.update((anomaly_score.ravel(), resized_targets.ravel()))
            self.iou_lnfa0.update(lnfa.detach().cpu(), resized_targets.cpu())

            # Image level metric
            image_targets = torch.IntTensor([0 if 'good' in p else 1 for p in paths])
            image_anomaly_score = torch.amax(anomaly_score, dim=(1, 2, 3))
            self.image_auroc.update((image_anomaly_score.ravel().cpu(), image_targets.ravel().cpu()))

    def validation_epoch_end(self, outputs) -> None:
        # Log metrics
        if self.current_epoch % self.log_every_n_epochs == 0:
            # Compute metrics
            pixel_auroc = float(self.pixel_auroc.compute())
            image_auroc = float(self.image_auroc.compute())
            pixel_iou = float(self.iou_lnfa0.compute().numpy())
            self.log_dict(
                {'pixel_auroc': pixel_auroc, 'image_auroc': image_auroc, 'iou': pixel_iou},
                on_step=False, on_epoch=True, prog_bar=False, logger=True
            )

            self.pixel_auroc.reset()
            self.image_auroc.reset()
            self.iou_lnfa0.reset()

            self.logger.experiment.add_scalar("03_ValidationMetrics/PixelAuROC", pixel_auroc, self.current_epoch)
            self.logger.experiment.add_scalar("03_ValidationMetrics/ImageAuROC", image_auroc, self.current_epoch)
            self.logger.experiment.add_scalar("03_ValidationMetrics/PixelIoU", pixel_iou, self.current_epoch)

        # Log example images
        if self.current_epoch % self.save_debug_images_every == 0:
            anomaly_maps_to_show = []
            for i in range(len(self.test_images)):
                # Forward
                batch = self.test_images[i:i+1, ...]
                outputs, _ = self.model(batch)
                # Compute probabilities
                anomaly_maps_to_show.append(1 - self.model.get_probability(outputs, self.debug_img_size))

            # Generate output probability images
            images_grid = make_grid(
                mvtec_un_normalize(self.debug_img_resizer(self.test_images)).to('cpu'),
                normalize=True, nrow=1, value_range=(0, 1)
            )
            labels_grid = make_grid(
                self.debug_img_resizer(self.test_targets).to('cpu'),
                normalize=True, nrow=1, value_range=(0, 1)
            )
            anomaly_maps_grid = make_grid(
                torch.cat(anomaly_maps_to_show, dim=0).to('cpu'),
                normalize=True, nrow=1, value_range=(0, 1)
            )
            to_show = torch.dstack((images_grid, labels_grid, anomaly_maps_grid))
            self.logger.experiment.add_image(
                f"Example images",
                to_show,
                self.current_epoch,
                dataformats="CHW"
            )

    def configure_optimizers(self):
        def get_total_number_of_iterations():
            try:
                self.trainer.reset_train_dataloader()
                number_of_training_examples = len(self.trainer.train_dataloader.dataset)
                batch_size = self.trainer.train_dataloader.loaders.batch_size
                drop_last = 1 * self.trainer.train_dataloader.loaders.drop_last
                iterations_per_epoch = number_of_training_examples // batch_size + 1 - drop_last
                total_iterations = iterations_per_epoch * (self.trainer.max_epochs - 1)
            except:
                total_iterations = 25000
            return total_iterations

        lr = self.lr

        # Optimizer
        optimizer = torch.optim.Adam(
            [{"params": self.parameters(), "initial_lr": lr}],
            lr=lr,
            weight_decay=self.weight_decay
        )

        # Scheduler for slowly reducing learning rate
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1., end_factor=0.4, total_iters=get_total_number_of_iterations()
        )
        return [optimizer], [scheduler]


def get_training_dir(base_dir, prefix="exp_"):
    out_path = Path(base_dir)
    previous_experiments = [int(l.stem.split('_')[1]) for l in out_path.glob(f'{prefix}*')]
    last_experiment = max(previous_experiments) if len(previous_experiments) > 0 else 0
    out_path = out_path / f"{prefix}{last_experiment + 1:04d}"
    out_path.mkdir(exist_ok=True, parents=True)
    return out_path


def train(args):
    config_path = f"configs/{args.category}.yaml" if args.config_path is None else args.config_path
    config = yaml.safe_load(open(config_path, "r"))

    # Model
    # ------------------------------------------------------------------------------------------------------------------
    uflow = UFlow(config['model']['input_size'], config['model']['flow_steps'], config['model']['backbone'])

    uflow_trainer = UFlowTrainer(
        uflow,
        args.category,
        config['trainer']['learning_rate'],
        config['trainer']['weight_decay'],
        config['trainer']['log_every_n_epochs'],
        config['trainer']['save_debug_images_every'],
        config['trainer']['log_predefined_debug_images'],
        config['trainer']['log_n_images']
    )

    # Data
    # ------------------------------------------------------------------------------------------------------------------
    datamodule = MVTecLightningDatamodule(
        data_dir=args.data,
        category=args.category,
        input_size=config['model']['input_size'],
        batch_train=config['trainer']['batch_train'],
        batch_test=config['trainer']['batch_val'],
        shuffle_test=True
    )

    # Train
    # ------------------------------------------------------------------------------------------------------------------
    training_dir = get_training_dir(Path(args.training_dir) / args.category)
    callbacks = [
        MyPrintingCallback(),
        ModelCheckpointByAuROC(training_dir),
        ModelCheckpointByIoU(training_dir),
        # ModelCheckpointByInterval(training_dir, config['trainer']['save_ckpt_every']),
        LearningRateMonitor('epoch'),
        EarlyStopping(
            monitor="pixel_auroc",
            mode="max",
            patience=20,
        ),
    ]
    logger = TensorBoardLogger(save_dir=str(training_dir / 'logs'), name='UFlow')
    shutil.copy(config_path, str(training_dir / "config.yaml"))

    trainer = Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=config['trainer']['epochs'] + 1,
        log_every_n_steps=10,
        callbacks=callbacks,
        logger=logger,
        num_sanity_val_steps=0,
    )

    trainer.fit(uflow_trainer, train_dataloaders=datamodule.train_dataloader(), val_dataloaders=datamodule.val_dataloader())


if __name__ == "__main__":
    # seed_everything(0)

    # Args
    # ------------------------------------------------------------------------------------------------------------------
    p = argparse.ArgumentParser()
    p.add_argument("-cat", "--category", type=str, required=True)
    p.add_argument("-config", "--config_path", default=None, type=str)
    p.add_argument("-data", "--data", default="data", type=str)
    p.add_argument("-train_dir", "--training_dir", default="training", type=str)
    cmd_args, _ = p.parse_known_args()

    # Execute
    # ------------------------------------------------------------------------------------------------------------------
    train(cmd_args)
