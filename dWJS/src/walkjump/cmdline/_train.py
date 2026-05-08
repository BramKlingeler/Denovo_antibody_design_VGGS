import dotenv
import hydra
import pytorch_lightning as pl
import wandb
import torch
from pytorch_lightning.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.trainer import Trainer

from walkjump.cmdline.utils import instantiate_callbacks

dotenv.load_dotenv(".env")


@hydra.main(version_base=None, config_path="../hydra_config", config_name="train")
def train(cfg: DictConfig) -> bool:
    log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)

    wandb.require("service")
    if rank_zero_only.rank == 0:
        print(OmegaConf.to_yaml(log_cfg))

    hydra.utils.instantiate(cfg.setup)

    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)

    if not cfg.dryrun:
        logger = hydra.utils.instantiate(cfg.logger)
    else:
        logger = None

    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, devices=[torch.cuda.current_device()])
    
    #trainer: Trainer = Trainer(**cfg.trainer, callbacks=callbacks, logger=logger, check_val_every_n_epoch=10) #, devices=[torch.cuda.current_device()]
    
    if rank_zero_only.rank == 0 and isinstance(trainer.logger, pl.loggers.WandbLogger):
        trainer.logger.experiment.config.update({"cfg": log_cfg})

    if not cfg.dryrun:
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
        trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    wandb.finish()
    return True
