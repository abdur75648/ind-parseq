import os,shutil, torch
from pathlib import Path
from omegaconf import DictConfig, open_dict
import hydra
from hydra.core.hydra_config import HydraConfig

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.model_summary import summarize

from strhub.data.module import SceneTextDataModule
from strhub.models.base import BaseSystem
from strhub.models.utils import create_model

@hydra.main(config_path='configs', config_name='main', version_base='1.2')
def main(config: DictConfig):
    trainer_strategy = None
    with open_dict(config):
        # Resolve absolute path to data.root_dir
        config.data.root_dir = hydra.utils.to_absolute_path(config.data.root_dir)
        # Special handling for GPU-affected config
        #gpus = config.trainer.get('gpus', 0)
        devices = config.trainer.get('devices', [])
        gpus = len(devices)
        print("Gpus: ", gpus)
        if gpus:
            # Use mixed-precision training
            config.trainer.precision = 16
        if gpus > 1:
            # Use DDP
            config.trainer.strategy = 'ddp'
            # DDP optimizations
            trainer_strategy = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)
            # Scale steps-based config
            config.trainer.val_check_interval //= gpus
            if config.trainer.get('max_steps', -1) > 0:
                config.trainer.max_steps //= gpus

    # Special handling for PARseq
    if config.model.get('perm_mirrored', False):
        assert config.model.perm_num % 2 == 0, 'perm_num should be even if perm_mirrored = True'

    # If specified, use pretrained weights to initialize the model
    if config.pretrained is not None:
        print("\n\nLoading pretrained model : ", str(config.pretrained)," ...\n\n")
        model: BaseSystem = create_model(config.pretrained, True,config.learning_rate)
    else:
        print("Instantiating model from scratch : ")
        model: BaseSystem = hydra.utils.instantiate(config.model)
    print(summarize(model, max_depth=1 if model.hparams.name.startswith('parseq') else 2))
    
    if config.trainedmodel is not None:
        print("Loading trained model : ", str(config.trainedmodel))
        state_dict = torch.load(config.trainedmodel)['state_dict']
        if config.remove_head:
            print("Removing head")
            # remove head.weigth, head.bias and text_embed.embedding.weight, as number of classes are different
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            state_dict.pop('text_embed.embedding.weight')
        else:
            print("Not removing head")
        model.load_state_dict(state_dict, strict=False)

    hp = model.hparams
    print("Model HPs: ",hp)

    datamodule: SceneTextDataModule = hydra.utils.instantiate(config.data)

    checkpoint = ModelCheckpoint(monitor='val_NED', mode='max', save_top_k=3, save_last=True, filename='{epoch}-{step}-{val_accuracy:.4f}-{val_NED:.4f}')
    swa = StochasticWeightAveraging(swa_epoch_start=0.75)
    cwd = HydraConfig.get().runtime.output_dir if config.ckpt_path is None else \
        str(Path(config.ckpt_path).parents[1].absolute())
    
    
    if config.pretrained is not None:
        config.ckpt_path = os.path.join(config.pretrained,"checkpoints","last.ckpt")
        print("Passing ckpt_path = ",os.path.join(config.pretrained,"checkpoints","last.ckpt")," to the trainer.fit()")
    trainer: Trainer = hydra.utils.instantiate(config.trainer, logger=TensorBoardLogger(cwd, '', '.'),
                                               strategy=trainer_strategy, enable_model_summary=False,
                                               callbacks=[checkpoint, swa])
    trainer.fit(model, datamodule=datamodule, ckpt_path=config.ckpt_path)

if __name__ == '__main__':
    main()
