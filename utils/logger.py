import pytorch_lightning
import wandb

class Logger(pytorch_lightning.loggers.WandbLogger):
    def __init__(self, project, name, log_model = False, save_dir = None):
        super(Logger, self).__init__(name = name, project = project, log_model = log_model, save_dir = save_dir)