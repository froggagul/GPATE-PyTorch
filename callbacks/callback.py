from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trainers import Trainer
    from logger import EpochDataStore
    from pytorch_lightning.loggers import Logger

class Callback():
    def __init__(self, store: "EpochDataStore", trainer: "Trainer", logger: "Logger") -> None:
        self.store = store
        self.trainer = trainer
        self.logger = logger
    
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("__call__ is not implemented here, use implementation of Callback")