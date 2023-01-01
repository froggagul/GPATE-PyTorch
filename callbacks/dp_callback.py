from .callback import Callback
import numpy as np
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trainers import Trainer
    from logger import EpochDataStore
    from pytorch_lightning.loggers import Logger


class DPCallback(Callback):
    def __init__(self, store: "EpochDataStore", trainer: "Trainer", logger: "Logger") -> None:
        super().__init__(store, trainer, logger)
    
    def _compute_eps_from_delta(self, orders, rdp, delta):
        """Translates between RDP and (eps, delta)-DP.
        Args:
            orders: A list (or a scalar) of orders.
            rdp: A list of RDP guarantees (of the same length as orders).
            delta: Target delta.
        Returns:
            Pair of (eps, optimal_order).
        Raises:
            ValueError: If input is malformed.
        """
        if len(orders) != len(rdp):
            raise ValueError("Input lists must have the same length.")
        eps = np.array(rdp) - math.log(delta) / (np.array(orders) - 1)
        idx_opt = np.argmin(eps)
        return eps[idx_opt], orders[idx_opt]


    def __call__(self):
        orders = self.store.get('orders')
        rdp_counter = self.store.get('rdp_counter')
        dp_delta = self.store.get('dp_delta')
        eps, order = self._compute_eps_from_delta(orders, rdp_counter, dp_delta)

        # TODO - more kind way to get config of model params
        print(eps)
        if eps > self.trainer.model.max_eps:
            self.trainer.call_end_of_loop()
        
        # TODO - log order and eps with logger