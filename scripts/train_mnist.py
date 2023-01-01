import os
import sys
sys.path.insert(0, os.getcwd())

import trainers

if __name__ == "__main__":
    trainer = trainers.Trainer()
    trainer.fit()