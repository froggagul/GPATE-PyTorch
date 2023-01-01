class EpochDataStore():
    def __init__(self) -> None:
        self.status_current = {}

    def log(self, key: str, value, epoch: int = None):
        prev_value = self.status_current.get(key, None)
        
        if prev_value is not None:
            if isinstance(prev_value, list) and isinstance(value, list):
                new_value = prev_value + value
            else:
                new_value = value
        else:
            new_value = value

        self.status_current[key] = new_value

    def get(self, key, default_value = None):
        return self.status_current.get(key, default_value)

    def initialize_status_current(self):
        self.status_current = {}

    def step_epoch(self):
        self.initialize_status_current()
