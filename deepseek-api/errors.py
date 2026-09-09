class MissingApiKeyError(ValueError):
    def __init__(self, key_name):
        super().__init__(f"missing {key_name}")
        self.key_name = key_name
