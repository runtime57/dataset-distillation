class NoOpWriter:
    """
    Writer with the same surface as experiment trackers, but no side effects.
    """

    def __init__(self, *args, **kwargs):
        self.step = 0
        self.mode = "train"

    def set_step(self, step, mode="train"):
        self.step = step
        self.mode = mode

    def add_checkpoint(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_scalars(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def add_audio(self, *args, **kwargs):
        pass

    def add_text(self, *args, **kwargs):
        pass

    def add_histogram(self, *args, **kwargs):
        pass

    def add_table(self, *args, **kwargs):
        pass
