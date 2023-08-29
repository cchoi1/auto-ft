import os
from .imagenet import ImageNet


class ImageNetSketch(ImageNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.train:
            self.populate_train()
        else:
            self.populate_test()

    def populate_train(self):
        pass

    def get_test_path(self):
        return os.path.join(self.location, 'sketch')

    def __str__(self):
        return "ImageNetSketch"
