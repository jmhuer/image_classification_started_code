import torch
import torchvision
from .utils import overrides

##all models use same classes
classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot',
           'jackhammer', 'siren', 'street_music']


class CNNClassifier(torch.nn.Module):
    def __init__(self, n_classes=len(classes), layers=[16, 32, 64, 128], n_input_channels=3, kernel_size=5):
        super().__init__()
        self.classes = classes
        L = []
        c = n_input_channels
        for l in layers:
            L.append(torch.nn.Conv2d(c, l, kernel_size, stride=2, padding=kernel_size // 2))
            L.append(torch.nn.ReLU())
            c = l
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, n_classes)

    def forward(self, x):
        return self.classifier(self.network(x).mean(dim=[2, 3]))

    def predict(self, image):
        '''takes one image torch tensor outputs class'''
        logits = self.forward(image[None].float())
        argmax = int(torch.argmax(logits))
        print(logits)
        return self.classes[argmax]


class LastLayer_Alexnet(torch.nn.Module):
    def __init__(self, n_classes=len(classes)):
        super().__init__()
        self.classes = classes
        self.network = torchvision.models.alexnet(pretrained=True)  ##first time it will download weights
        self.new_layer = torch.nn.Linear(4096, n_classes)
        self.network.classifier[6] = self.new_layer

    def forward(self, x):
        return self.network(x)

    def predict(self, image):
        '''takes one image torch tensor outputs class'''
        logits = self.forward(image[None].float())
        argmax = int(torch.argmax(logits))
        return self.classes[argmax]

    @overrides(torch.nn.Module)
    def parameters(self, recurse: bool = True):
        return self.new_layer.parameters()


def save_model(model, info):
    from torch import save
    from os import path
    # if isinstance(model, CNNClassifier):
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'saved_models/' + info + ".th"))
    # raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_weights(model, my_path=None):
    from torch import load
    r = model
    if my_path:
        r.load_state_dict(load(my_path, map_location='cpu'))
    return r
