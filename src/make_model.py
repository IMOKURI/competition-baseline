import timm
import torch.cuda.amp as amp
import torch.nn as nn


def make_model(c):
    if False:  # c.params.model_type == "xxx":
        pass
    else:
        model = BaseModel(c)

    if c.settings.multi_gpu:
        model = nn.DataParallel(model)
    return model


class BaseModel(nn.Module):
    def __init__(self, c, pretrained=True):
        super().__init__()
        self.amp = c.settings.amp
        self.model_name = c.params.model_name
        self.model = timm.create_model(
            self.model_name, pretrained=pretrained, num_classes=c.settings.n_class
        )

        if "convmixer" in self.model_name:
            self.head = nn.Linear(1000, c.settings.n_class)

    def forward(self, x):
        with amp.autocast(enabled=self.amp):
            x = self.model(x)
            if "convmixer" in self.model_name:
                x = self.head(x)

        return x
