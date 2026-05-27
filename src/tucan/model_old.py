import torch.nn as nn

ACT_FUNC = {
    'relu': nn.ReLU(),
    'silu': nn.SiLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
}

class SturgeonSubmodel(nn.Module):

    def __init__(
            self,
            input_size,
            output_size,
            activation,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.input_size = input_size
        self.output_size = output_size
        self.activation = ACT_FUNC[activation]

        self.logit_layer = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(input_size, 256, bias=True),
            nn.Dropout(0.5),
            self.activation,
            nn.Linear(256, 128, bias=True),
            nn.Dropout(0.5),
            self.activation,
            nn.Linear(128, output_size, bias=False)
        )

    def forward(self, x):
        y = self.logit_layer(x)

        output = {}

        output['y'] = y

        return output


