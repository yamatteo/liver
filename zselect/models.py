from torch import Tensor, nn


class ZPredict(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(40, 40),
            nn.ReLU(True),
            nn.Linear(40, 40),
            nn.ReLU(True),
            nn.Linear(40, 20),
            nn.ReLU(True),
            nn.Linear(20, 10),
            nn.ReLU(True),
            nn.Linear(10, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x).squeeze(-1)


def get_model(name: str) -> nn.Module:
    if name == "zoffset_v0":
        return ZPredict()
