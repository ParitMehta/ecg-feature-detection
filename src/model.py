#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: mehta
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PTB-XL baseline 1D-CNN."""
from torch import nn


class ECGNet(nn.Module):
    """Three conv-bn-relu-pool blocks + global average pool + linear head."""

    def __init__(self, n_leads: int = 12, n_classes: int = 5,
                 channels=(32, 64, 128)):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=7, padding=3),
                nn.BatchNorm1d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
            )

        c1, c2, c3 = channels
        self.features = nn.Sequential(
            block(n_leads, c1),
            block(c1, c2),
            block(c2, c3),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(c3, n_classes)

    def forward(self, x):          # x: (batch, 12, time)
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)        # logits, shape (batch, n_classes)