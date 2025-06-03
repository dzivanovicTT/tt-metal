# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.experimental.detr3d.ttnn.common import TtnnConv2D


class TttnnSharedMLP:
    def __init__(self, module, parameters, device):
        self.device = device
        self.parameters = parameters
        self.conv1 = TtnnConv2D(module, parameters, device, activation="relu")
        self.conv2 = TtnnConv2D(module, parameters, device, activation="relu")
        self.conv3 = TtnnConv2D(module, parameters, device, activation="relu")

    def __call__(self, features):
        features = self.conv1(features)
        features = self.conv1(features)
        features = self.conv1(features)
        return features
