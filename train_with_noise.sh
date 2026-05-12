#!/bin/bash
# Train ResNet-50 with different noise

python -m models.train_with_noise 0.50

python -m models.train_with_noise 0.75

python -m models.train_with_noise 1.00
