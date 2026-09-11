import pytest

torch = pytest.importorskip("torch")
from torch import nn

from week_2_deep_learning.cnn_image_classifier import CNNImageClassifier
from week_2_deep_learning.hyperparameter_experimentation import ExperimentLogger, ExperimentResult
from week_2_deep_learning.pytorch_neural_network import FeedForwardNetwork, make_dataloader


def test_feed_forward_and_loader() -> None:
    model = FeedForwardNetwork(4, hidden_dim=8, output_dim=1)
    loader = make_dataloader(torch.ones(5, 4), torch.ones(5, 1), batch_size=2, shuffle=False)
    output = model(next(iter(loader))[0])
    assert output.shape == (2, 1)


def test_cnn_features_and_logits() -> None:
    model = CNNImageClassifier(num_classes=3)
    assert model(torch.randn(2, 1, 28, 28)).shape == (2, 3)
    assert model.extract_features(torch.randn(2, 1, 28, 28)).shape == (2, 64)


def test_experiment_logger() -> None:
    logger = ExperimentLogger()
    logger.log(ExperimentResult("AdamW", 1e-3, 2, 0.1))
    assert logger.results()[0].optimizer == "AdamW"
