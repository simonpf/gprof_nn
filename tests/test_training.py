"""
Tests for the gprof_nn.training module.
=======================================
"""
import pytest

from gprof_nn.training import load_inference_config


@pytest.mark.parametrize("model", [("1d", "gprof_nn_1d"), ("3d", "gprof_nn_3d")])
def test_load_inference_config(model, request):
    """
    Test loading the inference config for a give GPROF-NN retrieval model.
    """
    config, model_name = model
    model = request.getfixturevalue(model_name)
    output_config = model.output_config
    inference_config = load_inference_config(
        config, output_config, ancillary=True
    )
    assert inference_config is not None
