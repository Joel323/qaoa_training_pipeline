#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Class to test the FromConfigParamsProvider."""

from qaoa_training_pipeline.training.from_config_provider import FromConfigParamsProvider

from ..training_pipeline_test_case import TrainingPipelineTestCase


class TestFixedAngleConjecture(TrainingPipelineTestCase):
    """Class to test the FromConfigParamsProvider parameter provider."""

    def setUp(self):  # pylint: disable=invalid-name
        """Setup the class."""
        self.params_provider = FromConfigParamsProvider([0.5, 0.5])

    def test_parameter_pass(self):
        """Test that provider can provide params0"""
        params0 = self.params_provider.provide_params()
        self.assertEqual(params0["optimized_params"], [0.5, 0.5])
