#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Class inheriting from ParamsProvider used to provide initial QAOA angles via configuration """

from qaoa_training_pipeline.framework import ParamsProvider
from qaoa_training_pipeline.framework.param_result import ParamResult


class FromConfigParamsProvider(ParamsProvider):
    """Class for providing initial QAOA angles or parameters from a configuration file"""

    def __init__(self, params0: list):
        """Initialize the parameter provider.

        Args:
            params0: the initial parameters to pass to the next pipeline component
        """
        super().__init__()
        self._params0 = params0

    def provide_params(self) -> ParamResult:
        """Provide QAOA angles to the next element in the pipeline.

        Returns:
            ParamResult object containing the QAOA angles
        """
        return ParamResult(self._params0, 0, self, None)

    @classmethod
    def from_config(cls, config: dict) -> "FromConfigParamsProvider":
        """Create an instance of the parameter provider

        Args:
            config: Dictionary containing the initial parameters to initialize the class

        Returns:
            An instance of FromConfigParamsProvider.

        """
        return cls(config["params0"])
