#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A class that implements a parameter transfer method based on the fixed angle conjecture"""

from qaoa_training_pipeline.training.transfer_trainer import TransferTrainer
from qaoa_training_pipeline.pre_processing.feature_extraction import GraphFeatureExtractor
from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator
from qaoa_training_pipeline.training.data_loading import LoadFromJson

from pathlib import Path


class FixedAngleConjecture(TransferTrainer):

    def __init__(self, reps: int):
        PROJECT_ROOT = Path(__file__).resolve().parent
        super().__init__(
            LoadFromJson(PROJECT_ROOT / "data" / "fixed_angle_conjecture.json", nested=True),
            GraphFeatureExtractor(
                extract_num_nodes=False,
                extract_num_edges=False,
                extract_avg_node_degree=True,
                extract_avg_edge_weights=False,
                extract_standard_devs=False,
                extract_density=False,
                include_one_local=False,
            ),
            reps=reps,
        )

    @classmethod
    def from_config(cls, config: dict) -> "TransferTrainer":
        """Create a class from a config."""
        return cls(
            reps=config["reps"],
        )
