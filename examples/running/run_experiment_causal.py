#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from benchmarl.algorithms import CausalIqlConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.causal_mlp import CausalMlpConfig

if __name__ == "__main__":

    # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
    experiment_config = ExperimentConfig.get_from_yaml()

    # Loads from "benchmarl/conf/task/vmas/balance.yaml"
    task = VmasTask.NAVIGATION.get_from_yaml()

    # Loads from "benchmarl/conf/algorithm/mappo.yaml"0.
    algorithm_config = CausalIqlConfig.get_from_yaml()

    # Loads from "benchmarl/conf/model/layers/mlp.yaml"
    CausalMlpConfig.task = task
    model_config = CausalMlpConfig.get_from_yaml()
    critic_model_config = CausalMlpConfig.get_from_yaml()

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=0,
        config=experiment_config,
    )
    experiment.run()
