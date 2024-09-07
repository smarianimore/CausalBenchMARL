#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from benchmarl.algorithms import CausalIqlConfig
from benchmarl.benchmark import Benchmark
from benchmarl.environments import VmasTask
from benchmarl.experiment import ExperimentConfig
from benchmarl.models.causal_mlp import CausalMlpConfig

if __name__ == "__main__":

    # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
    experiment_config = ExperimentConfig.get_from_yaml()

    # Loads from "benchmarl/conf/task/vmas"
    tasks = [VmasTask.NAVIGATION.get_from_yaml()]

    # Loads from "benchmarl/conf/algorithm"
    algorithm_configs = [
        CausalIqlConfig.get_from_yaml(),
    ]

    # Loads from "benchmarl/conf/model/layers"
    model_config = CausalMlpConfig.get_from_yaml()
    critic_model_config = CausalMlpConfig.get_from_yaml()

    benchmark = Benchmark(
        algorithm_configs=algorithm_configs,
        tasks=tasks,
        seeds={0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        experiment_config=experiment_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
    )
    benchmark.run_sequential()
