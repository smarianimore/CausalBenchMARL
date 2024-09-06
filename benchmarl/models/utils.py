import itertools
from typing import Dict, Tuple, List, Any, Union
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import networkx as nx
import numpy as np
import pandas as pd
import psutil
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import CausalInference
from pgmpy.models import BayesianNetwork
import os
import logging

from torch import Tensor

from benchmarl.models._labels import LABEL_kind_group_var, LABEL_reward_action_values, LABEL_discrete_intervals, \
    LABEL_grouped_features, LABEL_value_group_var
from tqdm import tqdm


def list_to_graph(graph: list) -> nx.DiGraph:
    # Create a new directed graph
    dg = nx.DiGraph()

    # Add edges to the directed graph
    for cause, effect in graph:
        dg.add_edge(cause, effect)

    return dg


" ******************************************************************************************************************** "

def values_to_bins(values: List[float], intervals: List[float]) -> List[int]:
    # Sort intervals to ensure they are in ascending order
    intervals = sorted(intervals)

    # Initialize the list to store the bin index for each value
    new_values = []

    # Iterate through each value and determine its bin
    for value in values:
        for i in range(len(intervals) - 1):
            if intervals[i] <= value < intervals[i + 1]:
                new_values.append(i)
                break
        # To handle the case where the value is exactly equal to the last interval's end
        if value == intervals[-1]:
            new_values.append(len(intervals) - 2)

    return new_values


def discretize_value(value: int | float | Tensor, intervals: List) -> int | float:
    if isinstance(value, Tensor):
        value = value.to('cpu')
    idx = np.digitize(value, intervals, right=False)
    if idx == 0:
        return intervals[0]
    elif idx >= len(intervals):
        return intervals[-1]
    else:
        if abs(value - intervals[idx - 1]) <= abs(value - intervals[idx]):
            return intervals[idx - 1]
        else:
            return intervals[idx]


def _create_intervals(min_val: int | float, max_val: int | float, n_intervals: int, scale='linear') -> List:
    if scale == 'exponential':
        # Generate n_intervals points using exponential scaling
        intervals = np.logspace(0, 1, n_intervals, base=10) - 1
        intervals = intervals / (10 - 1)  # Normalize to range 0-1
        intervals = min_val + (max_val - min_val) * intervals
    elif scale == 'linear':
        intervals = np.linspace(min_val, max_val, n_intervals)
    else:
        raise ValueError("Unsupported scale type. Use 'exponential' or 'linear'.")

    return list(intervals)


def discretize_dataframe(df: pd.DataFrame, n_bins: int = 50, scale='linear', not_discretize_these: List = None):
    discrete_df = df.copy()
    variable_discrete_intervals = {}
    for column in df.columns:
        if column not in not_discretize_these:
            min_value = df[column].min()
            max_value = df[column].max()
            intervals = _create_intervals(min_value, max_value, n_bins, scale)
            variable_discrete_intervals[column] = intervals
            discrete_df[column] = df[column].apply(lambda x: discretize_value(x, intervals))
            # discrete_df[column] = np.vectorize(lambda x: intervals[_discretize_value(x, intervals)])(df[column])
    return discrete_df, variable_discrete_intervals


def group_row_variables(input_obs: Union[Dict, List], variable_columns: list, N: int = 1) -> Dict:
    obs = input_obs.copy()

    if isinstance(obs, list):
        row_dict = {i: obs[i] for i in variable_columns}
        is_list = True
    else:
        row_dict = {col: obs[col] for col in variable_columns}
        is_list = False

    sorted_variables = sorted(row_dict.items(), key=lambda x: x[1], reverse=True)[:N]
    obs_grouped = {}

    for i, (variable_name, variable_value) in enumerate(sorted_variables):
        try:
            variable_number = ''.join(filter(str.isdigit, str(variable_name)))
            obs_grouped[f'{LABEL_kind_group_var}_{i}'] = int(variable_number)
            obs_grouped[f'{LABEL_value_group_var}_{i}'] = variable_value
            # Remove the grouped part from the original observation
            if not is_list:
                del obs[variable_name]
            else:
                obs[variable_columns[i]] = None
        except (IndexError, ValueError):
            obs_grouped[f'{LABEL_kind_group_var}_{i}'] = None
            obs_grouped[f'{LABEL_value_group_var}_{i}'] = None

    # Add empty groups if N is greater than the number of sorted variables
    for i in range(len(sorted_variables), N):
        obs_grouped[f'{LABEL_kind_group_var}_{i}'] = None
        obs_grouped[f'{LABEL_value_group_var}_{i}'] = None

    # Remove the variable_columns from the original observation
    if not is_list:
        for col in variable_columns:
            if col in obs:
                del obs[col]

    # Combine the remaining original observation and grouped parts
    combined_obs = {**obs, **obs_grouped}

    return combined_obs


def _navigation_inverse_approximation(input_obs: Dict, **kwargs) -> Dict:
    n_groups, features_group = kwargs[LABEL_grouped_features]  # 2, [obs4-ob5-...]
    obs_grouped = group_row_variables(input_obs, features_group, n_groups)
    discrete_intervals = kwargs[LABEL_discrete_intervals]
    final_obs = {key: discretize_value(value, discrete_intervals[key]) for key, value in obs_grouped.items()}

    return final_obs


def _discovery_inverse_approximation(input_obs: Dict, **kwargs) -> Dict:
    n_groups, features_group = kwargs[LABEL_grouped_features]  # 2, [obs4-ob5-...]
    obs_grouped = group_row_variables(input_obs, features_group, n_groups)

    discrete_intervals = kwargs[LABEL_discrete_intervals]
    final_obs = {key: discretize_value(value, discrete_intervals[key]) for key, value in obs_grouped.items()}

    return final_obs


def _flocking_inverse_approximation(input_obs: Dict, **kwargs) -> Dict:
    n_groups, features_group = kwargs[LABEL_grouped_features]  # 2, [obs4-ob5-...]
    obs_grouped = group_row_variables(input_obs, features_group, n_groups)
    discrete_intervals = kwargs[LABEL_discrete_intervals]
    final_obs = {key: discretize_value(value, discrete_intervals[key]) for key, value in obs_grouped.items()}
    return final_obs

def _give_way_inverse_approximation(input_obs: Dict, **kwargs) -> Dict:
    discrete_intervals = kwargs[LABEL_discrete_intervals]
    final_obs = {key: discretize_value(value, discrete_intervals[key]) for key, value in input_obs.items()}
    return final_obs


def inverse_approximation_function(task: str):
    if task == 'navigation':
        return _navigation_inverse_approximation
    elif task == 'discovery':
        return _discovery_inverse_approximation
    elif task == 'flocking':
        return _flocking_inverse_approximation
    elif task == 'give_way':
        return _give_way_inverse_approximation
    # TODO: others
    else:
        raise NotImplementedError("The inverse approximation function for this task has not been implemented")

def dict_to_bn(model_data) -> BayesianNetwork:
    model = BayesianNetwork()
    model.add_nodes_from(model_data["nodes"])
    model.add_edges_from(model_data["edges"])

    for variable, cpd_data in model_data["cpds"].items():
        variable_card = cpd_data["variable_card"]
        evidence_card = cpd_data["evidence_card"]

        values = np.array(cpd_data["values"])
        if evidence_card:
            values = values.reshape(variable_card, np.prod(evidence_card))
        else:
            values = values.reshape(variable_card, 1)

        cpd = TabularCPD(
            variable=cpd_data["variable"],
            variable_card=variable_card,
            values=values.tolist(),
            evidence=cpd_data["evidence"],
            evidence_card=evidence_card,
            state_names=cpd_data["state_names"]
        )
        model.add_cpds(cpd)

    model.check_model()
    return model

def extract_intervals_from_bn(model: BayesianNetwork):
    intervals_dict = {}
    for node in model.nodes():
        cpd = model.get_cpds(node)
        if cpd:
            # Assuming discrete nodes with states
            intervals_dict[node] = cpd.state_names[node]
    return intervals_dict

" ******************************************************************************************************************** "
def check_values_in_states(known_states, observation, evidence):
    not_in_observation = {}
    not_in_evidence = {}

    for state, values in known_states.items():
        obs_value = observation.get(state, None)
        evid_value = evidence.get(state, None)

        if obs_value is not None and obs_value not in values:
            print('*** ERROR ****')
            print(state)
            print(values)
            print(obs_value)
            not_in_observation[state] = obs_value

        if evid_value is not None and evid_value not in values:
            print('*** ERROR ****')
            print(state)
            print(values)
            print(evid_value)
            not_in_evidence[state] = evid_value

    if not_in_observation != {}:
        print("Values not in observation: ", not_in_observation)

    if not_in_evidence != {}:
        print("\nValues not in evidence: ", not_in_evidence)

" ******************************************************************************************************************** "


class SingleCausalInference:
    def __init__(self, df: pd.DataFrame, causal_graph: nx.DiGraph, dict_init_cbn: Dict = None):
        self.df = df
        self.causal_graph = causal_graph
        self.features = self.causal_graph.nodes

        if dict_init_cbn is None and (df is None and causal_graph is None):
            raise ImportError('dataframe - causal graph - bayesian network are None')

        if dict_init_cbn is None:
            self.cbn = BayesianNetwork()
            self.cbn.add_edges_from(ebunch=self.causal_graph.edges())
            self.cbn.fit(self.df, estimator=MaximumLikelihoodEstimator)
        else:
            self.cbn = dict_to_bn(dict_init_cbn)
        del dict_init_cbn
        assert self.cbn.check_model()

        self.ci = CausalInference(self.cbn)

    def return_cbn(self) -> BayesianNetwork:
        return self.cbn

    def return_discrete_intervals_bn(self):
        intervals_dict = {}
        for node in self.cbn.nodes():
            cpd = self.cbn.get_cpds(node)
            if cpd:
                # Assuming discrete nodes with states
                intervals_dict[node] = cpd.state_names[node]
        return intervals_dict

    def infer(self, input_dict_do: Dict, target_variable: str, evidence=None, adjustment_set=None) -> Dict:
        # print(f'infer: {input_dict_do} - {target_variable}')

        # Ensure the target variable is not in the evidence
        input_dict_do_ok = {k: v for k, v in input_dict_do.items() if k != target_variable}

        # print(f'Cleaned input (evidence): {input_dict_do_ok}')
        # print(f'Target variable: {target_variable}')

        if adjustment_set is None:
            # Compute an adjustment set if not provided
            do_vars = [var for var, state in input_dict_do_ok.items()]
            adjustment_set = set(
                itertools.chain(*[self.causal_graph.predecessors(var) for var in do_vars])
            )
            # print(f'Computed adjustment set: {adjustment_set}')
        else:
            # print(f'Provided adjustment set: {adjustment_set}')
            pass

        # Ensure target variable is not part of the adjustment set
        adjustment_set.discard(target_variable)
        query_result = self.ci.query(
            variables=[target_variable],
            do=input_dict_do_ok,
            evidence=input_dict_do_ok if evidence is None else evidence,
            adjustment_set=adjustment_set,
            show_progress=False
        )
        # print(f'Query result: {query_result}')

        # Convert DiscreteFactor to a dictionary
        result_dict = {str(state): float(query_result.values[idx]) for idx, state in
                       enumerate(query_result.state_names[target_variable])}
        # print(f'Result distributions: {result_dict}')

        return result_dict

    @staticmethod
    def _check_states(input_dict_do, evidence, adjustment_set):
        def is_numeric(value):
            try:
                # Convert value to a numpy array and check for NaN or infinite values
                value = np.array(value, dtype=float)
                return np.isnan(value).any() or np.isinf(value).any()
            except (ValueError, TypeError):
                return False

        def check_for_nan_or_infinite(data):
            if isinstance(data, dict):
                for key, value in data.items():
                    if is_numeric(value):
                        print(f"Warning: {key} contains NaN or infinite values.")
                        return True
            elif isinstance(data, set):
                for value in data:
                    if is_numeric(value):
                        print(f"Warning: Set contains NaN or infinite values: {value}")
                        return True
            else:
                print(f"Unsupported data type: {type(data)}")
                return True  # Return True to signal an issue if data type is unsupported
            return False

        # Check the input data
        if check_for_nan_or_infinite(input_dict_do):
            raise ValueError("Input data contains NaN or infinite values.")
        if evidence is not None and check_for_nan_or_infinite(evidence):
            raise ValueError("Evidence data contains NaN or infinite values.")
        if adjustment_set is not None and check_for_nan_or_infinite(adjustment_set):
            raise ValueError("Adjustment set data contains NaN or infinite values.")


class CausalInferenceForRL:
    def __init__(self, online: bool, df_train: pd.DataFrame, causal_graph: nx.DiGraph,
                 bn_dict: Dict = None, causal_table: pd.DataFrame = None,
                 obs_train_to_test=None, grouped_features: Tuple = None):

        self.online = online

        self.df_train = df_train
        self.causal_graph = causal_graph
        self.bn_dict = bn_dict

        self.obs_train_to_test = obs_train_to_test

        del df_train, causal_graph, bn_dict

        self.ci = SingleCausalInference(self.df_train, self.causal_graph, self.bn_dict)

        self.discrete_intervals_bn = self.ci.return_discrete_intervals_bn()

        self.grouped_features = grouped_features

        self.reward_variable = [s for s in self.df_train.columns.to_list() if 'reward' in s][0]
        self.reward_values = self.df_train[self.reward_variable].unique().tolist()
        self.action_variable = [s for s in self.df_train.columns.to_list() if 'action' in s][0]
        self.action_values = self.df_train[self.action_variable].unique().tolist()

        self.causal_table = causal_table

    def _single_query(self, obs: Dict) -> Dict:
        def get_action_distribution(reward_value):
            # Create a new evidence dictionary by merging obs with the current reward_value
            evidence = {**obs, f'{self.reward_variable}': reward_value}
            # Check the values in the states
            check_values_in_states(self.ci.cbn.states, obs, evidence)
            # Infer the action distribution based on the evidence
            return self.ci.infer(obs, self.action_variable, evidence)

        # Construct the result dictionary using a dictionary comprehension
        reward_actions_values = {
            reward_value: get_action_distribution(reward_value)
            for reward_value in self.reward_values
        }

        return reward_actions_values

    def _compute_reward_action_values(self, input_obs: Dict) -> Dict:
        if self.obs_train_to_test is not None:
            kwargs = {}
            kwargs[LABEL_discrete_intervals] = self.discrete_intervals_bn
            kwargs[LABEL_grouped_features] = self.grouped_features

            obs = self.obs_train_to_test(input_obs, **kwargs)
        else:
            obs = input_obs

        reward_action_values = self._single_query(obs)

        row_result = obs.copy()
        row_result[f'{LABEL_reward_action_values}'] = reward_action_values

        return row_result

    def return_reward_action_values(self, input_obs: Dict) -> Dict:
        dict_input_and_rav = self._compute_reward_action_values(input_obs)
        reward_action_values = dict_input_and_rav[LABEL_reward_action_values]

        return reward_action_values


    def _process_combination(self, combination: Dict) -> Dict:
        # initial_time = time.time()
        try:
            reward_action_values = self._single_query(combination)
        except Exception as e:
            raise ValueError(e)

        return reward_action_values



