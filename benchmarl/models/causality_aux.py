
import numpy as np
import pandas as pd
import torch
import json

LABEL_reward_action_values = 'reward_action_values'


class CausalActionsFilter:
    def __init__(self, ci_online: bool, task: str, **kwargs):
        self.ci_online = ci_online
        self.task = task
        script_path = __file__.replace('causality_aux.py', 'causality_best')

        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.path_best = f'{script_path}\\{self.task}'

        self.last_obs_continuous = None

        causal_table = pd.read_pickle(f'{self.path_best}\\causal_table.pkl')
        with open(f'{self.path_best}/best_others.json', 'r') as file:
            info = json.load(file)

        self.indexes_to_discr = [
            str(s).replace('agent_0_obs_', '') for s in info['discrete_intervals']
            if 'reward' not in s and 'value' not in s and 'kind' not in s
        ]
        self.n_groups = None if info['grouped_features'][0] == 0 else info['grouped_features'][0]
        self.indexes_to_group = None if self.n_groups is None else [
            str(s).replace('agent_0_obs_', '') for s in info['grouped_features'][1]
        ]

        self._define_action_mask_inputs(causal_table)

    def _define_action_mask_inputs(self, causal_table: pd.DataFrame):
        def actions_mask_filter(reward_action_values, possible_rewards, possible_actions):
            old_min, old_max = min(possible_rewards), max(possible_rewards)
            averaged_mean_dict = {float(action): 0.0 for action in possible_actions}

            for reward_value, action_probs in reward_action_values.items():
                rescaled_value = (float(reward_value) - old_min) / (old_max - old_min)
                for action, prob in action_probs.items():
                    averaged_mean_dict[float(action)] += prob * rescaled_value

            num_entries = len(reward_action_values)
            averaged_mean_dict = {action: value / num_entries for action, value in averaged_mean_dict.items()}

            values = list(averaged_mean_dict.values())
            percentile_25 = np.percentile(values, 25)
            actions_mask = torch.tensor([0 if value <= percentile_25 else 1 for value in values], device=self.device)

            if actions_mask.sum() == 0:
                actions_mask = torch.tensor([0 if value <= 0 else 1 for value in values], device=self.device)

            return actions_mask

        def process_rav(values):
            possible_rewards = [float(key) for key in values[0].keys()]
            possible_actions = [float(key) for key in values[0][str(possible_rewards[0])].keys()]
            self.n_actions = len(possible_actions)
            return [actions_mask_filter(val, possible_rewards, possible_actions) for val in values]

        def df_to_tensors(df: pd.DataFrame):
            return {
                str(key).replace('agent_0_obs_', ''): torch.tensor(columns_values.values, dtype=torch.float32, device=self.device)
                if key != LABEL_reward_action_values else process_rav(columns_values.values)
                for key, columns_values in df.items()
            }

        self.dict_causal_table_tensors = df_to_tensors(causal_table)
        self.action_masks_from_causality = self.dict_causal_table_tensors[LABEL_reward_action_values]

        self.indexes_obs_in_causal_table = [
            causal_table.columns.get_loc(s) for s in causal_table.columns if s != LABEL_reward_action_values
        ]
        self.values_obs_in_causal_table = torch.stack([
            value for key, value in self.dict_causal_table_tensors.items() if key != LABEL_reward_action_values
        ])
        self.values_obs_in_causal_table_expanded = self.values_obs_in_causal_table.unsqueeze(0)

        ok_indexes_obs = []
        if any('agent_0_obs' in col for col in causal_table.columns):
            ok_indexes_obs.extend([int(str(s).replace('agent_0_obs_', '')) for s in causal_table.columns if 'agent_0_obs' in s])

        if any('kind' in col for col in causal_table.columns) or any('value' in col for col in causal_table.columns):
            start_group = len(self.indexes_to_discr)
            for n in range(self.n_groups):
                index = start_group + n * 2
                ok_indexes_obs.extend([index, index + 1])

        self.ok_indexes_obs = torch.tensor(ok_indexes_obs, device=self.device)

    def get_actions(self, multiple_observation: torch.Tensor):
        def validate_input(observation):
            if not isinstance(observation, torch.Tensor):
                raise ValueError('multiple_observation must be a tensor')
            return observation

        def calculate_delta_obs_continuous(current_obs, last_obs):
            return current_obs - last_obs

        def group_obs(obs):
            indexes_to_group = [int(i) for i in self.indexes_to_group]
            mask_ok = torch.ones(obs.shape[1], dtype=bool, device=self.device)
            mask_ok[indexes_to_group] = False

            values_ok = obs[:, mask_ok]
            values_to_group = obs[:, ~mask_ok]

            top_values, top_indices = torch.topk(values_to_group, self.n_groups, dim=1)
            index_value_pairs = torch.stack((top_indices.float(), top_values), dim=-1)

            return torch.cat((values_ok.flatten(1), index_value_pairs.flatten(1)), dim=1)

        def discretize_obs(obs):
            # Move obs_values to the correct device (GPU/CPU)
            obs_values = obs[:, self.ok_indexes_obs].unsqueeze(2).to(self.device)
            # Ensure both tensors are on the same device
            values_obs_in_causal_table_expanded = self.values_obs_in_causal_table_expanded.to(self.device)

            # Calculate differences
            differences = torch.abs(obs_values - values_obs_in_causal_table_expanded)

            # Ensure there are no invalid values like NaN or Inf
            if torch.isnan(differences).any() or torch.isinf(differences).any():
                raise ValueError("Differences tensor contains NaN or Inf values!")

            # Perform argmin operation
            closest_indices = torch.argmin(differences, dim=2)  # Keep this operation on the same device

            # Return the discretized observation
            discretized = torch.gather(
                values_obs_in_causal_table_expanded.expand(obs_values.size(0), -1, -1),
                2,
                closest_indices.unsqueeze(2)
            ).squeeze(2).to(self.device)

            return discretized

        def process_obs(obs):
            delta_obs_cont = calculate_delta_obs_continuous(obs, self.last_obs_continuous)
            grouped_obs = group_obs(delta_obs_cont) if self.n_groups else delta_obs_cont
            return get_action_mask(discretize_obs(grouped_obs))

        def get_action_mask(obs):
            # Adjust the comparison to slice the first 4 elements from values_obs_in_causal_table
            comparison = torch.stack([self.values_obs_in_causal_table[i][:obs.shape[1]] == obs[i] for i in
                                      range(len(self.indexes_obs_in_causal_table))], dim=0)

            valid_indices = comparison.all(dim=0).nonzero(as_tuple=True)[0]

            return self.action_masks_from_causality[valid_indices[0]] if len(valid_indices) > 0 else torch.ones(
                self.n_actions, device=self.device)

        # multiple_observation = multiple_observation.to(self.device)
        multiple_observation = validate_input(multiple_observation)
        num_envs, num_agents, _ = multiple_observation.shape

        multiple_observation_flatten = multiple_observation.view(-1, multiple_observation.size(-1))

        if self.last_obs_continuous is not None:
            action_masks = torch.stack([process_obs(obs_input) for obs_input in multiple_observation_flatten], dim=0)
        else:
            action_masks = torch.ones((num_envs * num_agents, self.n_actions), device=self.device)

        self.last_obs_continuous = multiple_observation_flatten
        return action_masks.view(num_envs, num_agents, -1).bool()


if __name__ == '__main__':
    online_ci = True
    causal_action_filter = CausalActionsFilter(online_ci, 'navigation')
