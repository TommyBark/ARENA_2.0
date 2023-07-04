# %%
import os

os.environ["ACCELERATE_DISABLE_RICH"] = "1"
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import time
import sys
import re
import pandas as pd
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
from numpy.random import Generator
import plotly.express as px
import torch as t
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
import gym
from gym.envs.classic_control.cartpole import CartPoleEnv
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from gym.spaces import Discrete
import einops
import copy
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union, Callable, Optional
from jaxtyping import Float, Int, Bool
import wandb
from IPython.display import clear_output

# Make sure exercises are in the path
chapter = r"chapter2_rl"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part3_ppo"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from part1_intro_to_rl.utils import make_env
from part2_dqn.utils import set_global_seeds
from part2_dqn.solutions import Probe1, Probe2, Probe3, Probe4, Probe5
import part3_ppo.utils as utils
import part3_ppo.tests as tests
from plotly_utils import plot_cartpole_obs_and_dones

# Register our probes from last time
for idx, probe in enumerate([Probe1, Probe2, Probe3, Probe4, Probe5]):
    gym.envs.registration.register(id=f"Probe{idx+1}-v0", entry_point=probe)

Arr = np.ndarray

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"


# %%
@dataclass
class PPOArgs:
    exp_name: str = "PPO_Implementation"
    seed: int = 1
    cuda: bool = t.cuda.is_available()
    log_dir: str = "logs"
    use_wandb: bool = False
    wandb_project_name: str = "PPOCart"
    wandb_entity: str = None
    capture_video: bool = True
    env_id: str = "CartPole-v1"
    total_timesteps: int = 500000
    learning_rate: float = 0.00025
    num_envs: int = 4
    num_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    batches_per_epoch: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    batch_size: int = 512
    minibatch_size: int = 128

    def __post_init__(self):
        assert (
            self.batch_size % self.minibatch_size == 0
        ), "batch_size must be divisible by minibatch_size"
        self.total_epochs = self.total_timesteps // (self.num_steps * self.num_envs)
        self.total_training_steps = (
            self.total_epochs
            * self.batches_per_epoch
            * (self.batch_size // self.minibatch_size)
        )


args = PPOArgs(minibatch_size=256)
utils.arg_help(args)


# %%
def layer_init(layer: nn.Linear, std=np.sqrt(2), bias_const=0.0):
    t.nn.init.orthogonal_(layer.weight, std)
    t.nn.init.constant_(layer.bias, bias_const)
    return layer


def get_actor_and_critic(envs: gym.vector.SyncVectorEnv) -> Tuple[nn.Module, nn.Module]:
    """
    Returns (actor, critic), the networks used for PPO.
    """
    obs_shape = envs.single_observation_space.shape
    num_obs = np.array(obs_shape).prod()
    num_actions = envs.single_action_space.n

    actor = nn.Sequential(
        layer_init(nn.Linear(in_features=num_obs, out_features=64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, num_actions), std=0.01),
    )
    critic = nn.Sequential(
        layer_init(nn.Linear(in_features=num_obs, out_features=64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 1), std=1),
    )
    return actor, critic


tests.test_get_actor_and_critic(get_actor_and_critic)


# %%
def get_test_networks():
    actor = nn.Sequential(
        layer_init(nn.Linear(in_features=1, out_features=64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 2), std=0.01),
    )
    critic = nn.Sequential(
        layer_init(nn.Linear(in_features=1, out_features=64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 1), std=1),
    )

    return actor, critic


# %%
@t.inference_mode()
def compute_advantages(
    next_value: t.Tensor,
    next_done: t.Tensor,
    rewards: t.Tensor,
    values: t.Tensor,
    dones: t.Tensor,
    gamma: float,
    gae_lambda: float,
) -> t.Tensor:
    """Compute advantages using Generalized Advantage Estimation.
    next_value: shape (env,)
    next_done: shape (env,)
    rewards: shape (buffer_size, env)
    values: shape (buffer_size, env)
    dones: shape (buffer_size, env)
    Return: shape (buffer_size, env)
    """
    T = values.shape[0]
    A = t.zeros(values.shape)
    next_values = t.concat((values[1:, :], next_value.unsqueeze(0)), dim=0)
    next_dones = t.concat((dones[1:, :], next_done.unsqueeze(0)), dim=0)
    deltas = rewards + gamma * next_values * (1.0 - next_dones) - values

    A[-1, :] = deltas[-1, :]
    for i in reversed(range(1, T)):
        # delta = rewards[i] + gamma * values[i+1] - values[i]
        # A[i,:] += deltas[i,:] * (gamma * gae_lambda) ** (T - i + 1)
        A[i - 1, :] += (
            A[i, :] * (1 - dones[i]) * (gamma * gae_lambda) + deltas[i - 1, :]
        )
    return A


tests.test_compute_advantages(compute_advantages)


# %%
def minibatch_indexes(
    rng: Generator, batch_size: int, minibatch_size: int
) -> List[np.ndarray]:
    """
    Return a list of length (batch_size // minibatch_size) where each element is an array of indexes into the batch.

    Each index should appear exactly once.
    """
    assert batch_size % minibatch_size == 0
    # SOLUTION
    indices = rng.permutation(batch_size)
    indices = einops.rearrange(
        indices, "(mb_num mb_size) -> mb_num mb_size", mb_size=minibatch_size
    )
    return list(indices)


rng = np.random.default_rng(0)
batch_size = 6
minibatch_size = 2
indexes = minibatch_indexes(rng, batch_size, minibatch_size)

assert np.array(indexes).shape == (batch_size // minibatch_size, minibatch_size)
assert sorted(np.unique(indexes)) == [0, 1, 2, 3, 4, 5]
print("All tests in `test_minibatch_indexes` passed!")


# %%
@dataclass
class ReplayBufferSamples:
    """
    Samples from the replay buffer, converted to PyTorch for use in neural network training.
    """

    obs: Float[Tensor, "minibatch_size *obs_shape"]
    dones: Float[Tensor, "minibatch_size"]
    actions: Int[Tensor, "minibatch_size"]
    logprobs: Float[Tensor, "minibatch_size"]
    values: Float[Tensor, "minibatch_size"]
    advantages: Float[Tensor, "minibatch_size"]
    returns: Float[Tensor, "minibatch_size"]


class ReplayBuffer:
    """
    Contains buffer; has a method to sample from it to return a ReplayBufferSamples object.

    Needs to be initialized with the first obs, dones and values.
    """

    def __init__(self, args: PPOArgs, envs: gym.vector.SyncVectorEnv):
        """Defining all the attributes the buffer's methods will need to access."""
        self.rng = np.random.default_rng(args.seed)
        self.num_envs = envs.num_envs
        self.obs_shape = envs.single_observation_space.shape
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.batch_size = args.batch_size
        self.minibatch_size = args.minibatch_size
        self.num_steps = args.num_steps
        self.batches_per_epoch = args.batches_per_epoch
        self.experiences = []

    def add(
        self,
        obs: t.Tensor,
        actions: t.Tensor,
        rewards: t.Tensor,
        dones: t.Tensor,
        logprobs: t.Tensor,
        values: t.Tensor,
    ) -> None:
        """
        obs: shape (n_envs, *observation_shape)
            Observation before the action
        actions: shape (n_envs,)
            Action chosen by the agent
        rewards: shape (n_envs,)
            Reward after the action
        dones: shape (n_envs,)
            If True, the episode ended and was reset automatically
        logprobs: shape (n_envs,)
            Log probability of the action that was taken (according to old policy)
        values: shape (n_envs,)
            Values, estimated by the critic (according to old policy)
        """
        assert obs.shape == (self.num_envs, *self.obs_shape)
        assert actions.shape == (self.num_envs,)
        assert rewards.shape == (self.num_envs,)
        assert dones.shape == (self.num_envs,)
        assert logprobs.shape == (self.num_envs,)
        assert values.shape == (self.num_envs,)

        self.experiences.append((obs, dones, actions, logprobs, values, rewards))

    def get_minibatches(
        self, next_value: t.Tensor, next_done: t.Tensor
    ) -> List[ReplayBufferSamples]:
        minibatches = []

        # Turn all experiences to tensors on our device (we only want to do this once, not every time we add a new experience)
        obs, dones, actions, logprobs, values, rewards = [
            t.stack(arr).to(device) for arr in zip(*self.experiences)
        ]

        # Compute advantages and returns (then get a list of everything we'll need for our replay buffer samples)
        advantages = compute_advantages(
            next_value,
            next_done,
            rewards,
            values,
            dones.float(),
            self.gamma,
            self.gae_lambda,
        )
        returns = advantages + values
        replaybuffer_args = [obs, dones, actions, logprobs, values, advantages, returns]

        # We cycle through the entire buffer `self.batches_per_epoch` times
        for _ in range(self.batches_per_epoch):
            # Get random indices we'll use to generate our minibatches
            indices = minibatch_indexes(self.rng, self.batch_size, self.minibatch_size)

            # Get our new list of minibatches, and add them to the list
            for index in indices:
                minibatch = ReplayBufferSamples(
                    *[arg.flatten(0, 1)[index].to(device) for arg in replaybuffer_args]
                )
                minibatches.append(minibatch)

        # Reset the buffer
        self.experiences = []

        return minibatches


# %%
args = PPOArgs()
envs = gym.vector.SyncVectorEnv(
    [make_env("CartPole-v1", i, i, False, "test") for i in range(4)]
)
next_value = t.zeros(envs.num_envs).to(device)
next_done = t.zeros(envs.num_envs).to(device)
rb = ReplayBuffer(args, envs)
actions = t.zeros(envs.num_envs).int().to(device)
obs = envs.reset()

for i in range(args.num_steps):
    (next_obs, rewards, dones, infos) = envs.step(actions.cpu().numpy())
    real_next_obs = next_obs.copy()
    for i, done in enumerate(dones):
        if done:
            real_next_obs[i] = infos[i]["terminal_observation"]
    logprobs = values = t.zeros(envs.num_envs)
    rb.add(
        t.from_numpy(obs).to(device),
        actions,
        t.from_numpy(rewards).to(device),
        t.from_numpy(dones).to(device),
        logprobs,
        values,
    )
    obs = next_obs

obs, dones, actions, logprobs, values, rewards = [
    t.stack(arr).to(device) for arr in zip(*rb.experiences)
]

plot_cartpole_obs_and_dones(obs, dones, show_env_jumps=True)
# %%


class PPOAgent(nn.Module):
    critic: nn.Sequential
    actor: nn.Sequential

    def __init__(self, args: PPOArgs, envs: gym.vector.SyncVectorEnv):
        super().__init__()
        self.args = args
        self.envs = envs
        self.num_envs = envs.num_envs
        self.obs_shape = envs.single_observation_space.shape
        self.num_obs = np.array(self.obs_shape).prod()
        self.num_actions = envs.single_action_space.n

        # Keep track of global number of steps taken by agent
        self.steps = 0
        # Define actor and critic (using our previous methods)
        self.actor, self.critic = get_actor_and_critic(envs)

        # Define our first (obs, done, value), so we can start adding experiences to our replay buffer
        self.next_obs = t.tensor(self.envs.reset()).to(device)
        self.next_done = t.zeros(self.envs.num_envs).to(device, dtype=t.float)

        # Create our replay buffer
        self.rb = ReplayBuffer(args, envs)

    def play_step(self) -> List[dict]:
        """
        Carries out a single interaction step between the agent and the environment, and adds results to the replay buffer.
        """
        # SOLUTION
        obs = self.next_obs
        dones = self.next_done
        with t.inference_mode():
            values = self.critic(obs).flatten()
            logits = self.actor(obs)

        probs = Categorical(logits=logits)
        actions = probs.sample()
        logprobs = probs.log_prob(actions)
        next_obs, rewards, next_dones, infos = self.envs.step(actions.cpu().numpy())
        rewards = t.from_numpy(rewards).to(device)

        # (s_t, a_t, r_t+1, d_t, logpi(a_t|s_t), v(s_t))
        self.rb.add(obs, actions, rewards, dones, logprobs, values)

        self.next_obs = t.from_numpy(next_obs).to(device)
        self.next_done = t.from_numpy(next_dones).to(device, dtype=t.float)
        self.steps += self.num_envs

        return infos

    def get_minibatches(self) -> None:
        """
        Gets minibatches from the replay buffer.
        """
        with t.inference_mode():
            next_value = self.critic(self.next_obs).flatten()
        return self.rb.get_minibatches(next_value, self.next_done)


tests.test_ppo_agent(PPOAgent)


# %%
def calc_clipped_surrogate_objective(
    probs: Categorical,
    mb_action: Int[Tensor, "minibatch_size"],
    mb_advantages: Float[Tensor, "minibatch_size"],
    mb_logprobs: Float[Tensor, "minibatch_size"],
    clip_coef: float,
    eps: float = 1e-8,
) -> Float[Tensor, ""]:
    """Return the clipped surrogate objective, suitable for maximisation with gradient ascent.

    probs:
        a distribution containing the actor's unnormalized logits of shape (minibatch_size, num_actions)
    mb_action:
        what actions actions were taken in the sampled minibatch
    mb_advantages:
        advantages calculated from the sampled minibatch
    mb_logprobs:
        logprobs of the actions taken in the sampled minibatch (according to the old policy)
    clip_coef:
        amount of clipping, denoted by epsilon in Eq 7.
    eps:
        used to add to std dev of mb_advantages when normalizing (to avoid dividing by zero)
    """
    assert mb_action.shape == mb_advantages.shape == mb_logprobs.shape

    r_t = t.exp(probs.log_prob(mb_action) - mb_logprobs)
    normalized_advantages = (mb_advantages - mb_advantages.mean()) / (
        mb_advantages.std() + eps
    )
    clipped = t.clip(r_t, 1 - clip_coef, 1 + clip_coef)

    L_clip = t.minimum(r_t * normalized_advantages, clipped * normalized_advantages)
    return L_clip.mean()


tests.test_calc_clipped_surrogate_objective(calc_clipped_surrogate_objective)


# %%
def calc_value_function_loss(
    values: Float[Tensor, "minibatch_size"],
    mb_returns: Float[Tensor, "minibatch_size"],
    vf_coef: float,
) -> Float[Tensor, ""]:
    """Compute the value function portion of the loss function.

    values:
        the value function predictions for the sampled minibatch (using the updated critic network)
    mb_returns:
        the target for our updated critic network (computed as `advantages + values` from the old network)
    vf_coef:
        the coefficient for the value loss, which weights its contribution to the overall loss. Denoted by c_1 in the paper.
    """
    assert values.shape == mb_returns.shape

    L_vf = ((values - mb_returns) ** 2).mean() / 2
    return vf_coef * L_vf


tests.test_calc_value_function_loss(calc_value_function_loss)


# %%
def calc_entropy_bonus(probs: Categorical, ent_coef: float):
    """Return the entropy bonus term, suitable for gradient ascent.

    probs:
        the probability distribution for the current policy
    ent_coef:
        the coefficient for the entropy loss, which weights its contribution to the overall objective function. Denoted by c_2 in the paper.
    """
    return ent_coef * probs.entropy().mean()


tests.test_calc_entropy_bonus(calc_entropy_bonus)


# %%
class PPOScheduler:
    def __init__(
        self,
        optimizer: Optimizer,
        initial_lr: float,
        end_lr: float,
        total_training_steps: int,
    ):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.total_training_steps = total_training_steps
        self.n_step_calls = 0

    def step(self):
        """Implement linear learning rate decay so that after total_training_steps calls to step, the learning rate is end_lr."""
        step = (self.initial_lr - self.end_lr) / self.total_training_steps
        self.n_step_calls += 1
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.initial_lr - self.n_step_calls * step


def make_optimizer(
    agent: PPOAgent, total_training_steps: int, initial_lr: float, end_lr: float
) -> Tuple[optim.Adam, PPOScheduler]:
    """Return an appropriately configured Adam with its attached scheduler."""
    optimizer = optim.Adam(agent.parameters(), lr=initial_lr, eps=1e-5, maximize=True)
    scheduler = PPOScheduler(optimizer, initial_lr, end_lr, total_training_steps)
    return (optimizer, scheduler)


tests.test_ppo_scheduler(PPOScheduler)


# The agent.get_minibatches() function empties the list of experiences stored in the replay buffer, so you don't need to worry about doing this manually.
# You can log variables using wandb.log({"variable_name": variable_value}, steps=steps).
# The agent.play_step() method returns a list of dictionaries containing data about the current run. If the agent terminated at that step, then the dictionary will contain {"episode": {"l": episode_length, "r": episode_reward}}.
# You might want to create a separate method like _compute_ppo_objective(minibatch) which returns the objective function (to keep your code clean for the learning_phase method).
# For clipping gradients, you can use nn.utils.clip_grad_norm(parameters, max_norm) (see here for more details).
# %%
class PPOTrainer:
    def __init__(self, args: PPOArgs):
        super().__init__()
        self.args = args
        set_global_seeds(args.seed)
        self.run_name = (
            f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        )
        self.envs = gym.vector.SyncVectorEnv(
            [
                make_env(
                    args.env_id, args.seed + i, i, args.capture_video, self.run_name
                )
                for i in range(args.num_envs)
            ]
        )
        self.agent = PPOAgent(self.args, self.envs).to(device)
        self.optimizer, self.scheduler = make_optimizer(
            self.agent, self.args.total_training_steps, self.args.learning_rate, 0.0
        )
        if args.use_wandb:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                name=args.exp_name,
                config=args,
            )
            if args.capture_video:
                wandb.gym.monitor()

    def rollout_phase(self):
        """Should populate the replay buffer with new experiences."""
        # SOLUTION
        last_episode_len = None
        for step in range(self.args.num_steps):
            infos = self.agent.play_step()
            for info in infos:
                if "episode" in info.keys():
                    last_episode_len = info["episode"]["l"]
                    last_episode_return = info["episode"]["r"]
                    if args.use_wandb:
                        wandb.log(
                            {
                                "episode_length": last_episode_len,
                                "episode_return": last_episode_return,
                            },
                            step=self.agent.steps,
                        )
        # Return this for use in the progress bar
        return last_episode_len

    def learning_phase(self) -> None:
        """Should get minibatches and iterate through them (performing an optimizer step at each one)."""
        # SOLUTION
        minibatches = self.agent.get_minibatches()
        for minibatch in minibatches:
            objective_fn = self._compute_ppo_objective(minibatch)
            objective_fn.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), args.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

    def _compute_ppo_objective(
        self, minibatch: ReplayBufferSamples
    ) -> Float[Tensor, ""]:
        """Handles learning phase for a single minibatch. Returns objective function to be maximized."""
        logits = self.agent.actor(minibatch.obs)
        probs = Categorical(logits=logits)
        values = self.agent.critic(minibatch.obs).squeeze()

        clipped_surrogate_objective = calc_clipped_surrogate_objective(
            probs,
            minibatch.actions,
            minibatch.advantages,
            minibatch.logprobs,
            self.args.clip_coef,
        )
        value_loss = calc_value_function_loss(
            values, minibatch.returns, self.args.vf_coef
        )
        entropy_bonus = calc_entropy_bonus(probs, self.args.ent_coef)

        total_objective_function = (
            clipped_surrogate_objective - value_loss + entropy_bonus
        )

        with t.inference_mode():
            newlogprob = probs.log_prob(minibatch.actions)
            logratio = newlogprob - minibatch.logprobs
            ratio = logratio.exp()
            approx_kl = (ratio - 1 - logratio).mean().item()
            clipfracs = [
                ((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()
            ]
        if args.use_wandb:
            wandb.log(
                dict(
                    total_steps=self.agent.steps,
                    values=values.mean().item(),
                    learning_rate=self.scheduler.optimizer.param_groups[0]["lr"],
                    value_loss=value_loss.item(),
                    clipped_surrogate_objective=clipped_surrogate_objective.item(),
                    entropy=entropy_bonus.item(),
                    approx_kl=approx_kl,
                    clipfrac=np.mean(clipfracs),
                ),
                step=self.agent.steps,
            )

        return total_objective_function


def train(args: PPOArgs) -> PPOAgent:
    """Implements training loop, used like: agent = train(args)"""

    trainer = PPOTrainer(args)

    progress_bar = tqdm(range(args.total_epochs))

    for epoch in progress_bar:
        last_episode_len = trainer.rollout_phase()
        if last_episode_len is not None:
            progress_bar.set_description(
                f"Epoch {epoch:02}, Episode length: {last_episode_len}"
            )

        trainer.learning_phase()

    return trainer.agent


# %%
from gym.envs.classic_control.cartpole import CartPoleEnv


class EasyCart(CartPoleEnv):
    def step(self, action):
        (obs, reward, done, infos) = super().step(action)
        # YOUR CODE HERE - calculate new reward
        angle = obs[2]
        pos = obs[0]
        ang_velocity = obs[3]
        abs_angle = np.abs(angle)
        abs_position = np.abs(pos)

        max_position = 2.4
        max_angle = 0.2095
        angle_importance = 1.0

        linear_loss = (abs_angle / max_angle) * angle_importance + (
            abs_position / max_position
        ) * (1 - angle_importance)

        squared_loss = (abs_angle / max_angle) ** 2 * angle_importance + (
            abs_position / max_position
        ) ** 2 * (1 - angle_importance)

        # new_reward = 1 - squared_loss

        new_reward = np.clip(ang_velocity, -1, 100)
        for i, info in enumerate(infos):
            episode_len = info["episode"]["l"]
            if episode_len >= 500:
                done[i] = True
            else:
                done[i] = False
        return (obs, new_reward, done, infos)


gym.envs.registration.register(
    id="EasyCart-v0", entry_point=EasyCart, max_episode_steps=500
)
args = PPOArgs(env_id="EasyCart-v0", use_wandb=True)
# YOUR CODE HERE - train agent


id = f"EasyCart-v0-angle-imp-{1.0}"
gym.envs.registration.register(id=id, entry_point=EasyCart, max_episode_steps=500)
args = PPOArgs(env_id=id, use_wandb=True)
# YOUR CODE HERE - train agent
wandb.init()
args = PPOArgs(
    env_id=id,
    exp_name=f"test-{id}",
    total_timesteps=10000,
    learning_rate=0.001,
    capture_video=True,
    use_wandb=False,
)
args.wandb_project_name = "PPOCart_spin"
agent = train(args)
wandb.finish()

# %%
