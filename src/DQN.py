import random
import typing as tp
from collections import namedtuple
from time import time

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim

BATCH_SIZE = 64

GAMMA = 0.999
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 500

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.capacity: int = capacity
        self.memory: tp.List[Transition] = []
        self.position: int = 0

    def push(self, *args: tp.Any) -> None:
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(*args))
        else:
            self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(
        self, batch_size: int
    ) -> tp.Union[tp.Sequence[Transition], tp.AbstractSet[Transition]]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class ExpSmoothedArr:
    def __init__(
        self, beta: float, init_arr: tp.Optional[tp.Sequence[float]] = None
    ) -> None:
        self.beta: float = beta
        self.arr: tp.List[float] = []

        if init_arr is not None:
            for val in init_arr:
                self.append(val)

    def append(self, val: float) -> None:
        if len(self.arr) > 0:
            self.arr.append(self.arr[-1] * self.beta + val * (1 - self.beta))
        else:
            self.arr.append(val)


class DQNNet(nn.Module):
    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(inputs),
            torch.nn.Linear(inputs, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(256, 128), torch.nn.BatchNorm1d(128), torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(128, 64), torch.nn.BatchNorm1d(64), torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(torch.nn.Linear(64, outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class DQN:
    # pylint:disable=too-many-instance-attributes
    def __init__(self, inputs: int, outputs: int) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"using {self.device}")

        self.policy_net = DQNNet(inputs, outputs).to(self.device)

        self.target_net = DQNNet(inputs, outputs)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-2)
        self.memory = ReplayMemory(100000)

        self.current_epoch = 0
        self.scores: tp.List[int] = []
        self.episode_durations: tp.List[int] = []
        self.smooth_scores = ExpSmoothedArr(beta=0.9)
        self.smooth_episode_durations = ExpSmoothedArr(beta=0.9)
        self.steps_done = 0
        self.training_time = 0

    def backup(self, path: str) -> None:
        timer = time()
        torch.save(
            {
                "current_epoch": self.current_epoch,
                "model_state_dict": self.policy_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "memory": self.memory,
                "scores": self.scores,
                "episode_durations": self.episode_durations,
                "steps_done": self.steps_done,
                "training_time": self.training_time,
            },
            path,
        )
        print(f"backup done in {round(time() - timer, 1)} sec")

    def restore(self, path: str) -> None:
        timer = time()
        checkpoint = torch.load(path, map_location=self.device)  # type: ignore
        self.policy_net.load_state_dict(checkpoint["model_state_dict"])
        self.policy_net.train()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.memory = checkpoint["memory"]

        self.current_epoch = checkpoint["current_epoch"]
        self.scores = checkpoint["scores"]
        self.episode_durations = checkpoint["episode_durations"]
        self.steps_done = checkpoint["steps_done"]
        self.training_time = checkpoint["training_time"]

        self.smooth_scores = ExpSmoothedArr(beta=0.9, init_arr=self.scores)
        self.smooth_episode_durations = ExpSmoothedArr(
            beta=0.9, init_arr=self.episode_durations
        )
        print(f"Duration of model reading was {round(time() - timer, 1)} sec")

    def select_action(self, state: torch.Tensor, n_actions: int) -> torch.Tensor:
        sample = random.random()  # noqa:S311
        eps_threshold = EPS_END + (EPS_START - EPS_END) * (
            1e6 - self.steps_done
        ) / 1e6 * (self.steps_done < 1e6)
        # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        #     math.exp(-1. * steps_done / EPS_DECAY)
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action: torch.Tensor = self.target_net(state).max(1)[1].view(1, 1)
                return action
        else:
            return torch.tensor(
                [[random.randrange(n_actions)]],  # noqa:S311
                device=self.device,
                dtype=torch.int64,
            )

    def optimize(self) -> None:
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # pylint:disable=not-callable
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)

        next_state_values[non_final_mask] = (
            self.target_net(non_final_next_states).max(1)[0].detach()
        )

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute loss
        loss = torch.nn.functional.mse_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()  # type: ignore
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.steps_done += 1

    def update_target_net(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def show_stat(self) -> None:
        print(f"{self.current_epoch} episodes played at all")
        print(f"{self.steps_done} learning steps done at all")
        plt.figure(figsize=(18, 6))
        plt.plot(self.scores)
        plt.plot(self.smooth_scores.arr)
        plt.show()

        plt.figure(figsize=(18, 6))
        plt.plot(self.episode_durations)
        plt.plot(self.smooth_episode_durations.arr)
        plt.show()
