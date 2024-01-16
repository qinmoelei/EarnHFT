import sys

sys.path.append(".")
from RL.util.sum_tree import SumTree
import torch
import numpy as np
from collections import deque, namedtuple
import random


# traiditional 1 step td error
class ReplayMemory:
    def __init__(self, memory_size, state_shape, info_len):
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.state_shape = state_shape
        self.memory_counter = 0
        self.memory_size = memory_size
        self.state_memory = torch.FloatTensor(self.memory_size, state_shape)
        self.action_memory = torch.LongTensor(self.memory_size)
        self.reward_memory = torch.FloatTensor(self.memory_size)
        self.done_memory = torch.FloatTensor(self.memory_size)
        self.state__memory = torch.FloatTensor(self.memory_size, state_shape)

    def reset(self):
        self.memory_counter = 0
        self.state_memory = torch.FloatTensor(self.memory_size,
                                              self.state_shape)
        self.action_memory = torch.LongTensor(self.memory_size)
        self.reward_memory = torch.FloatTensor(self.memory_size)
        self.done_memory = torch.FloatTensor(self.memory_size)
        self.state__memory = torch.FloatTensor(self.memory_size,
                                               self.state_shape)

    def store(self, s, a, r, s_, done):
        index = self.memory_counter % self.memory_size
        self.state_memory[index] = s
        self.action_memory[index] = torch.LongTensor([a.tolist()])
        self.reward_memory[index] = torch.FloatTensor([r])
        self.state__memory[index] = s_
        self.done_memory[index] = torch.FloatTensor([done])

        self.memory_counter += 1

    def sample(self, size):
        sample_index = np.random.choice(self.memory_size, size)
        state_sample = torch.FloatTensor(size,
                                         self.state_shape).to(self.device)
        action_sample = torch.LongTensor(size, 1).to(self.device)
        reward_sample = torch.FloatTensor(size, 1).to(self.device)
        state__sample = torch.FloatTensor(size,
                                          self.state_shape).to(self.device)
        for index in range(sample_index.size):
            state_sample[index] = self.state_memory[sample_index[index]]
            action_sample[index] = self.action_memory[sample_index[index]]
            reward_sample[index] = self.reward_memory[sample_index[index]]
            state__sample[index] = self.state__memory[sample_index[index]]
        return state_sample, action_sample, reward_sample, state__sample


# multi step replay buffer: set parallel_env=1 and n_step=3600 in our case for defalt configuration
class Multi_step_ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self,
                 buffer_size,
                 batch_size,
                 device,
                 seed,
                 gamma,
                 n_step,
                 parallel_env=1):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.buffer_size = buffer_size
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        random.seed(seed)
        self.gamma = gamma
        self.n_step = n_step
        self.parallel_env = parallel_env
        self.n_step_buffer = [
            deque(maxlen=self.n_step) for i in range(parallel_env)
        ]
        self.iter_ = 0

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # if we want to have multi core
        if self.iter_ == self.parallel_env:
            self.iter_ = 0
        self.n_step_buffer[self.iter_].append(
            (state, action, reward, next_state, done))
        if len(self.n_step_buffer[self.iter_]) == self.n_step:
            state, action, reward, next_state, done = self.calc_multistep_return(
                self.n_step_buffer[self.iter_])
            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)
        self.iter_ += 1

    def reset(self):
        self.memory = deque(maxlen=self.buffer_size)

    def calc_multistep_return(self, n_step_buffer):
        Return = 0
        for idx in range(self.n_step):
            Return += self.gamma**idx * n_step_buffer[idx][2]

        return (
            n_step_buffer[0][0],
            n_step_buffer[0][1],
            Return,
            n_step_buffer[-1][3],
            n_step_buffer[-1][4],
        )

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = (torch.from_numpy(
            np.stack([e.state for e in experiences
                      if e is not None])).float().to(self.device))
        actions = (torch.from_numpy(
            np.vstack([e.action for e in experiences
                       if e is not None])).long().to(self.device))
        rewards = (torch.from_numpy(
            np.vstack([e.reward for e in experiences
                       if e is not None])).float().to(self.device))
        next_states = (torch.from_numpy(
            np.stack([e.next_state for e in experiences
                      if e is not None])).float().to(self.device))
        dones = (torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None
                       ]).astype(np.uint8)).float().to(self.device))

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


# multi step replay buffer: set parallel_env=1 and n_step=3600 in our case for defalt configuration
class Multi_step_ReplayBuffer_info:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self,
                 buffer_size,
                 batch_size,
                 device,
                 seed,
                 gamma,
                 n_step,
                 parallel_env=1):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.buffer_size = buffer_size
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=[
                "state", "action", "reward", "next_state", "done", "info"
            ],
        )
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.n_step = n_step
        self.parallel_env = parallel_env
        self.n_step_buffer = [
            deque(maxlen=self.n_step) for i in range(parallel_env)
        ]
        self.iter_ = 0

    def add(self, state, action, reward, next_state, done, info: dict):
        """Add a new experience to memory."""
        # if we want to have multi core
        if self.iter_ == self.parallel_env:
            self.iter_ = 0
        self.n_step_buffer[self.iter_].append(
            (state, action, reward, next_state, done, info))
        if len(self.n_step_buffer[self.iter_]) == self.n_step:
            state, action, reward, next_state, done, info = self.calc_multistep_return(
                self.n_step_buffer[self.iter_])
            e = self.experience(state, action, reward, next_state, done, info)
            self.memory.append(e)
        self.iter_ += 1

    def reset(self):
        self.memory = deque(maxlen=self.buffer_size)

    def calc_multistep_return(self, n_step_buffer):
        Return = 0
        for idx in range(self.n_step):
            Return += self.gamma**idx * n_step_buffer[idx][2]
        self.info_key = n_step_buffer[0][5].keys()

        return (
            n_step_buffer[0][0],
            n_step_buffer[0][1],
            Return,
            n_step_buffer[-1][3],
            n_step_buffer[-1][4],
            n_step_buffer[0][5],
        )

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = (torch.from_numpy(
            np.stack([e.state for e in experiences
                      if e is not None])).float().to(self.device))
        actions = (torch.from_numpy(
            np.vstack([e.action for e in experiences
                       if e is not None])).long().to(self.device))
        rewards = (torch.from_numpy(
            np.vstack([e.reward for e in experiences
                       if e is not None])).float().to(self.device))
        next_states = (torch.from_numpy(
            np.stack([e.next_state for e in experiences
                      if e is not None])).float().to(self.device))
        dones = (torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None
                       ]).astype(np.uint8)).float().to(self.device))
        infos = dict()
        for key in self.info_key:
            infos[key] = (torch.from_numpy(
                np.stack([e.info[key] for e in experiences if e is not None
                          ]).astype(np.uint8)).float().to(self.device))

        return (states, actions, rewards, next_states, dones, infos)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class Multi_step_ReplayBuffer_multi_info:
    """Fixed-size buffer to store experience tuples."""

    # the different lies on the number of infos that we can store, not only do we store the previous state current state and current
    # info, we also store the previous info, which is important (here we need to add bonus to the )

    def __init__(self,
                 buffer_size,
                 batch_size,
                 device,
                 seed,
                 gamma,
                 n_step,
                 parallel_env=1):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.buffer_size = buffer_size
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=[
                "state",
                "info",
                "action",
                "reward",
                "next_state",
                "done",
                "next_info",
            ],
        )
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.n_step = n_step
        self.parallel_env = parallel_env
        self.n_step_buffer = [
            deque(maxlen=self.n_step) for i in range(parallel_env)
        ]
        self.iter_ = 0

    def add(
        self,
        state,
        info: dict,
        action,
        reward,
        next_state,
        next_info: dict,
        done,
    ):
        """Add a new experience to memory."""
        # if we want to have multi core
        if self.iter_ == self.parallel_env:
            self.iter_ = 0
        self.n_step_buffer[self.iter_].append(
            (state, info, action, reward, next_state, next_info, done))

        if len(self.n_step_buffer[self.iter_]) == self.n_step:
            (
                state,
                info,
                action,
                reward,
                next_state,
                next_info,
                done,
            ) = self.calc_multistep_return(self.n_step_buffer[self.iter_])
            e = self.experience(state, info, action, reward, next_state, done,
                                next_info)
            self.memory.append(e)
        self.iter_ += 1

    def add_transition(
        self,
        transition,
    ):
        (state, info, action, reward, next_state, next_info, done) = transition
        """Add a new experience to memory."""
        # if we want to have multi core
        if self.iter_ == self.parallel_env:
            self.iter_ = 0
        self.n_step_buffer[self.iter_].append(
            (state, info, action, reward, next_state, next_info, done))

        if len(self.n_step_buffer[self.iter_]) == self.n_step:
            (
                state,
                info,
                action,
                reward,
                next_state,
                next_info,
                done,
            ) = self.calc_multistep_return(self.n_step_buffer[self.iter_])
            e = self.experience(state, info, action, reward, next_state, done,
                                next_info)
            self.memory.append(e)
        self.iter_ += 1

    def reset(self):
        self.memory = deque(maxlen=self.buffer_size)

    def calc_multistep_return(self, n_step_buffer):
        state, info, action = n_step_buffer[0][0], n_step_buffer[0][
            1], n_step_buffer[0][2]
        next_state, next_info, done = n_step_buffer[-1][4], n_step_buffer[-1][
            5], n_step_buffer[0][6],
        n_steps_reward = 0
        for i in reversed(range(self.n_step)):  # 逆序计算n_steps_reward
            (
                single_reward,
                single_state_,
                single_info_,
                single_terminal,
            ) = n_step_buffer[i][3], n_step_buffer[i][4], n_step_buffer[i][
                5], n_step_buffer[i][6],
            n_steps_reward = (single_reward + self.gamma *
                              (1 - single_terminal) * n_steps_reward)
            if (
                    single_terminal
            ):  # 如果done=True，说明一个回合结束，保存deque中当前这个transition的s'和terminal作为这个n_steps_transition的next_state和terminal
                (next_state, next_info, done) = (
                    single_state_,
                    single_info_,
                    single_terminal,
                )
            self.info_key = n_step_buffer[0][5].keys()

        return (
            state,
            info,
            action,
            n_steps_reward,
            next_state,
            next_info,
            done,
        )

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = (torch.from_numpy(
            np.stack([e.state for e in experiences
                      if e is not None])).float().to(self.device))
        infos = dict()
        for key in self.info_key:
            infos[key] = (torch.from_numpy(
                np.stack([e.info[key] for e in experiences if e is not None
                          ]).astype(float)).float().to(self.device))
        actions = (torch.from_numpy(
            np.vstack([e.action for e in experiences
                       if e is not None])).long().to(self.device))
        rewards = (torch.from_numpy(
            np.vstack([e.reward for e in experiences
                       if e is not None])).float().to(self.device))
        next_states = (torch.from_numpy(
            np.stack([e.next_state for e in experiences
                      if e is not None])).float().to(self.device))
        next_infos = dict()
        for key in self.info_key:
            next_infos[key] = (torch.from_numpy(
                np.stack([
                    e.next_info[key] for e in experiences if e is not None
                ]).astype(float)).float().to(self.device))
        dones = (torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None
                       ]).astype(float)).float().to(self.device))

        return (states, infos, actions, rewards, next_states, next_infos,
                dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)



































class Multi_step_Prioritized_ReplayBuffer_multi_info(object):
    def __init__(
        self,
        alpha,
        beta,
        batch_size,
        n_steps,
        state_dim,
        action_dim,
        buffer_capacity=100000,
        gamma=1,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.sum_tree = SumTree(self.buffer_capacity)
        self.n_steps = n_steps
        self.n_steps_deque = deque(maxlen=self.n_steps)
        self.buffer = {
            "market_state": np.zeros((self.buffer_capacity, state_dim)),
            "previous_action": np.zeros((self.buffer_capacity)),
            "avaliable_action": np.zeros((self.buffer_capacity, action_dim)),
            "q_action": np.zeros((self.buffer_capacity, action_dim)),
            "action": np.zeros((self.buffer_capacity, 1)),
            "reward": np.zeros(self.buffer_capacity),
            "next_market_state": np.zeros((self.buffer_capacity, state_dim)),
            "next_previous_action": np.zeros((self.buffer_capacity)),
            "next_avaliable_action": np.zeros(
                (self.buffer_capacity, action_dim)),
            "next_q_action": np.zeros((self.buffer_capacity, action_dim)),
            "terminal": np.zeros(self.buffer_capacity),
        }
        self.current_size = 0
        self.count = 0

    def store_transition(
        self,
        market_state,
        previous_action,
        avaliable_action,
        q_action,
        action,
        reward,
        next_market_state,
        next_previous_action,
        next_avaliable_action,
        next_q_action,
        terminal,
    ):
        transition = (
            market_state,
            previous_action,
            avaliable_action,
            q_action,
            action,
            reward,
            next_market_state,
            next_previous_action,
            next_avaliable_action,
            next_q_action,
            terminal,
        )
        self.n_steps_deque.append(transition)
        if len(self.n_steps_deque) == self.n_steps:
            (
                market_state,
                previous_action,
                avaliable_action,
                q_action,
                action,
                n_steps_reward,
                next_market_state,
                next_previous_action,
                next_avaliable_action,
                next_q_action,
                terminal,
            ) = self.get_n_steps_transition()
            self.buffer["market_state"][self.count] = market_state
            self.buffer["previous_action"][self.count] = previous_action
            self.buffer["avaliable_action"][self.count] = avaliable_action
            self.buffer["q_action"][self.count] = q_action
            self.buffer["action"][self.count] = action
            self.buffer["reward"][self.count] = n_steps_reward
            self.buffer["next_market_state"][self.count] = next_market_state
            self.buffer["next_previous_action"][
                self.count] = next_previous_action
            self.buffer["next_avaliable_action"][
                self.count] = next_avaliable_action
            self.buffer["next_q_action"][self.count] = next_q_action
            self.buffer["terminal"][self.count] = terminal

            # 如果是buffer中的第一条经验，那么指定priority为1.0；否则对于新存入的经验，指定为当前最大的priority
            priority = 1.0 if self.current_size == 0 else self.sum_tree.priority_max
            self.sum_tree.update(data_index=self.count,
                                 priority=priority)  # 更新当前经验在sum_tree中的优先级
            self.count = (
                self.count + 1
            ) % self.buffer_capacity  # When 'count' reaches buffer_capacity, it will be reset to 0.
            self.current_size = min(self.current_size + 1,
                                    self.buffer_capacity)

    def sample(self):
        batch_index, IS_weight = self.sum_tree.get_batch_index(
            current_size=self.current_size,
            batch_size=self.batch_size,
            beta=self.beta)

        batch = {}
        for key in self.buffer.keys():  # numpy->tensor
            if key in ["action", "previous_action", "next_previous_action"]:
                batch[key] = torch.tensor(self.buffer[key][batch_index],
                                          dtype=torch.long)

            else:
                batch[key] = torch.tensor(self.buffer[key][batch_index],
                                          dtype=torch.float32)

        return batch, batch_index, IS_weight

    def get_n_steps_transition(self):
        (
            market_state,
            previous_action,
            avaliable_action,
            q_action,
            action,
        ) = self.n_steps_deque[0][:5]  # 获取deque中第一个transition的s和a
        (
            next_market_state,
            next_previous_action,
            next_avaliable_action,
            next_q_action,
            terminal,
        ) = self.n_steps_deque[-1][6:]  # 获取deque中最后一个transition的s'和terminal
        n_steps_reward = 0
        for i in reversed(range(self.n_steps)):  # 逆序计算n_steps_reward
            (
                single_reward,
                single_next_market_state,
                single_next_previous_action,
                single_next_avaliable_action,
                single_next_q_action,
                single_terminal,
            ) = self.n_steps_deque[i][5:]
            n_steps_reward = (single_reward + self.gamma *
                              (1 - single_terminal) * n_steps_reward)
            if (
                    single_terminal
            ):  # 如果done=True，说明一个回合结束，保存deque中当前这个transition的s'和terminal作为这个n_steps_transition的next_state和terminal
                (
                    next_market_state,
                    next_previous_action,
                    next_avaliable_action,
                    next_q_action,
                    terminal,
                ) = (
                    single_next_market_state,
                    single_next_previous_action,
                    single_next_avaliable_action,
                    single_next_q_action,
                    single_terminal,
                )

        return (
            market_state,
            previous_action,
            avaliable_action,
            q_action,
            action,
            n_steps_reward,
            next_market_state,
            next_previous_action,
            next_avaliable_action,
            next_q_action,
            terminal,
        )

    def update_batch_priorities(
            self, batch_index,
            td_errors):  # 根据传入的td_error，更新batch_index所对应数据的priorities
        priorities = (np.abs(td_errors) + 0.01)**self.alpha
        for index, priority in zip(batch_index, priorities):
            self.sum_tree.update(data_index=index, priority=priority)














class Multi_step_SIL_ReplayBuffer_multi_info(object):
    # like PER but use reward as priority. no need to update priority list
    def __init__(
        self,
        alpha,
        beta,
        batch_size,
        n_steps,
        state_dim,
        action_dim,
        buffer_capacity=100000,
        gamma=1,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.sum_tree = SumTree(self.buffer_capacity)
        self.n_steps = n_steps
        self.n_steps_deque = deque(maxlen=self.n_steps)
        self.buffer = {
            "market_state": np.zeros((self.buffer_capacity, state_dim)),
            "previous_action": np.zeros((self.buffer_capacity)),
            "avaliable_action": np.zeros((self.buffer_capacity, action_dim)),
            "q_action": np.zeros((self.buffer_capacity, action_dim)),
            "action": np.zeros((self.buffer_capacity, 1)),
            "reward": np.zeros(self.buffer_capacity),
            "next_market_state": np.zeros((self.buffer_capacity, state_dim)),
            "next_previous_action": np.zeros((self.buffer_capacity)),
            "next_avaliable_action": np.zeros(
                (self.buffer_capacity, action_dim)),
            "next_q_action": np.zeros((self.buffer_capacity, action_dim)),
            "terminal": np.zeros(self.buffer_capacity),
        }
        self.current_size = 0
        self.count = 0

    def store_transition(
        self,
        market_state,
        previous_action,
        avaliable_action,
        q_action,
        action,
        reward,
        next_market_state,
        next_previous_action,
        next_avaliable_action,
        next_q_action,
        terminal,
    ):
        transition = (
            market_state,
            previous_action,
            avaliable_action,
            q_action,
            action,
            reward,
            next_market_state,
            next_previous_action,
            next_avaliable_action,
            next_q_action,
            terminal,
        )
        self.n_steps_deque.append(transition)
        if len(self.n_steps_deque) == self.n_steps:
            (
                market_state,
                previous_action,
                avaliable_action,
                q_action,
                action,
                n_steps_reward,
                next_market_state,
                next_previous_action,
                next_avaliable_action,
                next_q_action,
                terminal,
            ) = self.get_n_steps_transition()
            self.buffer["market_state"][self.count] = market_state
            self.buffer["previous_action"][self.count] = previous_action
            self.buffer["avaliable_action"][self.count] = avaliable_action
            self.buffer["q_action"][self.count] = q_action
            self.buffer["action"][self.count] = action
            self.buffer["reward"][self.count] = n_steps_reward
            self.buffer["next_market_state"][self.count] = next_market_state
            self.buffer["next_previous_action"][
                self.count] = next_previous_action
            self.buffer["next_avaliable_action"][
                self.count] = next_avaliable_action
            self.buffer["next_q_action"][self.count] = next_q_action
            self.buffer["terminal"][self.count] = terminal

            # 如果是buffer中的第一条经验，那么指定priority为1.0；否则对于新存入的经验，指定为当前最大的priority
            priority = n_steps_reward
            self.sum_tree.update(data_index=self.count,
                                 priority=priority)  # 更新当前经验在sum_tree中的优先级
            self.count = (
                self.count + 1
            ) % self.buffer_capacity  # When 'count' reaches buffer_capacity, it will be reset to 0.
            self.current_size = min(self.current_size + 1,
                                    self.buffer_capacity)

    def sample(self):
        batch_index, IS_weight = self.sum_tree.get_batch_index(
            current_size=self.current_size,
            batch_size=self.batch_size,
            beta=self.beta)

        batch = {}
        for key in self.buffer.keys():  # numpy->tensor
            if key in ["action", "previous_action", "next_previous_action"]:
                batch[key] = torch.tensor(self.buffer[key][batch_index],
                                          dtype=torch.long)

            else:
                batch[key] = torch.tensor(self.buffer[key][batch_index],
                                          dtype=torch.float32)

        return batch, batch_index, IS_weight

    def get_n_steps_transition(self):
        (
            market_state,
            previous_action,
            avaliable_action,
            q_action,
            action,
        ) = self.n_steps_deque[0][:5]  # 获取deque中第一个transition的s和a
        (
            next_market_state,
            next_previous_action,
            next_avaliable_action,
            next_q_action,
            terminal,
        ) = self.n_steps_deque[-1][6:]  # 获取deque中最后一个transition的s'和terminal
        n_steps_reward = 0
        for i in reversed(range(self.n_steps)):  # 逆序计算n_steps_reward
            (
                single_reward,
                single_next_market_state,
                single_next_previous_action,
                single_next_avaliable_action,
                single_next_q_action,
                single_terminal,
            ) = self.n_steps_deque[i][5:]
            n_steps_reward = (single_reward + self.gamma *
                              (1 - single_terminal) * n_steps_reward)
            if (
                    single_terminal
            ):  # 如果done=True，说明一个回合结束，保存deque中当前这个transition的s'和terminal作为这个n_steps_transition的next_state和terminal
                (
                    next_market_state,
                    next_previous_action,
                    next_avaliable_action,
                    next_q_action,
                    terminal,
                ) = (
                    single_next_market_state,
                    single_next_previous_action,
                    single_next_avaliable_action,
                    single_next_q_action,
                    single_terminal,
                )

        return (
            market_state,
            previous_action,
            avaliable_action,
            q_action,
            action,
            n_steps_reward,
            next_market_state,
            next_previous_action,
            next_avaliable_action,
            next_q_action,
            terminal,
        )






















if __name__ == "__main__":
    a = 1234545
