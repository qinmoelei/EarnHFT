import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, args):
        self.s = np.zeros((args.batch_size, args.state_dim))
        self.a = np.zeros((args.batch_size, 1))
        self.a_logprob = np.zeros((args.batch_size, 1))
        self.r = np.zeros((args.batch_size, 1))
        self.s_ = np.zeros((args.batch_size, args.state_dim))
        self.dw = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.count = 0

    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        a = torch.tensor(
            self.a, dtype=torch.long
        )  # In discrete action space, 'a' needs to be torch.long
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)

        return s, a, a_logprob, r, s_, dw, done


class ReplayBuffer_Multi_info:
    def __init__(self, batch_size, state_dim, action_dim, device="cuda"):
        self.s = np.zeros((batch_size, state_dim))
        self.previous_action = np.zeros((batch_size, 1))
        self.avaliable_action = np.zeros((batch_size, action_dim))
        self.q_value = np.zeros((batch_size, action_dim))

        self.a = np.zeros((batch_size, 1))
        self.a_logprob = np.zeros((batch_size, 1))
        self.r = np.zeros((batch_size, 1))
        self.all_act_prob = np.zeros((batch_size, action_dim))

        self.s_ = np.zeros((batch_size, state_dim))
        self.previous_action_ = np.zeros((batch_size, 1))
        self.avaliable_action_ = np.zeros((batch_size, action_dim))
        self.q_value_ = np.zeros((batch_size, action_dim))

        self.done = np.zeros((batch_size, 1))
        self.count = 0
        self.device = device

    def store(self, s, previous_action, avaliable_action, q_value, a,
              a_logprob, r, all_act_prob, s_, previous_action_,
              avaliable_action_, q_value_, done):
        self.s[self.count] = s
        self.previous_action[self.count] = previous_action
        self.avaliable_action[self.count] = avaliable_action
        self.q_value[self.count] = q_value

        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.all_act_prob[self.count] = all_act_prob

        self.s_[self.count] = s_
        self.previous_action_[self.count] = previous_action_
        self.avaliable_action_[self.count] = avaliable_action_
        self.q_value_[self.count] = q_value_
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float).to(self.device)
        previous_action = torch.tensor(self.previous_action,
                                       dtype=torch.float).to(self.device)
        avaliable_action = torch.tensor(self.avaliable_action,
                                        dtype=torch.float).to(self.device)
        q_value = torch.tensor(self.q_value, dtype=torch.float).to(self.device)

        a = torch.tensor(self.a, dtype=torch.long).to(
            self.device
        )  # In discrete action space, 'a' needs to be torch.long
        a_logprob = torch.tensor(self.a_logprob,
                                 dtype=torch.float).to(self.device)
        r = torch.tensor(self.r, dtype=torch.float).to(self.device)
        all_act_prob = torch.tensor(self.all_act_prob,
                                    dtype=torch.float).to(self.device)

        s_ = torch.tensor(self.s_, dtype=torch.float).to(self.device)
        previous_action_ = torch.tensor(self.previous_action_,
                                        dtype=torch.float).to(self.device)
        avaliable_action_ = torch.tensor(self.avaliable_action_,
                                         dtype=torch.float).to(self.device)
        q_value_ = torch.tensor(self.q_value_,
                                dtype=torch.float).to(self.device)

        done = torch.tensor(self.done, dtype=torch.float).to(self.device)

        return s, previous_action, avaliable_action, q_value, a, a_logprob, r, all_act_prob, s_, previous_action_, avaliable_action_, q_value_, done
