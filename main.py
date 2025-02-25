import os
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pygame
import matplotlib
from Config import Config
from Logger import Logger
import matplotlib.pyplot as plt
from collections import deque

matplotlib.use('Agg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


class Plotter:
    @staticmethod
    def plot_curve(data_list, label, title, save_file):
        plt.figure()
        plt.plot(data_list, label=label)
        plt.title(title)
        plt.legend()
        plt.savefig(save_file)
        plt.close()


class GomokuEnv:
    def __init__(self, board_size=9, win_length=5, agent_player=1):
        self.board_size = board_size
        self.win_length = win_length
        self.agent_player = agent_player
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None
        return tuple(self.board.flatten())

    def available_actions(self):
        acts = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 0:
                    acts.append((i, j))
        return acts

    def step(self, action):
        i, j = action
        if self.done:
            return self._get_state(), 0.0, True, {}
        if not (0 <= i < self.board_size and 0 <= j < self.board_size):
            self.done = True
            return self._get_state(), -1.0, True, {"illegal_move": True}
        if self.board[i, j] != 0:
            self.done = True
            return self._get_state(), -1.0, True, {"illegal_move": True}
        self.board[i, j] = self.current_player
        if self._check_winner(i, j):
            self.done = True
            self.winner = self.current_player
            if self.current_player == self.agent_player:
                return self._get_state(), 10.0, True, {}
            else:
                return self._get_state(), -10.0, True, {}
        if len(self.available_actions()) == 0:
            self.done = True
            return self._get_state(), 0.0, True, {}

        reward = self._calc_connected_reward(i, j)
        self.current_player = 3 - self.current_player
        return self._get_state(), reward, False, {}

    def _check_winner(self, row, col):
        p = self.board[row, col]
        dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for d in dirs:
            c = 1
            x, y = row + d[0], col + d[1]
            while 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x, y] == p:
                c += 1
                x += d[0]
                y += d[1]
            x, y = row - d[0], col - d[1]
            while 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x, y] == p:
                c += 1
                x -= d[0]
                y -= d[1]
            if c >= self.win_length:
                return True
        return False

    def _calc_connected_reward(self, row, col):
        p = self.board[row, col]
        o = 3 - p
        dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
        max_conn = 1
        opp_conn = 1
        for d in dirs:
            c = 1
            x, y = row + d[0], col + d[1]
            while 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x, y] == p:
                c += 1
                x += d[0]
                y += d[1]
            x, y = row - d[0], col - d[1]
            while 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x, y] == p:
                c += 1
                x -= d[0]
                y -= d[1]
            max_conn = max(max_conn, c)
            c2 = 1
            xx, yy = row + d[0], col + d[1]
            while 0 <= xx < self.board_size and 0 <= yy < self.board_size and self.board[xx, yy] == o:
                c2 += 1
                xx += d[0]
                yy += d[1]
            xx, yy = row - d[0], col - d[1]
            while 0 <= xx < self.board_size and 0 <= yy < self.board_size and self.board[xx, yy] == o:
                c2 += 1
                xx -= d[0]
                yy -= d[1]
            opp_conn = max(opp_conn, c2)
        reward = 0.0
        if max_conn == 2:
            reward += 0.3
        elif max_conn == 3:
            reward += 0.6
        elif max_conn == 4:
            reward += 1.0
        if opp_conn == 2:
            reward -= 0.3
        elif opp_conn == 3:
            reward -= 0.6
        elif opp_conn == 4:
            reward -= 1.0
        return reward

    def _get_state(self):
        return tuple(self.board.flatten())

    def render(self, screen, cell_size=50):
        screen.fill((245, 222, 179))
        for i in range(self.board_size):
            pygame.draw.line(screen, (139, 69, 19),
                             (0, i * cell_size + cell_size // 2),
                             (self.board_size * cell_size, i * cell_size + cell_size // 2), 3)
            pygame.draw.line(screen, (139, 69, 19),
                             (i * cell_size + cell_size // 2, 0),
                             (i * cell_size + cell_size // 2, self.board_size * cell_size), 3)
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 1:
                    pygame.draw.circle(screen, (50, 50, 50),
                                       (j * cell_size + cell_size // 2, i * cell_size + cell_size // 2),
                                       cell_size // 2 - 4)
                elif self.board[i, j] == 2:
                    pygame.draw.circle(screen, (200, 0, 0),
                                       (j * cell_size + cell_size // 2, i * cell_size + cell_size // 2),
                                       cell_size // 2 - 4)
        pygame.display.flip()


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000, alpha=0.6, beta_start=0.4, beta_frames=200000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.frame = 1

    def _beta_by_frame(self):
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)

    def push(self, transition, td_error=1.0):
        max_prio = self.priorities.max() if len(self.buffer) > 0 else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max(max_prio, td_error)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return [], [], torch.zeros(0)
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        beta = self._beta_by_frame()
        self.frame += 1
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        batch = [self.buffer[idx] for idx in indices]
        return batch, indices, torch.FloatTensor(weights).to(device)

    def update_priorities(self, indices, new_prios):
        for idx, prio in zip(indices, new_prios):
            self.priorities[idx] = prio

    def size(self):
        return len(self.buffer)


class MultiStepWrapper:
    def __init__(self, n_step=3, gamma=0.99):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque()

    def append(self, transition):
        self.buffer.append(transition)

    def get_ready(self):
        return len(self.buffer) >= self.n_step

    def flush(self):
        self.buffer.clear()

    def pop(self):
        R = 0.0
        for i, (s, a, r, ns, d) in enumerate(self.buffer):
            R += (self.gamma ** i) * r
        s0, a0, _, _, _ = self.buffer[0]
        _, _, _, s_n, done_n = self.buffer[-1]
        return s0, a0, R, s_n, done_n


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init * mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init * mu_range)

    def reset_noise(self):
        eps_in = torch.randn(self.in_features)
        eps_out = torch.randn(self.out_features)
        eps_in = eps_in.sign().mul_(eps_in.abs().sqrt_())
        eps_out = eps_out.sign().mul_(eps_out.abs().sqrt_())
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            w = self.weight_mu
            b = self.bias_mu
        return nn.functional.linear(x, w, b)


class FastDuelingNoisyNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.board_size = config.board_size
        self.atoms = config.atoms if config.use_distributional else 1
        self.v_min = config.v_min
        self.v_max = config.v_max
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.fc = nn.Sequential(
            nn.Linear(32 * self.board_size * self.board_size, 128),
            nn.ReLU()
        )
        if config.noisy_net:
            self.value_stream = nn.Sequential(
                NoisyLinear(128, 128),
                nn.ReLU(),
                NoisyLinear(128, self.atoms)
            )
            self.adv_stream = nn.Sequential(
                NoisyLinear(128, 128),
                nn.ReLU(),
                NoisyLinear(128, self.board_size * self.board_size * self.atoms)
            )
        else:
            self.value_stream = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, self.atoms)
            )
            self.adv_stream = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, self.board_size * self.board_size * self.atoms)
            )

    def forward(self, x):
        x = x.view(-1, 1, self.board_size, self.board_size)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        v = self.value_stream(x).view(-1, 1, self.atoms)
        a = self.adv_stream(x).view(-1, self.board_size * self.board_size, self.atoms)
        mean_a = a.mean(dim=1, keepdim=True)
        q = v + a - mean_a
        if self.config.use_distributional:
            q = nn.functional.softmax(q, dim=2)
        return q

    def reset_noise(self):
        if not self.config.noisy_net:
            return
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


class DQNAgent:
    def __init__(self, config):
        self.config = config
        self.gamma = config.gamma
        self.n_step = config.n_step
        self.use_per = config.use_per
        self.use_distributional = config.use_distributional
        self.v_min = config.v_min
        self.v_max = config.v_max
        self.atoms = config.atoms if config.use_distributional else 1
        self.support = torch.linspace(self.v_min, self.v_max, self.atoms).to(device)
        self.online_net = FastDuelingNoisyNet(config).to(device)
        self.target_net = FastDuelingNoisyNet(config).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config.learning_rate)
        if self.use_per:
            self.replay = PrioritizedReplayBuffer(config.capacity, config.per_alpha, config.per_beta_start,
                                                  config.per_beta_frames)
        else:
            self.replay = ReplayBuffer(config.capacity)
        if config.multi_step:
            self.multi_step_buffer = MultiStepWrapper(config.n_step, config.gamma)
        else:
            self.multi_step_buffer = None
        self.update_count = 0
        self.eps = config.eps_start
        self.eps_min = config.eps_min
        self.eps_decay = config.eps_decay
        self.losses = []
        self.learn_every = 4
        self.learn_counter = 0

    def choose_action(self, state, env=None):
        if self.config.noisy_net:
            eps_explore = 0.0
        else:
            eps_explore = self.eps
        board_size = self.config.board_size
        if env:
            valid_moves = env.available_actions()
        else:
            valid_moves = [(i, j) for i in range(board_size) for j in range(board_size)]
        if not valid_moves:
            return 0
        if random.random() < eps_explore:
            i, j = random.choice(valid_moves)
            return i * board_size + j
        else:
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(device)
                pmf = self.online_net(s)
                if self.use_distributional:
                    q_values = (pmf * self.support.view(1, 1, self.atoms)).sum(dim=2)
                else:
                    q_values = pmf.squeeze(-1)
                q_values = q_values[0].cpu().numpy()
            for idx in range(board_size * board_size):
                ii = idx // board_size
                jj = idx % board_size
                if (ii, jj) not in valid_moves:
                    q_values[idx] = -1e9
            return int(np.argmax(q_values))

    def store(self, s, a, r, ns, done):
        if self.multi_step_buffer:
            self.multi_step_buffer.append((s, a, r, ns, done))
            if self.multi_step_buffer.get_ready():
                s0, a0, R, s_n, done_n = self.multi_step_buffer.pop()
                self._push_to_buffer(s0, a0, R, s_n, done_n)
            if done:
                self.multi_step_buffer.flush()
        else:
            self._push_to_buffer(s, a, r, ns, done)

    def _push_to_buffer(self, s, a, r, ns, done):
        if self.use_per:
            self.replay.push((s, a, r, ns, done), td_error=1.0)
        else:
            self.replay.push((s, a, r, ns, done))

    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def learn(self, episode, ep_reward):
        self.learn_counter += 1
        if self.learn_counter % self.learn_every != 0:
            return
        if self.replay.size() < self.config.batch_size:
            return
        if self.use_per:
            transitions, indices, weights = self.replay.sample(self.config.batch_size)
        else:
            transitions = self.replay.sample(self.config.batch_size)
            weights = torch.ones(self.config.batch_size, device=device)
            indices = None
        batch = list(zip(*transitions))
        s_batch = torch.FloatTensor(batch[0]).to(device)
        a_batch = torch.LongTensor(batch[1]).to(device)
        r_batch = torch.FloatTensor(batch[2]).to(device)
        ns_batch = torch.FloatTensor(batch[3]).to(device)
        d_batch = torch.FloatTensor(batch[4]).to(device)
        if self.use_distributional:
            self._distributional_update(s_batch, a_batch, r_batch, ns_batch, d_batch, weights, indices)
        else:
            self._classic_update(s_batch, a_batch, r_batch, ns_batch, d_batch, weights, indices)
        self.update_count += 1
        if self.update_count % 10 == 0:
            print(
                f"[Learn {self.update_count}] Ep={episode} R={ep_reward:.2f} Size={self.replay.size()} Loss={self.losses[-1]:.4f} Eps={self.eps:.4f}")
        if self.update_count % self.config.update_target_freq == 0:
            self.update_target()
        if not self.config.noisy_net and self.eps > self.eps_min:
            self.eps *= self.eps_decay
        self.online_net.reset_noise()
        self.target_net.reset_noise()

    def _classic_update(self, s, a, r, ns, d, weights, indices):
        with torch.no_grad():
            q_online_next = self.online_net(ns).squeeze(-1)
            a_next = q_online_next.argmax(dim=1)
            q_target_next = self.target_net(ns).squeeze(-1)
            q_next = q_target_next.gather(1, a_next.unsqueeze(1)).squeeze(1)
            y = r + self.gamma * q_next * (1 - d)
        q_pred_all = self.online_net(s).squeeze(-1)
        q_pred = q_pred_all.gather(1, a.unsqueeze(1)).squeeze(1)
        loss = (weights * (q_pred - y).pow(2)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()
        self.losses.append(loss.item())
        if self.use_per and indices is not None:
            prios = (q_pred - y).abs().detach().cpu().numpy() + 1e-6
            self.replay.update_priorities(indices, prios)

    def _distributional_update(self, s, a, r, ns, d, weights, indices):
        bs = s.size(0)
        with torch.no_grad():
            p_online_ns = self.online_net(ns)
            q_online_ns = (p_online_ns * self.support.view(1, 1, self.atoms)).sum(dim=2)
            a_next = q_online_ns.argmax(dim=1)
            p_target_ns = self.target_net(ns)
            p_target_a = p_target_ns[range(bs), a_next]
        Tz = r.unsqueeze(1) + self.gamma * (1 - d).unsqueeze(1) * self.support.unsqueeze(0)
        Tz = Tz.clamp(self.v_min, self.v_max)
        b = (Tz - self.v_min) / ((self.v_max - self.v_min) / (self.atoms - 1))
        l = b.floor().long().clamp(0, self.atoms - 1)
        u = b.ceil().long().clamp(0, self.atoms - 1)
        m = torch.zeros(bs, self.atoms, device=device)
        for i in range(bs):
            for j in range(self.atoms):
                pj = p_target_a[i, j]
                lj = l[i, j]
                uj = u[i, j]
                m[i, lj] += pj * (uj.float() - b[i, j])
                m[i, uj] += pj * (b[i, j] - lj.float())
        p_out = self.online_net(s)
        p_act = p_out[range(bs), a]
        p_act = p_act.clamp(min=1e-5, max=1.0)
        loss_all = -(m * p_act.log()).sum(dim=1)
        prios = loss_all.detach().cpu().numpy() + 1e-6
        loss = (loss_all * weights).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()
        self.losses.append(loss.item())
        if self.use_per and indices is not None:
            self.replay.update_priorities(indices, prios)


class Trainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.env = GomokuEnv(config.board_size, config.win_length, config.agent_player)
        self.agent = DQNAgent(config)
        self.reward_history = []
        self.win_rate_history = []
        self.loss_history = self.agent.losses

    def train(self):
        self.logger.log(f"Start Training: episodes={self.config.episodes}, batch_size={self.config.batch_size}", True)
        for ep in range(1, self.config.episodes + 1):
            s = self.env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                if self.env.current_player == self.config.agent_player:
                    a_idx = self.agent.choose_action(s, self.env)
                    i, j = divmod(a_idx, self.config.board_size)
                    ns, r, done, _ = self.env.step((i, j))
                    self.agent.store(s, a_idx, r, ns, done)
                    self.agent.learn(ep, ep_reward)
                    s = ns
                    ep_reward += r
                else:
                    acts = self.env.available_actions()
                    if not acts:
                        break
                    op = random.choice(acts)
                    ns, r, done, _ = self.env.step(op)
                    s = ns
            self.reward_history.append(ep_reward)
            if ep % self.config.test_interval == 0:
                w_rate = self.test_agent(self.config.test_games)
                self.win_rate_history.append(w_rate)
                avg_loss = np.mean(self.loss_history[-100:]) if len(self.loss_history) > 100 else (
                    np.mean(self.loss_history) if len(self.loss_history) > 0 else 0.0)
                avg_reward = np.mean(self.reward_history[-100:]) if len(self.reward_history) > 100 else np.mean(
                    self.reward_history)
                self.logger.log(
                    f"Episode {ep}/{self.config.episodes} WinRate={w_rate * 100:.1f}% AvgR(100)={avg_reward:.2f} AvgL(100)={avg_loss:.4f} Eps={self.agent.eps:.4f}"
                )
        self.logger.log("Training done.")
        torch.save(self.agent.online_net.state_dict(), self.config.save_model_file)
        self.logger.log(f"Saved model to {self.config.save_model_file}")
        self.logger.save_log()
        Plotter.plot_curve(self.loss_history, "Loss", "Training Loss", self.config.save_figure_file)

    def test_agent(self, episodes=50):
        wins = 0
        for _ in range(episodes):
            s = self.env.reset()
            done = False
            while not done:
                if self.env.current_player == self.config.agent_player:
                    a_idx = self.agent.choose_action(s, self.env)
                    i, j = divmod(a_idx, self.config.board_size)
                    s, r, done, _ = self.env.step((i, j))
                else:
                    acts = self.env.available_actions()
                    if not acts:
                        break
                    op = random.choice(acts)
                    s, r, done, _ = self.env.step(op)
            if self.env.winner == self.config.agent_player:
                wins += 1
        return wins / episodes


def play_game(config):
    pygame.init()
    board_size = config.board_size
    cell_size = 50
    width = board_size * cell_size
    height = board_size * board_size
    screen = pygame.display.set_mode((board_size * cell_size, board_size * cell_size))
    env = GomokuEnv(board_size, config.win_length, config.agent_player)
    s = env.reset()
    agent = DQNAgent(config)
    if os.path.isfile(config.save_model_file):
        agent.online_net.load_state_dict(torch.load(config.save_model_file, map_location=device))
        agent.online_net.eval()
        agent.target_net.load_state_dict(agent.online_net.state_dict())
        agent.target_net.eval()
        print(f"Loaded model from {config.save_model_file}")
    else:
        print(f"Model not found: {config.save_model_file}, train first.")
        pygame.quit()
        return
    running = True
    clock = pygame.time.Clock()
    while running:
        env.render(screen, cell_size)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.MOUSEBUTTONDOWN:
                if env.current_player != config.agent_player and not env.done:
                    pos = pygame.mouse.get_pos()
                    x, y = pos
                    j = x // cell_size
                    i = y // cell_size
                    if 0 <= i < board_size and 0 <= j < board_size and env.board[i, j] == 0 and not env.done:
                        s, r, dn, _ = env.step((i, j))
        if env.current_player == config.agent_player and not env.done:
            a_idx = agent.choose_action(s, env)
            i, j = divmod(a_idx, board_size)
            if 0 <= i < board_size and 0 <= j < board_size and env.board[i, j] == 0:
                s, r, dn, _ = env.step((i, j))
        if env.done:
            env.render(screen, cell_size)
            font = pygame.font.SysFont('simhei', 48)
            if env.winner == config.agent_player:
                text = font.render("AI Wins!", True, (0, 0, 0))
            elif env.winner is not None:
                text = font.render("Player Wins!", True, (255, 0, 0))
            else:
                text = font.render("Draw!", True, (0, 0, 255))
            t_surf = pygame.Surface((text.get_width() + 20, text.get_height() + 20), pygame.SRCALPHA)
            t_surf.fill((255, 255, 255, 128))
            screen.blit(t_surf, (width // 2 - t_surf.get_width() // 2, height // 2 - t_surf.get_height() // 2))
            text_rect = text.get_rect(center=(width // 2, height // 2))
            screen.blit(text, text_rect)
            pygame.display.flip()
            pygame.time.wait(3000)
            running = False
        clock.tick(30)
    pygame.quit()


def main():
    config = Config()
    logger = Logger(config)
    mode = input("1-Train,2-Play:")
    if mode.strip() == "1":
        config.save_to_json()
        trainer = Trainer(config, logger)
        trainer.train()
        Plotter.plot_curve(trainer.win_rate_history, "WinRate", "Test Win Rate", "win_rate.png")
        print("Train done.")
    else:
        config.load_from_json()
        play_game(config)


if __name__ == "__main__":
    main()
