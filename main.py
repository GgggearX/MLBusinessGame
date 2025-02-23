import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pygame
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

# Rest of your code remains unchanged

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义 Q 网络（深度 Q 网络）
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)  # 增加神经元数量
        self.fc2 = nn.Linear(256, 256)  # 隐藏层
        self.fc3 = nn.Linear(256, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))  # 新增层前向传播
        x = self.fc3(x)
        return x


# 创建 Q 网络模型和目标网络
def create_model(input_dim, output_dim):
    model = QNetwork(input_dim, output_dim).to(device)
    target_model = QNetwork(input_dim, output_dim).to(device)
    target_model.load_state_dict(model.state_dict())
    return model, target_model


# 经验回放池（Experience Replay Buffer）
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = []
        self.capacity = capacity
        self.idx = 0

    def push(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.idx] = transition
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


# DQN 智能体
class DQNAgent:
    def __init__(self, state_dim, action_dim,
                 learning_rate=0.0001,
                 gamma=0.95,
                 epsilon=0.9,
                 epsilon_decay=0.999,
                 epsilon_min=0.01,
                 batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.model, self.target_model = create_model(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer()
        self.loss_history = []  # 新增：记录每次学习的损失

    def choose_action(self, state, env=None):
        if random.random() < self.epsilon:
            available = env.available_actions() if env else [(i, j) for i in range(9) for j in range(9)]
            i, j = random.choice(available)
            return i * 9 + j
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def learn(self):
        if self.replay_buffer.size() < self.batch_size:
            return

        transitions = self.replay_buffer.sample(self.batch_size)
        batch = list(zip(*transitions))
        state_batch = torch.tensor(batch[0], dtype=torch.float32).to(device)
        action_batch = torch.tensor(batch[1], dtype=torch.long).to(device)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32).to(device)
        next_state_batch = torch.tensor(batch[3], dtype=torch.float32).to(device)
        done_batch = torch.tensor(batch[4], dtype=torch.float32).to(device)

        q_values = self.model(state_batch)
        q_value = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_model(next_state_batch)
            next_q_value = next_q_values.max(1)[0]
            target = reward_batch + self.gamma * next_q_value * (1 - done_batch)

        loss = self.loss_fn(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_history.append(loss.item())  # 记录损失，不打印

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
        print("Target network updated.")  # 可选：保留但减少频率


# 游戏环境（五子棋）
class GomokuEnv:
    def __init__(self, board_size=9, win_length=5):
        self.board_size = board_size
        self.win_length = win_length
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        return tuple(self.board.flatten())

    def available_actions(self):
        actions = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 0:
                    actions.append((i, j))
        return actions

    def step(self, action):
        i, j = action  # action 是一个 (i, j) 元组
        if self.done:
            return self.get_state(), 0, True, {}

        if self.board[i, j] != 0:
            return self.get_state(), -1, True, {'illegal_move': True}

        self.board[i, j] = self.current_player
        if self.check_winner(i, j):
            self.done = True
            self.winner = self.current_player
            return self.get_state(), 1 if self.current_player == 1 else -1, True, {}

        if len(self.available_actions()) == 0:
            self.done = True
            return self.get_state(), 0, True, {}

        self.current_player = 3 - self.current_player
        return self.get_state(), 0, False, {}

    def check_winner(self, row, col):
        player = self.board[row, col]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for d in directions:
            count = 1
            i, j = row + d[0], col + d[1]
            while 0 <= i < self.board_size and 0 <= j < self.board_size and self.board[i, j] == player:
                count += 1
                i += d[0]
                j += d[1]
            i, j = row - d[0], col - d[1]
            while 0 <= i < self.board_size and 0 <= j < self.board_size and self.board[i, j] == player:
                count += 1
                i -= d[0]
                j -= d[1]
            if count >= self.win_length:
                return True
        return False

    def step(self, action):
        i, j = action  # action 是一个 (i, j) 元组
        if self.done:
            return self.get_state(), 0, True, {}

        if self.board[i, j] != 0:
            return self.get_state(), -1, True, {'illegal_move': True}

        self.board[i, j] = self.current_player
        if self.check_winner(i, j):
            self.done = True
            self.winner = self.current_player
            return self.get_state(), 1 if self.current_player == 1 else -1, True, {}

        if len(self.available_actions()) == 0:
            self.done = True
            return self.get_state(), 0, True, {}

        # 计算启发式奖励
        heuristic_reward = self.calculate_heuristic_reward(i, j)

        self.current_player = 3 - self.current_player
        return self.get_state(), heuristic_reward, False, {}

    def calculate_heuristic_reward(self, row, col):
        """计算启发式奖励，根据连子数量"""
        player = self.board[row, col]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 四个方向：水平、垂直、两条对角线
        max_connected = 0

        # 检查每个方向的连子数量
        for d in directions:
            count = 1  # 包括当前落子
            # 正方向
            i, j = row + d[0], col + d[1]
            while 0 <= i < self.board_size and 0 <= j < self.board_size and self.board[i, j] == player:
                count += 1
                i += d[0]
                j += d[1]
            # 反方向
            i, j = row - d[0], col - d[1]
            while 0 <= i < self.board_size and 0 <= j < self.board_size and self.board[i, j] == player:
                count += 1
                i -= d[0]
                j -= d[1]
            max_connected = max(max_connected, count)

        # 根据连子数量返回奖励
        if max_connected >= 4:
            return 0.5  # 接近胜利
        elif max_connected == 3:
            return 0.2  # 有潜力
        elif max_connected == 2:
            return 0.1  # 小进展
        return 0  # 无特别进展

    # 修改 GomokuEnv 的 render 方法
    def render(self, screen, cell_size=50):
        # 修改背景颜色和网格颜色
        screen.fill((245, 222, 179))  # 米白色背景

        # 绘制更粗的网格线
        for i in range(self.board_size):
            pygame.draw.line(screen, (139, 69, 19),  # 深棕色线条
                             (0, i * cell_size + cell_size // 2),
                             (self.board_size * cell_size, i * cell_size + cell_size // 2),
                             3)
            pygame.draw.line(screen, (139, 69, 19),
                             (i * cell_size + cell_size // 2, 0),
                             (i * cell_size + cell_size // 2, self.board_size * cell_size),
                             3)

        # 添加棋盘星位标记
        star_points = [(2, 2), (2, 6), (6, 2), (6, 6), (4, 4)]
        for (i, j) in star_points:
            pygame.draw.circle(screen, (139, 69, 19),
                               (j * cell_size + cell_size // 2, i * cell_size + cell_size // 2),
                               5)

        # 美化棋子样式
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 1:
                    # 为黑棋添加光泽效果
                    pygame.draw.circle(screen, (50, 50, 50),
                                       (j * cell_size + cell_size // 2, i * cell_size + cell_size // 2),
                                       cell_size // 2 - 4)
                    pygame.draw.circle(screen, (100, 100, 100),
                                       (j * cell_size + cell_size // 2 + 2, i * cell_size + cell_size // 2 - 2),
                                       8)
                elif self.board[i, j] == 2:
                    # 为白棋添加渐变效果
                    pygame.draw.circle(screen, (200, 0, 0),
                                       (j * cell_size + cell_size // 2, i * cell_size + cell_size // 2),
                                       cell_size // 2 - 4)
                    pygame.draw.circle(screen, (255, 100, 100),
                                       (j * cell_size + cell_size // 2 + 2, i * cell_size + cell_size // 2 - 2),
                                       8)
        pygame.display.flip()


# 训练模型
def train_dqn_agent(episodes=50000):
    env = GomokuEnv()
    state_dim = 81
    action_dim = 81
    agent = DQNAgent(state_dim, action_dim)
    win_rate = []
    reward_history = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action_idx = agent.choose_action(state, env)
            i, j = action_idx // 9, action_idx % 9
            next_state, reward, done, info = env.step((i, j))
            agent.replay_buffer.push((state, action_idx, reward, next_state, done))
            agent.learn()
            episode_reward += reward
            state = next_state

        reward_history.append(episode_reward)

        if episode % 100 == 0 and episode > 0:
            agent.update_target_network()
            avg_loss = np.mean(agent.loss_history[-1000:]) if agent.loss_history else 0
            avg_reward = np.mean(reward_history[-100:])
            win_count = 0

            for _ in range(100):
                state = env.reset()
                done = False
                while not done:
                    action_idx = agent.choose_action(state, env)
                    i, j = action_idx // 9, action_idx % 9
                    next_state, reward, done, _ = env.step((i, j))
                    state = next_state
                if env.winner == 1:
                    win_count += 1
            current_win_rate = win_count / 100
            win_rate.append(current_win_rate)

            progress = episode / episodes * 100
            print(f"{'-' * 50}")
            print(f"Episode: {episode}/{episodes} ({progress:.1f}%)")
            print(f"Epsilon: {agent.epsilon:.4f}")
            print(f"Avg Loss (last 1000 steps): {avg_loss:.4f}")
            print(f"Avg Reward (last 100 episodes): {avg_reward:.2f}")
            print(f"Win Rate (last 100 tests): {current_win_rate:.2%}")
            print(f"{'-' * 50}")

    # Replace plt.show() with saving to a file
    plt.plot(range(100, episodes, 100), win_rate, label="Win Rate")
    plt.xlabel("Episodes")
    plt.ylabel("Win Rate")
    plt.title("Training Win Rate")
    plt.legend()
    plt.savefig("training_win_rate.png")  # Save plot as an image
    plt.close()  # Close the figure to free memory

    timestamp = int(time.time())
    filename = f"dqn_model_{timestamp}.pth"
    torch.save(agent.model.state_dict(), filename)
    print(f"训练完成！模型保存为 {filename}")

    return agent


# 人机对战界面
def play_game(agent):
    pygame.init()
    board_size = 9
    cell_size = 50
    width = cell_size * board_size
    height = cell_size * board_size
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("五子棋 - 人机对战 (黑子：智能体，红子：玩家)")

    env = GomokuEnv(board_size=board_size)
    state = env.reset()

    running = True
    clock = pygame.time.Clock()

    while running:
        env.render(screen, cell_size=cell_size)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.MOUSEBUTTONDOWN and env.current_player == 2:
                pos = pygame.mouse.get_pos()
                x, y = pos
                j = x // cell_size
                i = y // cell_size
                if (i, j) in env.available_actions():
                    state, reward, done, info = env.step((i, j))

        if env.current_player == 1 and not env.done:
            action_idx = agent.choose_action(state)  # 获取动作索引
            i = action_idx // 9
            j = action_idx % 9
            state, reward, done, info = env.step((i, j))

        if env.done:
            env.render(screen, cell_size=cell_size)
            font = pygame.font.SysFont('simhei', 48)  # 使用中文字体
            if env.winner == 1:
                text = font.render("AI Wins!", True, (0, 0, 0))
            elif env.winner == 2:
                text = font.render("Player Wins!", True, (255, 0, 0))
            else:
                text = font.render("Draw!", True, (0, 0, 255))

            # 创建半透明背景
            text_surface = pygame.Surface((text.get_width() + 20, text.get_height() + 20), pygame.SRCALPHA)
            text_surface.fill((255, 255, 255, 128))  # 白色半透明背景
            screen.blit(text_surface,
                        (width // 2 - text_surface.get_width() // 2, height // 2 - text_surface.get_height() // 2))

            # 居中显示文字
            text_rect = text.get_rect(center=(width // 2, height // 2))
            screen.blit(text, text_rect)
            pygame.display.flip()
            pygame.time.wait(3000)
            running = False

        clock.tick(30)
    pygame.quit()
    sys.exit()


# 主程序入口
if __name__ == '__main__':
    mode = input("请选择模式：1-训练, 2-人机对战 (训练模式将花费较长时间)：")
    if mode.strip() == "1":
        trained_agent = train_dqn_agent(episodes=50000)
    else:
        try:
            trained_agent = DQNAgent(state_dim=81, action_dim=81)
            trained_agent.model.load_state_dict(torch.load("dqn_model.pth", map_location=device))
            trained_agent.model.eval()
            print("模型加载成功！")
        except FileNotFoundError:
            print("未找到模型，正在训练...")
            trained_agent = train_dqn_agent(episodes=50000)
        play_game(trained_agent)
