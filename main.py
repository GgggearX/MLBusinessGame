import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pygame
import sys

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义 Q 网络（深度 Q 网络）
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)  # 增加神经元数量
        self.fc2 = nn.Linear(256, 256)        # 隐藏层
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
    def __init__(self, capacity=10000):
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
                 learning_rate=0.0001,  # 降低学习率
                 gamma=0.99,
                 epsilon=0.9,
                 epsilon_decay=0.9995,  # 减缓 epsilon 衰减
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

    def choose_action(self, state):
        """
        ε-greedy 策略选择动作：
          以 epsilon 概率随机选择，或选取具有最高 Q 值的动作
        """
        if random.random() < self.epsilon:
            # 随机选择一个位置 (i, j)，然后将其映射为一个索引
            i, j = random.choice([(i, j) for i in range(9) for j in range(9)])
            action = i * 9 + j  # 将 (i, j) 转换为一个线性索引
            return action
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = self.model(state)
            max_action_idx = torch.argmax(q_values).item()  # 获取具有最高 Q 值的动作索引
            return max_action_idx  # 直接返回索引

    def learn(self):
        """
        从经验回放池中随机抽取批次进行 Q-learning 更新
        """
        if self.replay_buffer.size() < self.batch_size:
            return

        transitions = self.replay_buffer.sample(self.batch_size)
        batch = list(zip(*transitions))
        state_batch = torch.tensor(batch[0], dtype=torch.float32).to(device)
        action_batch = torch.tensor(batch[1], dtype=torch.long).to(device)  # action_batch 应该是一个形状为 (batch_size,)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32).to(device)
        next_state_batch = torch.tensor(batch[3], dtype=torch.float32).to(device)
        done_batch = torch.tensor(batch[4], dtype=torch.float32).to(device)

        # 计算当前 Q 值
        q_values = self.model(state_batch)  # q_values 的形状为 (batch_size, action_dim)

        # 使用 gather 选择特定动作的 Q 值
        q_value = q_values.gather(1, action_batch.unsqueeze(1))  # 将 action_batch 转换为 (batch_size, 1)
        q_value = q_value.squeeze(1)  # 去掉额外的维度，得到 (batch_size,)

        # 计算下一状态的最大 Q 值
        with torch.no_grad():
            next_q_values = self.target_model(next_state_batch)
            next_q_value = next_q_values.max(1)[0]  # 选择最大值
            target = reward_batch + self.gamma * next_q_value * (1 - done_batch)

        # 计算损失并更新模型
        loss = self.loss_fn(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ε 衰减
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        """
        定期更新目标网络
        """
        self.target_model.load_state_dict(self.model.state_dict())


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
def train_dqn_agent(episodes=1000):
    env = GomokuEnv()
    state_dim = 81  # 9x9 五子棋棋盘
    action_dim = 81  # 每个位置可能的动作
    agent = DQNAgent(state_dim, action_dim)

    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action_idx = agent.choose_action(state)  # 获取动作索引
            i = action_idx // 9  # 转换为行
            j = action_idx % 9   # 转换为列
            next_state, reward, done, info = env.step((i, j))  # 传递元组
            agent.replay_buffer.push((state, action_idx, reward, next_state, done))
            agent.learn()
            state = next_state

        if episode % 100 == 0:
            agent.update_target_network()
            print(f"Episode {episode}/{episodes}, Epsilon: {agent.epsilon:.3f}")

    # 保存训练好的模型
    torch.save(agent.model.state_dict(), "dqn_model.pth")
    print("训练完成！")
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
        trained_agent = train_dqn_agent(episodes=1000)
    else:
        try:
            # 尝试加载预训练模型
            trained_agent = DQNAgent(state_dim=81, action_dim=81)
            trained_agent.model.load_state_dict(torch.load("dqn_model.pth", map_location=device))
            trained_agent.model.eval()  # 设置为评估模式
            print("模型加载成功！")
        except FileNotFoundError:
            print("未找到模型，正在训练...")
            trained_agent = train_dqn_agent(episodes=1000)  # 训练后会保存模型
        play_game(trained_agent)