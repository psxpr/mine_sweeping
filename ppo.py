import random
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
matplotlib.use('TkAgg')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 策略网络
class PolicyNet(nn.Module):
    """卷积神经网络策略网络（适用于网格类状态，如扫雷）
    优化点：使用 nn.Flatten() 替代手动 view 展平，移除类型注解简化代码
    """

    def __init__(
            self,
            input_shape=(10, 10),  # 网格尺寸 (高, 宽)
            in_channels=3  # 输入通道数（扫雷：状态通道+数字通道+安全概率）
    ):
        super(PolicyNet, self).__init__()
        self.input_shape = input_shape
        self.in_channels = in_channels

        # 卷积特征提取 + 展平层（一体化设计）
        self.features = nn.Sequential(
            # 第一层卷积：提取基础空间特征
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1  # 保持特征图尺寸与输入一致
            ),
            nn.ReLU(inplace=True),  # inplace=True 节省内存

            # 第二层卷积：加深特征提取
            nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # 第三层卷积：压缩特征通道
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # 输出层卷积：将通道数压缩至1（对应动作概率的空间分布）
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),

            # 展平层：将 (batch, 1, H, W) 展平为 (batch, 1*H*W)
            nn.Flatten(start_dim=1)
        )

        # 动作概率归一化
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """前向传播：输入状态 -> 卷积特征提取 -> 展平 -> 动作概率分布"""
        # 特征提取 + 展平
        x = self.features(x)  # 输出形状：(batch_size, H*W)

        # 概率归一化
        action_probs = self.softmax(x)

        return action_probs


# 价值网络
class ValueNet(nn.Module):
    def __init__(self, input_shape=(10, 10), in_channels=3):
        super(ValueNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * input_shape[0] * input_shape[1], 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.features(x)


class PPO():
    def __init__(self, input_shape=(10, 10), up_time=10, batch_size=32, a_lr=1e-5, b_lr=1e-5, gama=0.9, epsilon=0.1):
        self.up_time = up_time  # 每次更新的迭代次数
        self.batch_size = batch_size  # 批量大小
        self.gama = gama  # 折扣因子（未来奖励的衰减系数）
        self.epsilon = epsilon  # PPO的clip参数（限制策略更新幅度）
        self.suffer = []  # 经验回放缓冲区（存储状态、动作、奖励等）
        self.action = PolicyNet(input_shape)  # 动作网络
        self.action.to(device)
        self.value = ValueNet(input_shape)  # 价值网络
        self.value.to(device)
        self.action_optim = torch.optim.Adam(self.action.parameters(), lr=a_lr)  # 动作网络优化器
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=b_lr)  # 价值网络优化器
        self.loss = nn.MSELoss().to(device)  # 价值网络的损失函数（均方误差）

        # 训练日志记录
        self.logs = defaultdict(list)
        self.episode_rewards = []  # 记录每个episode的总奖励
        self.current_episode_reward = 0  # 当前episode的累计奖励
        self.update_count = 0  # 更新次数计数器
        self.total_episodes = 0  # 总训练回合数
        self.total_steps = 0  # 总训练步数

    def append(self, buffer):
        self.suffer.append(buffer)  # 将单条经验（状态、动作、奖励等）加入缓冲区

        self.current_episode_reward += buffer.reward
        self.total_steps += 1  # 累计总步数
        if buffer.done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0  # 重置当前episode奖励
            self.total_episodes += 1  # 累计总回合数

    def load_net(self, path):
        self.action = torch.load(path)  # 从路径加载动作网络参数

    def save_checkpoint(self, path="checkpoint.pt"):
        """保存模型 checkpoint（支持续训）"""
        checkpoint = {
            # 模型参数
            "action_state_dict": self.action.state_dict(),
            "value_state_dict": self.value.state_dict(),
            # 优化器状态
            "action_optim_state_dict": self.action_optim.state_dict(),
            "value_optim_state_dict": self.value_optim.state_dict(),
            # 训练进度
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "episode_rewards": self.episode_rewards,
            "update_count": self.update_count,
            "logs": self.logs,
            # 随机种子状态
            "rng_state": torch.random.get_rng_state()
        }
        # 确保保存目录存在
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(checkpoint, path)
        print(f"已保存 checkpoint 到 {path}，回合数：{self.total_episodes}，总步数：{self.total_steps}")

    def load_checkpoint(self, path="checkpoint.pt"):
        """加载模型 checkpoint（续训入口）"""
        if not os.path.exists(path):
            print(f"未找到 {path}，将从头开始训练")
            return

        checkpoint = torch.load(path, map_location=device)

        # 恢复模型参数
        self.action.load_state_dict(checkpoint["action_state_dict"])
        self.value.load_state_dict(checkpoint["value_state_dict"])

        # 恢复优化器状态（关键：保证续训时优化方向连续）
        self.action_optim.load_state_dict(checkpoint["action_optim_state_dict"])
        self.value_optim.load_state_dict(checkpoint["value_optim_state_dict"])

        # 恢复训练进度
        self.total_episodes = checkpoint["total_episodes"]
        self.total_steps = checkpoint["total_steps"]
        self.episode_rewards = checkpoint["episode_rewards"]
        self.update_count = checkpoint["update_count"]
        self.logs = checkpoint["logs"]

        # 恢复随机种子状态
        torch.random.set_rng_state(checkpoint["rng_state"])

        print(f"已加载 checkpoint 从 {path}，继续训练：回合数 {self.total_episodes}，总步数 {self.total_steps}")

    def get_action(self, x):
        x = x.unsqueeze(dim=0).to(device)  # 扩展为batch维度 [1, 2, H, W]
        ac_prob = self.action(x)  # 得到动作概率分布 [1, N]（N为动作数）
        a = Categorical(ac_prob).sample()[0]  # 按概率分布采样动作
        ac_pro = ac_prob[0][a]  # 记录该动作的概率
        return [a.item()], [ac_pro.item()]  # 返回动作和对应的概率

    def update(self):
        if not self.suffer:
            return

        states = torch.stack([t.state for t in self.suffer], dim=0).to(device)
        actions = torch.tensor([t.ac for t in self.suffer], dtype=torch.int).to(device)
        rewards = [t.reward for t in self.suffer]
        done = [t.done for t in self.suffer]
        old_probs = torch.tensor([t.ac_prob for t in self.suffer], dtype=torch.float32).to(device)  # .detach()

        reward_all = []
        start_idx = 0  # 记录当前episode的起始索引
        for i in range(len(done)):
            if i == len(done) - 1 or done[i]:  # 到达末尾或episode结束
                episode_rewards = rewards[start_idx:i + 1]  # 提取当前episode的所有奖励
                R = 0
                episode_returns = []
                for r in reversed(episode_rewards):  # 从后向前计算折扣回报
                    R = r + self.gama * R
                    episode_returns.append(R)
                reward_all.extend(reversed(episode_returns))  # 保持时间顺序
                start_idx = i + 1  # 移动到下一个episode的起始位置
        # 如果最后一个 episode 未终止，强制截断
        if start_idx < len(rewards):
            episode_rewards = rewards[start_idx:]
            R = 0
            episode_returns = []
            for r in reversed(episode_rewards):
                R = r + self.gama * R
                episode_returns.append(R)
            reward_all.extend(reversed(episode_returns))

        # 更新网络
        self.action.train()
        self.value.train()
        Rs = torch.tensor(reward_all, dtype=torch.float32).to(device)

        total_action_loss = 0
        total_value_loss = 0
        total_advantage = 0
        total_ratio = 0

        for _ in range(self.up_time):
            for n in range(max(10, int(10 * len(self.suffer) / self.batch_size))):
                index = torch.tensor(random.sample(range(len(self.suffer)), self.batch_size),
                                     dtype=torch.int64).to(device)
                v_target = torch.index_select(Rs, dim=0, index=index).unsqueeze(dim=1)
                v = self.value(torch.index_select(states, 0, index))

                # 计算优势函数
                adta = v_target - v
                adta_detach = adta.detach()

                # 计算策略损失
                probs = self.action(torch.index_select(states, 0, index))
                pro_index = torch.index_select(actions, 0, index).to(torch.int64)
                probs_a = torch.gather(probs, 1, pro_index)
                ratio = probs_a / torch.index_select(old_probs, 0, index).to(device)

                surr1 = ratio * adta_detach
                surr2 = torch.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * adta_detach
                action_loss = -torch.mean(torch.minimum(surr1, surr2))

                # 更新策略网络
                self.action_optim.zero_grad()
                action_loss.backward(retain_graph=True)
                self.action_optim.step()

                # 计算并更新价值网络
                value_loss = self.loss(v_target, v)
                self.value_optim.zero_grad()
                value_loss.backward()
                self.value_optim.step()

                # 累计统计信息
                total_action_loss += action_loss.item()
                total_value_loss += value_loss.item()
                total_advantage += torch.mean(torch.abs(adta)).item()
                total_ratio += torch.mean(ratio).item()

                # 记录每步更新的损失
                self.logs['step_action_loss'].append(action_loss.item())
                self.logs['step_value_loss'].append(value_loss.item())

        # 计算平均统计信息并记录
        num_steps = self.up_time * max(10, int(10 * len(self.suffer) / self.batch_size))
        self.update_count += 1
        self.logs['update_action_loss'].append(total_action_loss / num_steps)
        self.logs['update_value_loss'].append(total_value_loss / num_steps)
        self.logs['update_advantage'].append(total_advantage / num_steps)
        self.logs['update_ratio'].append(total_ratio / num_steps)

        # 清空经验池
        self.suffer = []

    def plot_training_curves(self, smooth_window=10):
        """绘制训练曲线"""
        # 设置中文字体
        plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Heiti TC", "WenQuanYi Micro Hei", "SimSun"]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('PPO扫雷训练监控', fontsize=16)

        # 1. 每步损失曲线
        axes[0, 0].plot(self.logs['step_action_loss'], label='策略损失', alpha=0.3)
        axes[0, 0].plot(self.logs['step_value_loss'], label='价值损失', alpha=0.3)
        axes[0, 0].set_title('每步损失变化')
        axes[0, 0].set_xlabel('更新步骤')
        axes[0, 0].set_ylabel('损失值')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 2. 每轮更新的平均损失
        axes[0, 1].plot(self.logs['update_action_loss'], label='平均策略损失')
        axes[0, 1].plot(self.logs['update_value_loss'], label='平均价值损失')
        axes[0, 1].set_title('每轮更新平均损失')
        axes[0, 1].set_xlabel('更新轮次')
        axes[0, 1].set_ylabel('平均损失值')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 3. 优势函数和策略更新比例
        axes[1, 0].plot(self.logs['update_advantage'], label='平均优势函数绝对值')
        axes[1, 0].set_title('优势函数变化')
        axes[1, 0].set_xlabel('更新轮次')
        axes[1, 0].set_ylabel('优势函数值')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        axes[1, 1].plot(self.logs['update_ratio'], label='策略更新比例')
        axes[1, 1].axhline(y=1.0, color='r', linestyle='--', label='基准值1.0')
        axes[1, 1].axhline(y=1 + self.epsilon, color='g', linestyle='--', label=f'上限{1 + self.epsilon}')
        axes[1, 1].axhline(y=1 - self.epsilon, color='g', linestyle='--', label=f'下限{1 - self.epsilon}')
        axes[1, 1].set_title('策略更新比例')
        axes[1, 1].set_xlabel('更新轮次')
        axes[1, 1].set_ylabel('新旧策略比例')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        # 单独绘制奖励曲线
        plt.figure(figsize=(12, 6))
        plt.plot(self.episode_rewards, label='每局奖励', alpha=0.5)

        # 平滑奖励曲线
        if len(self.episode_rewards) >= smooth_window:
            smoothed_rewards = np.convolve(
                self.episode_rewards,
                np.ones(smooth_window) / smooth_window,
                mode='valid'
            )
            plt.plot(range(smooth_window - 1, len(self.episode_rewards)),
                     smoothed_rewards, label=f'{smooth_window}局平均奖励', color='red')

        plt.title('每局奖励变化')
        plt.xlabel('局数')
        plt.ylabel('奖励值')
        plt.legend()
        plt.grid(True)
        plt.show()
