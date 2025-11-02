import random
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

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
    def __init__(self):
        super(ValueNet, self).__init__()

        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(300, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.fc3(x)
        return out


class PPO():
    def __init__(self, input_shape=[10, 10], up_time=10, batch_size=32, a_lr=1e-5, b_lr=1e-5, gama=0.9, epsilon=0.1):
        self.up_time = up_time  # 每次更新的迭代次数
        self.batch_size = batch_size  # 批量大小
        self.gama = gama  # 折扣因子（未来奖励的衰减系数）
        self.epsilon = epsilon  # PPO的clip参数（限制策略更新幅度）
        self.suffer = []  # 经验回放缓冲区（存储状态、动作、奖励等）
        self.action = PolicyNet(input_shape)  # 动作网络
        self.action.to(device)
        self.value = ValueNet()  # 价值网络
        self.value.to(device)
        self.action_optim = torch.optim.Adam(self.action.parameters(), lr=a_lr)  # 动作网络优化器
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=b_lr)  # 价值网络优化器
        self.loss = nn.MSELoss().to(device)  # 价值网络的损失函数（均方误差）

    def append(self, buffer):
        self.suffer.append(buffer)  # 将单条经验（状态、动作、奖励等）加入缓冲区

    def load_net(self, path):
        self.action = torch.load(path)  # 从路径加载动作网络参数

    def get_action(self, x):
        x = x.unsqueeze(dim=0).to(device)  # 扩展为batch维度 [1, 2, H, W]
        ac_prob = self.action(x)  # 得到动作概率分布 [1, N]（N为动作数）
        a = Categorical(ac_prob).sample()[0]  # 按概率分布采样动作
        ac_pro = ac_prob[0][a]  # 记录该动作的概率
        return [a.item()], [ac_pro.item()]  # 返回动作和对应的概率

    def update(self):
        states = torch.stack([t.state for t in self.suffer], dim=0).to(device)
        actions = torch.tensor([t.ac for t in self.suffer], dtype=torch.int).to(device)
        rewards = [t.reward for t in self.suffer]
        done = [t.done for t in self.suffer]
        old_probs = torch.tensor([t.ac_prob for t in self.suffer], dtype=torch.float32).to(device)  # .detach()

        false_indexes = [i + 1 for i, val in enumerate(done) if not val]
        if len(false_indexes) >= 0:
            idx, reward_all = 0, []
            for i in false_indexes:
                reward = rewards[idx:i]
                R = 0
                Rs = []
                reward.reverse()
                for r in reward:
                    R = r + R * self.gama
                    Rs.append(R)
                Rs.reverse()
                reward_all.extend(Rs)
                idx = i
        else:
            R = 0
            reward_all = []
            rewards.reverse()
            for r in rewards:
                R = r + R * self.gama
                reward_all.append(R)
            reward_all.reverse()
        Rs = torch.tensor(reward_all, dtype=torch.float32).to(device)
        for _ in range(self.up_time):
            self.action.train()
            self.value.train()
            for n in range(max(10, int(10 * len(self.suffer) / self.batch_size))):
                # 生成 reward_all 后转换为 Rs 张量
                Rs = torch.tensor(reward_all, dtype=torch.float32).to(device)
                rs_length = Rs.size(0)

                # 1. 核心校验：确保 rs_length > 0，否则直接返回不更新
                if rs_length == 0:
                    print("警告：奖励序列为空，跳过本次更新（可能是游戏未产生有效奖励）")
                    return

                # 2. 确保 sample_size 有效（至少为1，且不超过 rs_length）
                sample_size = min(self.batch_size, rs_length)
                sample_size = max(1, sample_size)  # 防止 sample_size 为0

                # 3. 生成索引（此时 rs_length 必然 > 0，且 sample_size 有效）
                index = torch.randint(0, rs_length, (sample_size,), dtype=torch.int64, device=device)

                # 4. 索引选择操作（添加额外校验，确保索引有效）
                try:
                    v_target = torch.index_select(Rs, dim=0, index=index).unsqueeze(dim=1)
                except RuntimeError as e:
                    print(f"索引选择失败：{e}")
                    print(f"Rs 长度: {rs_length}, 索引范围: [{index.min()}, {index.max()}]")
                    return  # 出错时跳过本次更新

                v = self.value(torch.index_select(states, 0, index))
                adta = v_target - v
                adta = adta.detach()
                probs = self.action(torch.index_select(states, 0, index))
                pro_index = torch.index_select(actions, 0, index).to(torch.int64)

                probs_a = torch.gather(probs, 1, pro_index)
                ratio = probs_a / torch.index_select(old_probs, 0, index).to(device)
                surr1 = ratio * adta
                surr2 = torch.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * adta.to(device)
                action_loss = -torch.mean(torch.minimum(surr1, surr2))
                self.action_optim.zero_grad()
                action_loss.backward(retain_graph=True)
                self.action_optim.step()
                value_loss = self.loss(v_target, v)
                self.value_optim.zero_grad()
                value_loss.backward()
                self.value_optim.step()
        self.suffer = []
