from collections import namedtuple

import time

from env import Minesweeper, DIFFICULTIES
from ppo import PPO
from tqdm import tqdm
from torch.distributions.categorical import Categorical
from pyecharts.charts import Line
import torch
import numpy as np


batch_size = 32
a_lr = 0.0001
b_lr = 0.002
gama = 0.995
epsilon = 0.2
up_time = 10
epoch = 50

Transition = namedtuple('Transition', ['state', 'ac', 'ac_prob', 'reward', 'done'])


def get_action_coords(action_idx, width, height):
    """将动作索引转换为网格坐标"""
    # 确保索引在有效范围内
    action_idx = np.clip(action_idx, 0, width * height - 1)
    x = action_idx // width
    y = action_idx % width
    return x, y


def test_get_action(state, net, width, height):
    """
    测试阶段获取动作和概率分布

    参数:
        state: 当前游戏状态
        net: 神经网络模型
        width: 网格宽度
        height: 网格高度

    返回:
        (x, y): 动作坐标
        prob_dist: 概率分布
    """
    # 添加批次维度并获取动作概率
    state = state.unsqueeze(dim=0)
    action_probs = net(state)

    # 选择概率最高的15个动作并采样
    top_values, top_indices = action_probs.topk(k=15, dim=1)
    selected_idx = Categorical(top_values).sample()[0].item()
    action_idx = top_indices[0, selected_idx].item()

    # 转换为坐标并重塑概率分布
    coords = get_action_coords(action_idx, width, height)
    prob_dist = action_probs.detach().numpy().reshape([1, height, width])

    return coords, prob_dist


def model_train(times, settings):
    env = Minesweeper(difficulty=difficulty, window=False)
    net = PPO(input_shape=[mine_settings["width"], mine_settings["height"]], up_time=up_time, batch_size=batch_size, a_lr=a_lr, b_lr=b_lr, gama=gama, epsilon=epsilon)

    Rs = []  # 存储每一局游戏结束后的总奖励

    for i in range(times):
        with tqdm(total=epoch, desc='Iteration %d' % i) as pbar:
            # tqdm用于显示进度条，total=epoch表示每个times轮包含epoch局游戏
            for e in range(epoch):
                # 每局游戏的初始化与交互
                env.reset()  # 重置游戏环境（开始新一局）
                s = torch.tensor(env.get_status(), dtype=torch.float32)  # 获取初始状态并转为Tensor
                # 游戏循环：直到游戏结束（踩雷或获胜）或步数超过51（防止无限循环）
                while env.condition and env.t < 91:
                    # 智能体选择动作
                    a, a_p = net.get_action(s)  # a是动作索引，a_p是动作概率
                    at = get_action_coords(a[0], settings["width"], settings["height"])  # 将动作索引转换为游戏可理解的格式（如坐标(x,y)）
                    # 与环境交互：执行动作，获取新状态、奖励、是否结束
                    [s_t, r, d] = env.agent_step(at)
                    # 存储经验到PPO的缓冲区
                    buffer = Transition(s, a, a_p, r, d)  # Transition是一个经验数据类
                    net.append(buffer)  # 将经验加入PPO的suffer缓冲区
                    s = s_t  # 更新当前状态为新状态
                # 一局游戏结束后处理
                R = np.array(env.R).sum()  # 计算本局游戏的总奖励（假设env.R存储每步奖励）
                Rs.append(R)  # 记录总奖励
                # 当经验缓冲区大小超过batch_size时，更新PPO模型
                if len(net.suffer) > batch_size:
                    net.update()  # 调用PPO的update方法更新网络
                # 更新进度条显示
                pbar.set_postfix({'return': '%.2f' % R})  # 显示本局的总奖励
                pbar.update(1)  # 进度条加1

    torch.save(net.action,'net_model.pt')
    Re = []
    for i in range(int(len(Rs)/50)):
        idx = i*50
        Re.append(sum(Rs[idx:idx+50])/50)
    x = [str(i) for i in range(len(Re))]
    line = Line()
    line.add_xaxis(xaxis_data=x)
    line.add_yaxis(y_axis=Re, series_name='Recall')
    line.render('result.html')

    net.plot_training_curves()


def test(path, settings):
    env = Minesweeper(difficulty=difficulty, window=True)
    net = torch.load(path, weights_only=False)
    device = torch.device("cpu")
    net = net.to(device)
    s = torch.tensor(env.get_status(), dtype=torch.float32)
    a_p = 0
    for i in range(10):
        k = 0
        while env.condition:
            a, a_p = test_get_action(s, net, width=settings["width"], height=settings["height"])
            [s_t, r, d] = env.agent_step(a)
            print("第{}步：动作a = {}, 奖励r = {}".format(k, a, r))
            k = k + 1
            s = s_t
            time.sleep(1.)
        env.reset()


if __name__ == '__main__':
    difficulty = "简单"
    mine_settings = DIFFICULTIES[difficulty]
    model_train(times=20, settings=mine_settings)

    # model_path = 'net_model.pt'
    # test(path=model_path, settings=mine_settings)
