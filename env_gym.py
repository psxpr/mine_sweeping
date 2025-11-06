import pygame
import random
import sys
import time
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

# ---------------------- 基础配置与工具函数 ----------------------
# 初始化pygame
pygame.init()

# 颜色定义
WHITE = (255, 255, 255)
GRAY = (192, 192, 192)
LIGHT_GRAY = (220, 220, 220)
DARK_GRAY = (100, 100, 100)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 180, 0)
YELLOW = (255, 200, 0)
PURPLE = (128, 0, 128)
BUTTON_BG = (240, 240, 240)
BUTTON_HOVER = (220, 220, 255)
RESTART_BTN_NORMAL = (0, 160, 230)
RESTART_BTN_HOVER = (0, 190, 255)

# 游戏难度设置
DIFFICULTIES = {
    "简单": {"width": 10, "height": 10, "mines": 10},
    "中等": {"width": 16, "height": 16, "mines": 40},
    "困难": {"width": 30, "height": 16, "mines": 99}
}

# 中文字体加载
pygame.font.init()


def get_chinese_font(size):
    chinese_fonts = ["SimHei", "Microsoft YaHei", "Heiti TC", "WenQuanYi Micro Hei", "SimSun"]
    for font_name in chinese_fonts:
        try:
            return pygame.font.SysFont(font_name, size)
        except:
            continue
    return pygame.font.SysFont(None, size)


small_font = get_chinese_font(16)
medium_font = get_chinese_font(20)
large_font = get_chinese_font(32)


# ---------------------- 原始扫雷核心类 ----------------------
class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.is_mine = False
        self.is_revealed = False
        self.mark_type = 0  # 0:无标记, 1:地雷标记, 2:问号标记
        self.adjacent_mines = 0


class MinesweeperCore:
    """扫雷核心逻辑类（剥离界面与智能体交互，仅保留核心功能）"""

    def __init__(self, difficulty="简单"):
        self.difficulty = difficulty
        self.settings = DIFFICULTIES[difficulty]
        self.width = self.settings["width"]
        self.height = self.settings["height"]
        self.mines_count = self.settings["mines"]

        # 游戏状态
        self.cells = []
        self.mines_placed = False
        self.game_over = False
        self.victory = False
        self.safety_prob_cache = None

        # 智能体相关变量
        self.count = np.zeros([self.width, self.height])
        self.map = np.zeros([self.width, self.height])
        self.t = 0
        self.total_safe = self.width * self.height - self.mines_count
        self.stage_reward = False

        self.reset()

    def reset(self):
        """重置游戏状态"""
        self.cells = [[Cell(x, y) for y in range(self.height)] for x in range(self.width)]
        self.mines_placed = False
        self.game_over = False
        self.victory = False
        self.safety_prob_cache = None
        self.stage_reward = False

        self.count = np.zeros([self.width, self.height])
        self.map = np.zeros([self.width, self.height])
        self.t = 0

    def set_difficulty(self, difficulty):
        """更改难度"""
        if difficulty in DIFFICULTIES:
            self.difficulty = difficulty
            self.settings = DIFFICULTIES[difficulty]
            self.width = self.settings["width"]
            self.height = self.settings["height"]
            self.mines_count = self.settings["mines"]
            self.total_safe = self.width * self.height - self.mines_count
            self.reset()

    def place_mines(self, first_click_x, first_click_y):
        """第一次点击后放置地雷"""
        safe_zone = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = first_click_x + dx, first_click_y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    safe_zone.append((nx, ny))

        mines_placed = 0
        while mines_placed < self.mines_count:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if (x, y) not in safe_zone and not self.cells[x][y].is_mine:
                self.cells[x][y].is_mine = True
                mines_placed += 1

        self.calculate_adjacent_mines()
        self.mines_placed = True

    def calculate_adjacent_mines(self):
        """计算周围地雷数"""
        for x in range(self.width):
            for y in range(self.height):
                if not self.cells[x][y].is_mine:
                    count = 0
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.width and 0 <= ny < self.height and self.cells[nx][ny].is_mine:
                                count += 1
                    self.cells[x][y].adjacent_mines = count

    def calculate_safety_prob(self):
        """计算安全概率"""
        safety_prob = np.ones((self.width, self.height))
        revealed = np.array([[cell.is_revealed for cell in row] for row in self.cells])
        marked = np.array([[cell.mark_type == 1 for cell in row] for row in self.cells])

        # 处理数字格子相邻区域
        for x in range(self.width):
            for y in range(self.height):
                cell = self.cells[x][y]
                if not cell.is_revealed or cell.adjacent_mines == 0:
                    continue

                unknown_count = 0
                marked_count = 0
                unknown_positions = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            if not revealed[nx][ny] and not marked[nx][ny]:
                                unknown_count += 1
                                unknown_positions.append((nx, ny))
                            if marked[nx][ny]:
                                marked_count += 1

                remaining_mines = max(0, cell.adjacent_mines - marked_count)
                if unknown_count == 0:
                    continue
                if remaining_mines == 0:
                    for (nx, ny) in unknown_positions:
                        safety_prob[nx][ny] = 1.0
                elif remaining_mines == unknown_count:
                    for (nx, ny) in unknown_positions:
                        safety_prob[nx][ny] = 0.0
                else:
                    prob = 1.0 - (remaining_mines / unknown_count)
                    for (nx, ny) in unknown_positions:
                        if prob < safety_prob[nx][ny]:
                            safety_prob[nx][ny] = prob

        # 处理孤立格子
        revealed_safe = sum(1 for row in self.cells for cell in row if cell.is_revealed and not cell.is_mine)
        remaining_safe = self.total_safe - revealed_safe
        remaining_unknown = sum(
            1 for x in range(self.width) for y in range(self.height) if not revealed[x][y] and not marked[x][y])

        if remaining_unknown > 0:
            isolated_prob = remaining_safe / remaining_unknown
            for x in range(self.width):
                for y in range(self.height):
                    if not revealed[x][y] and not marked[x][y] and safety_prob[x][y] == 1.0:
                        safety_prob[x][y] = isolated_prob

        # 标记无效区域
        for x in range(self.width):
            for y in range(self.height):
                if revealed[x][y] or marked[x][y]:
                    safety_prob[x][y] = -1.0

        self.safety_prob_cache = safety_prob
        return safety_prob

    def reveal_cell(self, x, y):
        """翻开格子"""
        if not 0 <= x < self.width or not 0 <= y < self.height:
            return False, False

        cell = self.cells[x][y]
        if cell.is_revealed or cell.mark_type != 0:
            return False, False

        cell.is_revealed = True
        if cell.is_mine:
            self.map[x, y] = -10
            self.game_over = True
            # 翻开所有格子
            for rx in range(self.width):
                for ry in range(self.height):
                    if not self.cells[rx][ry].is_revealed:
                        self.cells[rx][ry].is_revealed = True
            return True, True  # 踩雷，游戏结束
        else:
            self.map[x, y] = cell.adjacent_mines
            # 空白格子递归翻开
            if cell.adjacent_mines == 0:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            self.reveal_cell(x + dx, y + dy)
            return True, False  # 正常翻开，未结束

    def check_victory(self):
        """检查胜利"""
        if self.game_over:
            return False
        for x in range(self.width):
            for y in range(self.height):
                if not self.cells[x][y].is_mine and not self.cells[x][y].is_revealed:
                    return False
        self.victory = True
        return True


# ---------------------- Gymnasium环境封装 ----------------------
class MinesweeperGym(gym.Env):
    """Gymnasium标准化扫雷环境"""
    metadata = {"render_modes": ["human", "rgb_array", "none"], "render_fps": 30}

    def __init__(self, difficulty="简单", render_mode="none"):
        super().__init__()
        # 核心逻辑初始化
        self.core = MinesweeperCore(difficulty=difficulty)
        self.width = self.core.width
        self.height = self.core.height

        # 渲染模式配置
        assert render_mode in self.metadata["render_modes"], f"无效渲染模式: {render_mode}"
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.agent_current_click = None

        # 动作空间：离散动作（每个格子对应一个索引）
        self.action_space = spaces.Discrete(self.width * self.height)

        # 观测空间：[状态层, 点击计数层, 安全概率层]，形状[3, H, W]
        self.observation_space = spaces.Box(
            low=-10.0,  # 地雷标记为-10
            high=1.0,  # 安全概率最大值1.0
            shape=(3, self.height, self.width),
            dtype=np.float32
        )

    def _get_obs(self):
        """获取观测状态"""
        # 状态层：未翻开=-1，已翻开地雷=-10，已翻开安全区=周围地雷数
        status_layer = (np.array([[cell.is_revealed for cell in row] for row in self.core.cells],
                                 dtype=np.float32) - 1) + self.core.map
        # 安全概率层
        safety_layer = self.core.calculate_safety_prob()
        # 堆叠三层状态
        obs = np.stack((status_layer, self.core.count, safety_layer), axis=0)
        return torch.tensor(obs, dtype=torch.float32)

    def _get_info(self):
        """获取额外信息"""
        revealed_safe = sum(1 for row in self.core.cells for cell in row if not cell.is_mine and cell.is_revealed)
        return {
            "step": self.core.t,
            "remaining_safe": self.core.total_safe - revealed_safe,
            "remaining_mines": self.core.mines_count - sum(
                1 for row in self.core.cells for cell in row if cell.mark_type == 1),
            "safety_prob": self.core.safety_prob_cache,
            "victory": self.core.victory,
            "game_over": self.core.game_over
        }

    def step(self, action):
        """执行动作（Gymnasium标准接口）"""
        # 动作转换：离散索引 → (x, y)坐标
        x = action // self.height
        y = action % self.height
        reward = 0.0

        # 边界校验惩罚
        if not (0 <= x < self.width and 0 <= y < self.height):
            reward = -2.0
            done = True
            obs = self._get_obs()
            info = self._get_info()
            return obs, reward, done, False, info

        cell = self.core.cells[x][y]
        # 已翻开/已标记格子惩罚
        if cell.is_revealed or cell.mark_type != 0:
            reward = -1.0
            self.core.t += 1
            self.core.count[x][y] += 1
            done = False
            obs = self._get_obs()
            info = self._get_info()
            return obs, reward, done, False, info

        # 首次点击放置地雷
        if not self.core.mines_placed:
            self.core.place_mines(x, y)
            safety_prob = 1.0  # 首步点击绝对安全
        else:
            safety_prob = self.core.calculate_safety_prob()[x][y]

        # 执行翻开操作
        is_revealed, is_mine = self.core.reveal_cell(x, y)

        # 计算已翻开安全格子数
        revealed_safe = sum(
            1 for row in self.core.cells
            for cell in row
            if not cell.is_mine and cell.is_revealed
        )

        # 奖励计算
        if not self.core.game_over:
            # 基础探索奖励
            reward = 0.8

            # 安全概率奖励（鼓励选择高安全格子）
            if safety_prob >= 1.0:
                reward += 1.5  # 绝对安全格子额外奖励
            elif safety_prob >= 0.8:
                reward += 0.8  # 高安全格子奖励
            elif safety_prob >= 0.5:
                reward += 0.3  # 中等安全格子奖励
            else:
                reward -= 2.0  # 低安全格子惩罚

            # 高安全格子存在时的冒险惩罚
            safe_cells = sum(
                1 for cx in range(self.width)
                for cy in range(self.height)
                if self.core.calculate_safety_prob()[cx][cy] >= 0.8
            )
            if safe_cells > 0 and safety_prob < 0.5:
                reward -= 5.0  # 有高安全格子却选择冒险的额外惩罚

            # 阶段性奖励（接近胜利时触发）
            threshold = {
                "简单": max(1, int(self.core.total_safe * 0.2)),
                "中等": max(1, int(self.core.total_safe * 0.15)),
                "困难": max(1, int(self.core.total_safe * 0.1))
            }[self.core.difficulty]

            if not self.core.stage_reward and (
                    self.core.total_safe - threshold) <= revealed_safe < self.core.total_safe:
                reward += 20.0  # 接近胜利的阶段性奖励
                self.core.stage_reward = True

        # 终局状态奖励
        if self.core.game_over:  # 踩雷失败
            reward = -50.0
            done = True
        elif self.core.check_victory():  # 完全胜利
            # 胜利奖励 = 基础奖励 + 快速胜利加成（步数越少奖励越高）
            reward = 100.0 + (self.core.total_safe - self.core.t) * 0.05
            done = True
        else:
            done = False

        # 重复点击惩罚（累积惩罚机制）
        if self.core.count[x][y] > 0:
            reward -= 0.3 * self.core.count[x][y]  # 重复次数越多惩罚越重

        # 步数限制（防止无限循环）
        self.core.t += 1
        self.core.count[x][y] += 1
        max_steps = max(100, self.core.total_safe * 2)  # 最大步数适配难度
        if self.core.t >= max_steps:
            reward = -5.0  # 超时惩罚
            done = True

        # 记录当前点击位置（用于渲染高亮）
        self.agent_current_click = (x, y)

        # 渲染（如果需要）
        if self.render_mode in ["human", "rgb_array"]:
            self.render()

        # 准备返回值
        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, done, False, info  # Gymnasium要求返回 (obs, reward, terminated, truncated, info)

    def render(self):
        """渲染游戏界面（支持human和rgb_array模式）"""
        if self.render_mode == "none":
            return

        # 初始化窗口（首次渲染时）
        if self.window is None and self.render_mode == "human":
            cell_size = 30
            header_height = 100
            window_width = self.width * cell_size
            window_height = self.height * cell_size + header_height
            self.window = pygame.display.set_mode((window_width, window_height))
            pygame.display.set_caption(f"扫雷 Gymnasium - {self.core.difficulty}")
            self.clock = pygame.time.Clock()

        # 绘制逻辑（复用原始界面逻辑并适配Gymnasium）
        if self.window:
            self._draw_window()

        # rgb_array模式返回像素数据
        if self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(self.window)

    def _draw_window(self):
        """内部绘制函数（实现界面细节）"""
        cell_size = 30
        header_height = 100
        window_width = self.width * cell_size
        window_height = self.height * cell_size + header_height

        # 填充背景
        self.window.fill(WHITE)

        # 绘制头部信息栏
        self._draw_header(header_height, window_width)

        # 绘制网格
        self._draw_grid(cell_size, header_height)

        # 绘制游戏结果弹窗（如果结束）
        self._draw_result_popup(window_width, window_height)

        # 刷新显示
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _draw_header(self, header_height, window_width):
        """绘制头部信息（剩余地雷、时间、难度按钮）"""
        # 头部背景
        header_rect = pygame.Rect(0, 0, window_width, header_height)
        pygame.draw.rect(self.window, LIGHT_GRAY, header_rect)
        pygame.draw.line(self.window, DARK_GRAY, (0, header_height), (window_width, header_height), 2)
        pygame.draw.line(self.window, DARK_GRAY, (0, 60), (window_width, 60), 1)

        # 剩余地雷数
        flagged_count = sum(1 for row in self.core.cells for cell in row if cell.mark_type == 1)
        mine_count = self.core.mines_count - flagged_count
        mine_text = medium_font.render(f"剩余地雷: {max(0, mine_count)}", True, BLACK)
        self.window.blit(mine_text, (20, 20))

        # 步数显示（替代原有用时，更适合智能体训练）
        step_text = medium_font.render(f"步数: {self.core.t}", True, BLACK)
        self.window.blit(step_text, (window_width - 120, 20))

        # 难度按钮
        btn_width, btn_height = 70, 30
        btn_y = 65
        difficulties = ["简单", "中等", "困难"]
        btn_rects = [
            pygame.Rect(window_width//2 - btn_width*1.5 - 10, btn_y, btn_width, btn_height),
            pygame.Rect(window_width//2 - btn_width//2, btn_y, btn_width, btn_height),
            pygame.Rect(window_width//2 + btn_width*0.5 + 10, btn_y, btn_width, btn_height)
        ]

        # 绘制难度按钮
        for rect, diff in zip(btn_rects, difficulties):
            color = GREEN if self.core.difficulty == diff else BUTTON_BG
            pygame.draw.rect(self.window, color, rect, border_radius=5)
            pygame.draw.rect(self.window, DARK_GRAY, rect, 1, border_radius=5)
            text = small_font.render(diff, True, BLACK)
            self.window.blit(text, text.get_rect(center=rect.center))

    def _draw_grid(self, cell_size, header_height):
        """绘制游戏网格和格子状态"""
        # 网格背景
        grid_rect = pygame.Rect(0, header_height, self.width*cell_size, self.height*cell_size)
        pygame.draw.rect(self.window, GRAY, grid_rect)

        # 遍历所有格子绘制
        for x in range(self.width):
            for y in range(self.height):
                cell = self.core.cells[x][y]
                # 计算格子位置
                rect = pygame.Rect(
                    x*cell_size,
                    y*cell_size + header_height,
                    cell_size - 1,
                    cell_size - 1
                )

                if cell.is_revealed:
                    # 已翻开格子
                    pygame.draw.rect(self.window, WHITE, rect)
                    if cell.is_mine:
                        # 地雷（红色圆形）
                        pygame.draw.circle(
                            self.window, RED,
                            (x*cell_size + cell_size//2, y*cell_size + header_height + cell_size//2),
                            cell_size//3
                        )
                    elif cell.adjacent_mines > 0:
                        # 周围地雷数（不同数字不同颜色）
                        color_map = {1: BLUE, 2: GREEN, 3: RED, 4: (0,0,128), 5: (128,0,0), 6: (0,128,128), 7: BLACK, 8: GRAY}
                        text = small_font.render(str(cell.adjacent_mines), True, color_map[cell.adjacent_mines])
                        self.window.blit(text, text.get_rect(center=rect.center))
                else:
                    # 未翻开格子（立体效果）
                    pygame.draw.rect(self.window, LIGHT_GRAY, rect)
                    pygame.draw.line(self.window, WHITE, rect.topleft, rect.topright, 2)
                    pygame.draw.line(self.window, WHITE, rect.topleft, rect.bottomleft, 2)
                    pygame.draw.line(self.window, DARK_GRAY, rect.bottomright, rect.topright, 1)
                    pygame.draw.line(self.window, DARK_GRAY, rect.bottomright, rect.bottomleft, 1)

                    # 标记绘制
                    if cell.mark_type == 1:  # 地雷标记（旗帜）
                        flag_points = [
                            (x*cell_size + 5, y*cell_size + header_height + 5),
                            (x*cell_size + 5, y*cell_size + header_height + cell_size - 10),
                            (x*cell_size + cell_size - 10, y*cell_size + header_height + cell_size//2)
                        ]
                        pygame.draw.polygon(self.window, RED, flag_points)
                        pygame.draw.rect(self.window, YELLOW, (x*cell_size + 5, y*cell_size + header_height + 5, 3, cell_size - 5))
                    elif cell.mark_type == 2:  # 问号标记
                        text = small_font.render("?", True, PURPLE)
                        self.window.blit(text, text.get_rect(center=rect.center))

                # 智能体当前点击位置高亮
                if self.agent_current_click == (x, y):
                    pygame.draw.rect(self.window, (255, 255, 0), rect, 3)  # 黄色边框

                # 安全概率标记（绝对安全/绝对地雷）
                if self.core.mines_placed and not cell.is_revealed and cell.mark_type == 0:
                    if self.core.safety_prob_cache is not None:
                        if self.core.safety_prob_cache[x][y] >= 1.0:
                            pygame.draw.rect(self.window, (0, 255, 0), rect, 2)  # 绿色边框（绝对安全）
                        elif self.core.safety_prob_cache[x][y] <= 0.0:
                            pygame.draw.rect(self.window, (255, 0, 0), rect, 2)  # 红色边框（绝对地雷）

    def _draw_result_popup(self, window_width, window_height):
        """绘制游戏结果弹窗（胜利/失败）"""
        if not (self.core.game_over or self.core.victory):
            return

        # 半透明遮罩
        overlay = pygame.Surface((window_width, window_height), pygame.SRCALPHA)
        overlay.fill((255, 255, 255, 200))
        self.window.blit(overlay, (0, 0))

        # 弹窗背景
        popup_rect = pygame.Rect(0, 0, 300, 150)
        popup_rect.center = (window_width // 2, window_height // 2)
        pygame.draw.rect(self.window, WHITE, popup_rect, border_radius=10)
        pygame.draw.rect(self.window, DARK_GRAY, popup_rect, 2, border_radius=10)

        # 结果文本
        text = "游戏结束!" if self.core.game_over else "恭喜胜利!"
        color = RED if self.core.game_over else GREEN
        result_text = large_font.render(text, True, color)
        self.window.blit(result_text, result_text.get_rect(center=(window_width // 2, window_height // 2 - 30)))

        # 修正：移除对 reward_history 的引用，改用当前步骤的累积奖励计算
        # 统计信息（从 info 中获取剩余安全格子数和步数）
        info = self._get_info()
        stats_text = medium_font.render(
            f"步数: {info['step']} | 剩余安全格: {info['remaining_safe']}",
            True, BLACK
        )
        self.window.blit(stats_text, stats_text.get_rect(center=(window_width // 2, window_height // 2 + 10)))

    def close(self):
        """关闭环境（释放资源）"""
        if self.window is not None:
            pygame.quit()
            self.window = None

    def reset(self, seed=None, options=None):
        """重置环境（Gymnasium标准接口）"""
        super().reset(seed=seed)  # 调用父类reset以支持种子

        # 重置核心游戏状态
        self.core.reset()
        self.agent_current_click = None  # 重置高亮标记

        # 生成初始观测和信息
        initial_obs = self._get_obs()
        initial_info = self._get_info()

        # 渲染初始状态（如果需要）
        if self.render_mode in ["human", "rgb_array"]:
            self.render()

        return initial_obs, initial_info  # 必须返回(观测, 信息)元组

    # 在MinesweeperGym类中新增以下方法
    def player_step(self):
        """处理玩家输入并执行动作（替代智能体的step方法）"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False  # 退出游戏

            if event.type == pygame.MOUSEBUTTONDOWN and not (self.core.game_over or self.core.victory):
                x, y = pygame.mouse.get_pos()
                cell_size = 30
                header_height = 100

                # 处理难度按钮点击
                if y < header_height:
                    self._handle_difficulty_click(x, y, header_height)
                    return True

                # 计算格子坐标（转换鼠标位置到网格坐标）
                grid_x = x // cell_size
                grid_y = (y - header_height) // cell_size

                # 左键：翻开格子
                if event.button == 1:
                    if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                        action = grid_x * self.height + grid_y  # 转换为Gymnasium动作格式
                        _, _, terminated, _, _ = self.step(action)

                # 右键：标记地雷（切换标记类型）
                elif event.button == 3:
                    if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                        cell = self.core.cells[grid_x][grid_y]
                        if not cell.is_revealed:  # 未翻开的格子才能标记
                            cell.mark_type = (cell.mark_type + 1) % 3  # 0→1→2→0循环

            # 游戏结束后按R键重新开始
            if (self.core.game_over or self.core.victory) and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.reset()

        # 持续渲染界面
        self.render()
        return True

    def _handle_difficulty_click(self, x, y, header_height):
        """处理难度按钮点击事件"""
        btn_width, btn_height = 70, 30
        btn_y = 65
        window_width = self.width * 30

        # 三个难度按钮的区域
        difficulty_areas = {
            "简单": pygame.Rect(window_width // 2 - btn_width * 1.5 - 10, btn_y, btn_width, btn_height),
            "中等": pygame.Rect(window_width // 2 - btn_width // 2, btn_y, btn_width, btn_height),
            "困难": pygame.Rect(window_width // 2 + btn_width * 0.5 + 10, btn_y, btn_width, btn_height)
        }

        # 检查点击位置是否在某个难度按钮上
        for diff, rect in difficulty_areas.items():
            if rect.collidepoint(x, y) and diff != self.core.difficulty:
                self.core.set_difficulty(diff)
                self.width = self.core.width
                self.height = self.core.height
                # 更新动作空间（难度变化后格子数量变化）
                self.action_space = spaces.Discrete(self.width * self.height)
                self.observation_space = spaces.Box(
                    low=-10.0,
                    high=1.0,
                    shape=(3, self.height, self.width),
                    dtype=np.float32
                )
                self.reset()  # 重置游戏
                break


# 玩家模式运行入口
def play_game():
    """启动玩家可交互的扫雷游戏"""
    # 初始化玩家模式环境（强制human渲染）
    env = MinesweeperGym(difficulty="简单", render_mode="human")
    env.reset()  # 初始化游戏

    # 游戏主循环（处理玩家输入）
    running = True
    while running:
        running = env.player_step()  # 处理输入并更新界面

    env.close()  # 退出时释放资源


# 测试Gymnasium扫雷环境
if __name__ == "__main__":
    play_game()
    
    # # 1. 随机策略测试（可视化模式）
    # env = MinesweeperGym(difficulty="简单", render_mode="human")
    # observation, info = env.reset()
    #
    # # 记录奖励历史（用于结果显示）
    # reward_history = []
    #
    # terminated = False
    # while not terminated:
    #     # 随机选择动作（实际训练时替换为智能体决策）
    #     action = env.action_space.sample()
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     reward_history.append(reward)
    #
    #     # 打印关键信息
    #     print(
    #         f"步数: {info['step']} | 动作: {action} → ({action // env.height}, {action % env.height}) | 奖励: {reward:.1f}")
    #     time.sleep(1.)
    #
    # print(f"\n游戏结束！总奖励: {sum(reward_history):.1f} | 总步数: {info['step']}")
    # env.close()

    # 2. 无可视化模式测试（训练模式）
    # env = MinesweeperGym(difficulty="中等", render_mode="none")
    # for episode in range(3):
    #     obs, info = env.reset()
    #     done = False
    #     total_reward = 0
    #     while not done:
    #         action = env.action_space.sample()
    #         obs, reward, done, _, _ = env.step(action)
    #         total_reward += reward
    #     print(f"回合 {episode+1} | 总奖励: {total_reward:.1f}")
    # env.close()
