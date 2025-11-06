import pygame
import random
import sys
import time
import numpy as np
import torch

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
RESTART_BTN_NORMAL = (0, 160, 230)  # 重新开始按钮默认色
RESTART_BTN_HOVER = (0, 190, 255)  # 重新开始按钮悬停色

# 游戏难度设置
DIFFICULTIES = {
    "简单": {"width": 10, "height": 10, "mines": 10},
    "中等": {"width": 16, "height": 16, "mines": 40},
    "困难": {"width": 30, "height": 16, "mines": 99}
}

# 设置字体 - 优先使用支持中文的字体
pygame.font.init()


# 尝试加载中文字体，若失败则使用默认字体
def get_chinese_font(size):
    chinese_fonts = ["SimHei", "Microsoft YaHei", "Heiti TC", "WenQuanYi Micro Hei", "SimSun"]
    for font_name in chinese_fonts:
        try:
            return pygame.font.SysFont(font_name, size)
        except:
            continue
    return pygame.font.SysFont(None, size)


# 初始化不同大小的字体
small_font = get_chinese_font(16)
medium_font = get_chinese_font(20)
large_font = get_chinese_font(32)


class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.is_mine = False
        self.is_revealed = False
        self.mark_type = 0  # 0: 无标记, 1: 地雷标记, 2: 问号标记
        self.adjacent_mines = 0


class Minesweeper:
    def __init__(self, difficulty="简单", window=True):
        # 模式控制参数
        self.mark_accuracy = 0.5
        self.window = window  # True: 玩家模式（带界面），False: 智能体模式（无界面）

        # 游戏基础设置
        self.difficulty = difficulty
        self.settings = DIFFICULTIES[self.difficulty]
        self.cell_size = 30
        self.header_height = 100 if self.window else 0  # 智能体模式无头部信息栏

        # 智能体相关变量
        self.r = 0.0  # 单步奖励
        self.R = []  # 奖励序列
        self.actions = []  # 动作序列
        self.condition = True  # 游戏是否继续，False表示游戏终止
        self.t = 0  # 步数计数
        self.count = np.zeros([self.settings["width"], self.settings["height"]])  # 格子点击次数记录
        self.map = np.zeros([self.settings["width"], self.settings["height"]])  # 格子状态映射（供智能体观察）
        self.agent_current_click = None  # 记录智能体当前点击位置（用于高亮）

        # 初始化窗口（仅玩家模式）
        if self.window:
            self.window_width = self.settings["width"] * self.cell_size
            self.window_height = self.settings["height"] * self.cell_size + self.header_height
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("扫雷")
            self.clock = pygame.time.Clock()

        # 游戏状态变量
        self.cells = []
        self.mines_placed = False
        self.game_over = False
        self.victory = False
        self.start_time = None
        self.elapsed_time = 0
        self.hovered_button = None
        self.hovered_restart_btn = False
        self.safety_prob_cache = None
        self.stage_reward = {  # 初始化进度奖励记录字典
            "20%": False,
            "40%": False,
            "60%": False,
            "80%": False
        }

        # 初始化游戏
        self.reset()

    def reset(self):
        """重置游戏状态（支持玩家和智能体模式）"""
        # 初始化格子
        self.cells = [
            [Cell(x, y) for y in range(self.settings["height"])]
            for x in range(self.settings["width"])
        ]

        # 重置核心游戏状态
        self.mines_placed = False
        self.game_over = False
        self.victory = False
        self.start_time = None
        self.elapsed_time = 0
        self.safety_prob_cache = None
        self.stage_reward = {  # 重置进度奖励记录
            "20%": False,
            "40%": False,
            "60%": False,
            "80%": False
        }

        # 重置智能体相关变量
        self.r = 0.0
        self.R = []
        self.actions = []
        self.condition = True
        self.t = 0
        self.count = np.zeros([self.settings["width"], self.settings["height"]])
        self.map = np.zeros([self.settings["width"], self.settings["height"]])
        self.agent_current_click = None

        # 重置玩家模式UI状态
        if self.window:
            self.hovered_button = None
            self.hovered_restart_btn = False

    def set_difficulty(self, difficulty):
        """更改游戏难度（仅玩家模式有效）"""
        if difficulty in DIFFICULTIES and difficulty != self.difficulty:
            self.difficulty = difficulty
            self.settings = DIFFICULTIES[difficulty]

            # 更新智能体相关数组尺寸
            self.count = np.zeros([self.settings["width"], self.settings["height"]])
            self.map = np.zeros([self.settings["width"], self.settings["height"]])

            # 玩家模式更新窗口
            if self.window:
                self.window_width = self.settings["width"] * self.cell_size
                self.window_height = self.settings["height"] * self.cell_size + self.header_height
                self.screen = pygame.display.set_mode((self.window_width, self.window_height))

            self.reset()

    def place_mines(self, first_click_x, first_click_y):
        """在第一次点击后放置地雷"""
        mines_placed = 0
        safe_zone = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = first_click_x + dx, first_click_y + dy
                if 0 <= nx < self.settings["width"] and 0 <= ny < self.settings["height"]:
                    safe_zone.append((nx, ny))

        while mines_placed < self.settings["mines"]:
            x = random.randint(0, self.settings["width"] - 1)
            y = random.randint(0, self.settings["height"] - 1)
            if (x, y) not in safe_zone and not self.cells[x][y].is_mine:
                self.cells[x][y].is_mine = True
                mines_placed += 1

        self.calculate_adjacent_mines()
        self.mines_placed = True
        if self.window:
            self.start_time = time.time()

    def calculate_adjacent_mines(self):
        """计算每个格子周围的地雷数量"""
        for x in range(self.settings["width"]):
            for y in range(self.settings["height"]):
                if not self.cells[x][y].is_mine:
                    count = 0
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.settings["width"] and 0 <= ny < self.settings["height"]:
                                if self.cells[nx][ny].is_mine:
                                    count += 1
                    self.cells[x][y].adjacent_mines = count

    def calculate_safety_prob(self):
        width = self.settings["width"]
        height = self.settings["height"]
        safety_prob = np.ones((width, height), dtype=np.float64)
        revealed = np.array([[cell.is_revealed for cell in row] for row in self.cells], dtype=bool)
        marked = np.array([[cell.mark_type == 1 for cell in row] for row in self.cells], dtype=bool)

        if not hasattr(self, 'mark_accuracy'):
            self.mark_accuracy = 0.5
        self.mark_accuracy = max(0.05, min(0.95, self.mark_accuracy))

        # 多轮迭代推理
        for _ in range(5):
            updated = False
            for x in range(width):
                for y in range(height):
                    cell = self.cells[x][y]
                    if not cell.is_revealed or cell.adjacent_mines == 0:
                        continue

                    unknown_count = 0
                    weighted_marked_count = 0.0
                    unknown_positions = []
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            # 1. 严格边界检查：确保nx和ny在有效范围内
                            if 0 <= nx < width and 0 <= ny < height:
                                if not revealed[nx, ny]:
                                    if marked[nx, ny]:
                                        weighted_marked_count += self.mark_accuracy
                                    else:
                                        unknown_count += 1
                                        unknown_positions.append((nx, ny))

                    remaining_mines = cell.adjacent_mines - weighted_marked_count
                    remaining_mines = max(0.0, min(unknown_count, remaining_mines))

                    if unknown_count > 0 and remaining_mines <= 1e-6:
                        for (nx, ny) in unknown_positions:
                            # 2. 再次检查（双重保险）
                            if 0 <= nx < width and 0 <= ny < height and safety_prob[nx, ny] != 1.0:
                                safety_prob[nx, ny] = 1.0
                                updated = True
                    elif unknown_count > 0 and remaining_mines >= unknown_count - 1e-6:
                        for (nx, ny) in unknown_positions:
                            if 0 <= nx < width and 0 <= ny < height and safety_prob[nx, ny] != 0.0:
                                safety_prob[nx, ny] = 0.0
                                updated = True
            if not updated:
                break

        # 标记有效性反向验证
        for x in range(width):
            for y in range(height):
                if marked[x, y] and not revealed[x, y]:
                    invalid = False
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            # 3. 边界检查
                            if 0 <= nx < width and 0 <= ny < height and self.cells[nx][ny].is_revealed:
                                cell = self.cells[nx][ny]
                                if cell.adjacent_mines == 0:
                                    continue
                                mark_count = 0
                                for ddx in [-1, 0, 1]:
                                    for ddy in [-1, 0, 1]:
                                        nnx, nny = nx + ddx, ny + ddy
                                        # 4. 嵌套循环中的边界检查
                                        if 0 <= nnx < width and 0 <= nny < height and marked[nnx, nny]:
                                            mark_count += 1
                                if mark_count > cell.adjacent_mines:
                                    invalid = True
                                    break
                        if invalid:
                            break
                    if invalid:
                        safety_prob[x, y] = 1.0
                        self.mark_accuracy = max(0.05, self.mark_accuracy - 0.15)

        # 跨格子关联验证
        prob_contrib = np.zeros((width, height))
        contrib_count = np.zeros((width, height), dtype=int)
        for x in range(width):
            for y in range(height):
                cell = self.cells[x][y]
                if not cell.is_revealed or cell.adjacent_mines == 0:
                    continue
                unknown_count = 0
                weighted_marked_count = 0.0
                unknown_positions = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        # 5. 边界检查
                        if 0 <= nx < width and 0 <= ny < height and not revealed[nx, ny]:
                            if marked[nx, ny]:
                                weighted_marked_count += self.mark_accuracy
                            else:
                                unknown_count += 1
                                unknown_positions.append((nx, ny))
                remaining_mines = max(0.0, cell.adjacent_mines - weighted_marked_count)
                if unknown_count == 0 or remaining_mines <= 1e-6:
                    continue
                mine_prob = remaining_mines / unknown_count
                for (nx, ny) in unknown_positions:
                    # 6. 再次检查
                    if 0 <= nx < width and 0 <= ny < height:
                        prob_contrib[nx, ny] += mine_prob
                        contrib_count[nx, ny] += 1

        # ---------------------- 孤立格子处理（沿用之前的优化） ----------------------
        total_safe = width * height - self.settings["mines"]
        revealed_safe = sum(1 for row in self.cells for cell in row if cell.is_revealed and not cell.is_mine)
        remaining_safe = total_safe - revealed_safe
        remaining_unknown = sum(
            1 for x in range(width) for y in range(height) if not revealed[x, y] and not marked[x, y])
        total_marked = np.sum(marked)
        estimated_valid_marks = total_marked * self.mark_accuracy
        remaining_mines_estimated = max(0.0, self.settings["mines"] - estimated_valid_marks)

        if remaining_unknown > 0:
            isolated_safe = remaining_unknown - remaining_mines_estimated
            isolated_prob = max(0.0, min(1.0, isolated_safe / remaining_unknown))
            for x in range(width):
                for y in range(height):
                    if not revealed[x, y] and not marked[x, y] and safety_prob[x, y] == 1.0:
                        safety_prob[x, y] = isolated_prob

        # 无效区域标记
        for x in range(width):
            for y in range(height):
                if revealed[x, y] or marked[x, y]:
                    safety_prob[x, y] = -1.0

        self.safety_prob_cache = safety_prob
        return safety_prob

    def check_unlock_safe_cells(self, x, y):
        """检查标记后是否解锁绝对安全格子，用于奖励计算"""

        # 辅助函数：检查坐标是否在有效范围内
        def is_valid(nx, ny):
            return 0 <= nx < self.settings["width"] and 0 <= ny < self.settings["height"]

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if not is_valid(nx, ny):
                    continue  # 直接跳过无效坐标，减少嵌套

                cell = self.cells[nx][ny]
                if not (cell.is_revealed and cell.adjacent_mines > 0):
                    continue  # 跳过不满足条件的单元格，减少嵌套

                # 计算周围标记数和未知格子数（拆分多行提高可读性）
                marked_count = sum(
                    1 for ddx in [-1, 0, 1]
                    for ddy in [-1, 0, 1]
                    if is_valid(nx + ddx, ny + ddy)
                    and self.cells[nx + ddx][ny + ddy].mark_type == 1
                )

                unknown_count = sum(
                    1 for ddx in [-1, 0, 1]
                    for ddy in [-1, 0, 1]
                    if is_valid(nx + ddx, ny + ddy)
                    and not self.cells[nx + ddx][ny + ddy].is_revealed
                    and self.cells[nx + ddx][ny + ddy].mark_type == 0
                )

                if cell.adjacent_mines - marked_count == 0 and unknown_count > 0:
                    self.r += 0.3  # 解锁安全格子的额外奖励

    def reveal_cell(self, x, y, is_agent=False):
        """翻开格子（兼容智能体模式，不绘制界面）"""
        if not 0 <= x < self.settings["width"] or not 0 <= y < self.settings["height"]:
            return

        cell = self.cells[x][y]
        if cell.is_revealed or cell.mark_type != 0:  # 已标记的格子不能翻开
            return

        cell.is_revealed = True

        # 更新智能体用map
        if cell.is_mine:
            self.map[x, y] = -10  # 地雷标记
            self.game_over = True
            self.condition = False
            for rx in range(self.settings["width"]):
                for ry in range(self.settings["height"]):
                    if not self.cells[rx][ry].is_revealed:
                        self.cells[rx][ry].is_revealed = True

            # 玩家模式：显示所有格子后延迟重启
            if not is_agent and self.window:
                self.draw_grid()  # 刷新界面显示所有地雷
                pygame.display.flip()
                time.sleep(2)  # 停留2秒让玩家观察
            return
        else:
            self.map[x, y] = cell.adjacent_mines  # 周围地雷数
            if not is_agent:
                self.r = 1.0  # 玩家模式安全翻开奖励（智能体奖励在update中计算）

        # 空白格子递归翻开
        if cell.adjacent_mines == 0:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    self.reveal_cell(x + dx, y + dy, is_agent)

            # 新增：验证周围标记的正确性，更新标记置信度
            if not cell.is_mine and cell.adjacent_mines > 0 and self.mines_placed:
                # 统计周围实际地雷数
                actual_mines = sum(
                    1 for dx in [-1, 0, 1]
                    for dy in [-1, 0, 1]
                    if 0 <= x + dx < self.settings["width"]
                    and 0 <= y + dy < self.settings["height"]
                    and self.cells[x + dx][y + dy].is_mine
                )
                # 统计周围标记数
                marked_count = sum(
                    1 for dx in [-1, 0, 1]
                    for dy in [-1, 0, 1]
                    if 0 <= x + dx < self.settings["width"]
                    and 0 <= y + dy < self.settings["height"]
                    and self.cells[x + dx][y + dy].mark_type == 1
                )

                # 验证标记是否正确（标记数 == 实际地雷数）
                if marked_count == actual_mines:
                    # 标记正确：提升置信度
                    self.mark_accuracy = min(0.95, self.mark_accuracy + 0.05)
                else:
                    # 标记错误：降低置信度
                    self.mark_accuracy = max(0.05, self.mark_accuracy - 0.05)


    def toggle_mark(self, x, y):
        """切换格子标记（仅玩家模式使用）"""
        if not self.window or self.game_over or self.victory:
            return
        if not 0 <= x < self.settings["width"] or not 0 <= y < self.settings["height"]:
            return
        cell = self.cells[x][y]
        if cell.is_revealed:
            return
        cell.mark_type = (cell.mark_type + 1) % 3

        # 同步更新self.map
        if cell.mark_type == 1:  # 标记为地雷，对应self.map=-1
            self.map[x, y] = -1
        elif cell.mark_type == 2:  # 标记为问号，可设为一个不冲突值，如10
            self.map[x, y] = 10
        else:  # 取消标记，恢复为0
            self.map[x, y] = 0

    def check_victory(self):
        """检查胜利（同步更新condition）"""
        if self.game_over:
            return False

        for x in range(self.settings["width"]):
            for y in range(self.settings["height"]):
                cell = self.cells[x][y]
                if not cell.is_mine and not cell.is_revealed:
                    return False
        self.victory = True
        self.condition = False
        return True

    def count_remaining_mines(self):
        """计算剩余地雷数（仅玩家模式使用）"""
        if not self.window:
            return 0
        flagged_count = sum(1 for x in range(self.settings["width"])
                            for y in range(self.settings["height"])
                            if self.cells[x][y].mark_type == 1)
        return self.settings["mines"] - flagged_count

    def update_time(self):
        """更新游戏用时（仅玩家模式使用）"""
        if self.window and self.start_time and not self.game_over and not self.victory:
            self.elapsed_time = int(time.time() - self.start_time)

    # ---------------------- 玩家模式UI绘制相关方法 ----------------------
    def draw_header(self):
        if not self.window:
            return None, None, None

        header_rect = pygame.Rect(0, 0, self.window_width, self.header_height)
        pygame.draw.rect(self.screen, LIGHT_GRAY, header_rect)
        pygame.draw.line(self.screen, DARK_GRAY, (0, self.header_height), (self.window_width, self.header_height), 2)
        pygame.draw.line(self.screen, DARK_GRAY, (0, 60), (self.window_width, 60), 1)

        # 剩余地雷数
        mine_count = self.count_remaining_mines()
        mine_text = medium_font.render(f"剩余地雷: {max(0, mine_count)}", True, BLACK)
        mine_rect = mine_text.get_rect(topleft=(20, 20))
        self.screen.blit(mine_text, mine_rect)

        # 用时
        time_text = medium_font.render(f"用时: {self.elapsed_time}s", True, BLACK)
        time_rect = time_text.get_rect(topright=(self.window_width - 20, 20))
        self.screen.blit(time_text, time_rect)

        # 难度按钮
        BUTTON_WIDTH = 70
        BUTTON_HEIGHT = 30
        BUTTON_Y = 65
        easy_rect = pygame.Rect(self.window_width // 2 - BUTTON_WIDTH * 1.5 - 10, BUTTON_Y, BUTTON_WIDTH, BUTTON_HEIGHT)
        medium_rect = pygame.Rect(self.window_width // 2 - BUTTON_WIDTH // 2, BUTTON_Y, BUTTON_WIDTH, BUTTON_HEIGHT)
        hard_rect = pygame.Rect(self.window_width // 2 + BUTTON_WIDTH * 0.5 + 10, BUTTON_Y, BUTTON_WIDTH, BUTTON_HEIGHT)

        # 检查悬停
        mouse_pos = pygame.mouse.get_pos()
        self.hovered_button = None
        if easy_rect.collidepoint(mouse_pos):
            self.hovered_button = "简单"
        elif medium_rect.collidepoint(mouse_pos):
            self.hovered_button = "中等"
        elif hard_rect.collidepoint(mouse_pos):
            self.hovered_button = "困难"

        # 绘制按钮
        difficulty_buttons = [(easy_rect, "简单"), (medium_rect, "中等"), (hard_rect, "困难")]
        for rect, text in difficulty_buttons:
            if self.difficulty == text:
                color = GREEN
            elif self.hovered_button == text:
                color = BUTTON_HOVER
            else:
                color = BUTTON_BG
            pygame.draw.rect(self.screen, color, rect, border_radius=5)
            pygame.draw.rect(self.screen, DARK_GRAY, rect, 1, border_radius=5)
            button_text = small_font.render(text, True, BLACK)
            text_rect = button_text.get_rect(center=rect.center)
            self.screen.blit(button_text, text_rect)

        return easy_rect, medium_rect, hard_rect

    def draw_grid(self):
        """绘制游戏网格（仅玩家模式）"""
        if not self.window:
            return

        # 网格背景
        grid_rect = pygame.Rect(0, self.header_height,
                                self.window_width, self.window_height - self.header_height)
        pygame.draw.rect(self.screen, GRAY, grid_rect)

        # 遍历所有格子绘制
        for x in range(self.settings["width"]):
            for y in range(self.settings["height"]):
                cell = self.cells[x][y]
                # 计算格子在屏幕上的位置（偏移头部高度）
                cell_rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size + self.header_height,
                    self.cell_size - 1,  # 减1留出网格线间隙
                    self.cell_size - 1
                )

                if cell.is_revealed:
                    # 已翻开的格子且不为智能体当前点击格子：白色背景
                    pygame.draw.rect(self.screen, WHITE, cell_rect)

                    if cell.is_mine:
                        # 地雷：红色圆形
                        pygame.draw.circle(self.screen, RED,
                                           (x * self.cell_size + self.cell_size // 2,
                                            y * self.cell_size + self.header_height + self.cell_size // 2),
                                           self.cell_size // 3)
                    elif cell.adjacent_mines > 0:
                        # 周围地雷数：不同数字不同颜色
                        color_map = {
                            1: BLUE,
                            2: GREEN,
                            3: RED,
                            4: (0, 0, 128),  # 深蓝色
                            5: (128, 0, 0),  # 棕色
                            6: (0, 128, 128),  # 青色
                            7: BLACK,
                            8: GRAY
                        }
                        color = color_map.get(cell.adjacent_mines, BLACK)
                        num_text = small_font.render(str(cell.adjacent_mines), True, color)
                        text_rect = num_text.get_rect(center=cell_rect.center)
                        self.screen.blit(num_text, text_rect)
                else:
                    # 未翻开的格子：立体效果（浅灰背景+边框高光）
                    pygame.draw.rect(self.screen, LIGHT_GRAY, cell_rect)
                    pygame.draw.line(self.screen, WHITE, cell_rect.topleft, cell_rect.topright, 2)
                    pygame.draw.line(self.screen, WHITE, cell_rect.topleft, cell_rect.bottomleft, 2)
                    pygame.draw.line(self.screen, DARK_GRAY, cell_rect.bottomright, cell_rect.topright, 1)
                    pygame.draw.line(self.screen, DARK_GRAY, cell_rect.bottomright, cell_rect.bottomleft, 1)

                    # 标记绘制
                    if cell.mark_type == 1:
                        # 地雷标记（旗帜）
                        flag_points = [
                            (x * self.cell_size + 5, y * self.cell_size + self.header_height + 5),
                            (x * self.cell_size + 5, y * self.cell_size + self.header_height + self.cell_size - 10),
                            (x * self.cell_size + self.cell_size - 10,
                             y * self.cell_size + self.header_height + self.cell_size // 2)
                        ]
                        pygame.draw.polygon(self.screen, RED, flag_points)
                        pygame.draw.rect(self.screen, YELLOW,
                                         (x * self.cell_size + 5,
                                          y * self.cell_size + self.header_height + 5,
                                          3, self.cell_size - 5))
                    elif cell.mark_type == 2:
                        # 问号标记
                        question_text = small_font.render("?", True, PURPLE)
                        text_rect = question_text.get_rect(center=cell_rect.center)
                        self.screen.blit(question_text, text_rect)

                # 智能体当前点击格子高亮（仅窗口+智能体模式）
                highlight = self.agent_current_click and (x, y) == self.agent_current_click
                if highlight:
                    # 黄色边框
                    pygame.draw.rect(self.screen, (255, 255, 0), cell_rect, 5)  # 3像素黄色边框，突出位置

                if self.mines_placed and not cell.is_revealed and cell.mark_type == 0:
                    safety_prob = self.calculate_safety_prob()[x][y]
                    # 标记绝对安全格子（安全概率=1，未翻开且未标记）
                    if self.safety_prob_cache[x][y] >= 1.0:
                        # 绿色细边框标记绝对安全格子
                        pygame.draw.rect(self.screen, (0, 255, 0), cell_rect, 2)
                    elif self.safety_prob_cache[x][y] <= 0.0:
                        # 红色细边框标记绝对雷区格子或已翻开格子
                        pygame.draw.rect(self.screen, (255, 0, 0), cell_rect, 2)

    def draw_game_result(self):
        """绘制游戏结果弹窗（胜利/失败 + 重新开始按钮，仅玩家模式）"""
        if not self.window or (not self.game_over and not self.victory):
            return None

        # 半透明遮罩
        overlay = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
        overlay.fill((255, 255, 255, 200))  # 透明度200/255
        self.screen.blit(overlay, (0, 0))

        # 结果弹窗背景
        popup_rect = pygame.Rect(0, 0, 300, 150)
        popup_rect.center = (self.window_width // 2, self.window_height // 2)
        pygame.draw.rect(self.screen, WHITE, popup_rect, border_radius=10)
        pygame.draw.rect(self.screen, DARK_GRAY, popup_rect, 2, border_radius=10)

        # 结果文本
        result_text = large_font.render(
            "游戏结束!", True, RED
        ) if self.game_over else large_font.render(
            "恭喜胜利!", True, GREEN
        )
        text_rect = result_text.get_rect(center=(self.window_width // 2, self.window_height // 2 - 30))
        self.screen.blit(result_text, text_rect)

        # 重新开始按钮
        restart_btn = pygame.Rect(0, 0, 150, 40)
        restart_btn.center = (self.window_width // 2, self.window_height // 2 + 20)

        # 按钮悬停检测
        mouse_pos = pygame.mouse.get_pos()
        self.hovered_restart_btn = restart_btn.collidepoint(mouse_pos)

        # 绘制按钮
        btn_color = RESTART_BTN_HOVER if self.hovered_restart_btn else RESTART_BTN_NORMAL
        pygame.draw.rect(self.screen, btn_color, restart_btn, border_radius=5)
        pygame.draw.rect(self.screen, DARK_GRAY, restart_btn, 2, border_radius=5)

        restart_text = medium_font.render("重新开始", True, WHITE)
        text_rect = restart_text.get_rect(center=restart_btn.center)
        self.screen.blit(restart_text, text_rect)

        return restart_btn

    def draw(self):
        """绘制完整游戏界面（仅玩家模式）"""
        if not self.window:
            return None, None, None, None

        self.screen.fill(WHITE)
        # 绘制头部信息栏并获取按钮区域
        easy_rect, medium_rect, hard_rect = self.draw_header()
        # 绘制网格
        self.draw_grid()
        # 绘制结果弹窗（如果游戏结束）
        restart_btn = self.draw_game_result() if (self.game_over or self.victory) else None

        pygame.display.flip()
        return easy_rect, medium_rect, hard_rect, restart_btn

    # ---------------------- 智能体交互核心方法 ----------------------
    def get_status(self):
        """获取智能体观察的环境状态（三维数组：[3, H, W], 对应着3层：[状态层, 点击计数层, 安全概率层]）"""
        # 状态层：地雷标记=-2，未翻开=-1，已翻开地雷=-10，已翻开安全区=周围地雷数
        status_layer = (np.array([[cell.is_revealed for cell in row]
                                  for row in self.cells], dtype=np.float64) - 1) + self.map
        # 计算安全概率层
        safety_layer = self.calculate_safety_prob()
        # 堆叠状态层,点击计数层,安全概率层（便于智能体观察历史行为）
        status = np.stack((status_layer, self.count, safety_layer), axis=0)
        return status

    def agent_step(self, action):
        """智能体执行一步动作（点击坐标(x,y)），优化奖励机制：鼓励接近胜利的阶段性成果"""
        action_type, x, y = action
        self.r = 0.0  # 重置单步奖励
        total_safe = self.settings["width"] * self.settings["height"] - self.settings["mines"]  # 总安全格子数

        # 获取当前格子的安全概率（用于奖励计算）
        safety_prob = self.calculate_safety_prob()[x][y] if self.mines_placed else 1.0  # 未放地雷时默认安全

        # 过滤绝对雷区动作（安全概率≤0.0）
        if self.mines_placed:
            if safety_prob <= 0.0:
                self.r = -1.0
                self.t += 1
                self.R.append(self.r)
                self.actions.append((x, y))
                return self._agent_return()

        if self.window:
            # 记录当前点击位置（用于高亮）
            self.agent_current_click = (x, y)

            self.clock.tick(30)  # 30FPS
            self.update_time()
            self.draw()

        # 1. 边界校验惩罚（无效动作）
        if not (0 <= x < self.settings["width"] and 0 <= y < self.settings["height"]):
            self.r = -2.0
            self.condition = False
            return self._agent_return()

        cell = self.cells[x][y]

        # 2. 已翻开/已标记格子点击惩罚（避免无效探索）
        if cell.is_revealed or cell.mark_type != 0:
            self.r = -1.0
            self.t += 1
            self.count[x][y] += 1
            self.R.append(self.r)
            self.actions.append((x, y))

            return self._agent_return()

        # 3. 首次点击生成地雷（保证首步安全）
        if not self.mines_placed:
            action_type = 0  # 首次点击的动作一定是揭开格子
            self.place_mines(x, y)
            safety_prob = 1.0  # 首步点击的格子绝对安全

        if action_type == 1:
            if cell.mark_type == 1:
                self.r = -0.8  # 重复标记：中等惩罚
            else:
                cell.mark_type = 1  # 执行标记
                self.map[x, y] = -1
                self.r = 0.5  # 基础标记奖励
                # 额外奖励：若标记后解锁安全格子（如周围数字剩余地雷=0）
                if self.mines_placed:
                    self.check_unlock_safe_cells(x, y)  # 新增辅助函数
        elif action_type == 2:
            if cell.mark_type != 1:
                self.r = -1.0  # 取消非标记格子：重惩罚
            else:
                cell.mark_type = 0  # 取消标记
                self.map[x, y] = 0
                self.r = 0.2  # 合理取消：轻微奖励
        elif action_type == 0:
            if cell.mark_type == 1:
                self.r = -2.0  # 翻开已标记格子：重惩罚

        if action_type == 0 and cell.mark_type == 0:
            # 4. 执行翻开操作（智能体模式）
            self.reveal_cell(x, y, is_agent=True)

            # 5. 计算当前已翻开的安全格子数（排除地雷）
            revealed_safe = sum(
                1 for row in self.cells
                for c in row
                if not c.is_mine and c.is_revealed  # 只统计安全且已翻开的格子
            )

            if total_safe == 0:  # 避免除以0（极端情况处理）
                progress = 0.0
            else:
                progress = revealed_safe / total_safe  # 计算当前进度比例

            # 6. 核心奖励逻辑：分阶段奖励（完全胜利 > 接近胜利 > 安全探索）
            if not self.game_over:  # 未踩雷时才给正向奖励
                # 6.1 安全翻开新格子的基础奖励
                self.r = 0.8  # 基础探索奖励

                # 6.2 信息价值奖励（高数字格子提供更多线索）
                self.r += cell.adjacent_mines * 0.3

                # 6.3 安全概率奖励（核心新增：鼓励选择高安全概率格子）
                if safety_prob >= 1.0:  # 绝对安全格子（优先选择）
                    self.r += 1.5  # 额外高额奖励
                elif safety_prob >= 0.8:  # 高安全概率
                    self.r += 0.8
                elif safety_prob >= 0.5:  # 中等安全概率
                    self.r += 0.3
                else:  # 低安全概率（冒险行为）
                    self.r -= 2.0  # 惩罚冒险

                # 6.4 低安全概率选择追加惩罚（若存在安全概率≥0.8的格子却选<0.5的）
                safe_cells = sum(
                    1 for x in range(self.settings["width"]) for y in range(self.settings["height"]) if
                    self.calculate_safety_prob()[x][y] >= 0.8)
                if safe_cells > 0 and safety_prob < 0.5:
                    self.r -= 5.0  # 有高安全格子却冒险，追加惩罚

                # 阶段性奖励动态阈值（按难度调整）
                if self.difficulty == "简单":
                    threshold = max(1, int(total_safe * 0.2))  # 简单难度阈值20%，更容易触发
                elif self.difficulty == "中等":
                    threshold = max(1, int(total_safe * 0.15))
                else:
                    threshold = max(1, int(total_safe * 0.1))

                # 一次性阶段性奖励
                if not self.stage_reward and (total_safe - threshold) <= revealed_safe < total_safe:
                    self.r += 20.0  # 接近胜利的额外奖励（比完全胜利低，留提升空间）
                    self.stage_reward = True

                # 进度奖励：每达到20%进度且未奖励过，则追加奖励
                if progress >= 0.2 and not self.stage_reward["20%"]:
                    self.r += 5.0  # 20%进度奖励
                    self.stage_reward["20%"] = True  # 标记为已奖励
                if progress >= 0.4 and not self.stage_reward["40%"]:
                    self.r += 5.0  # 40%进度奖励
                    self.stage_reward["40%"] = True
                if progress >= 0.6 and not self.stage_reward["60%"]:
                    self.r += 5.0  # 60%进度奖励
                    self.stage_reward["60%"] = True
                if progress >= 0.8 and not self.stage_reward["80%"]:
                    self.r += 5.0  # 80%进度奖励
                    self.stage_reward["80%"] = True

            # 7. 终局状态处理（完全胜利/踩雷失败）
            if self.game_over:  # 踩雷失败：重惩罚
                print("踩雷了，游戏失败！")
                self.r = -50.0
                self.condition = False
                # 智能体窗口模式下，延时1秒重启
                if self.window:
                    self.draw_grid()
                    pygame.display.flip()  # 刷新界面显示所有地雷
                    self.agent_current_click = None  # 重置高亮标记
                    time.sleep(1.)
            elif revealed_safe == total_safe:  # 完全胜利：最高奖励
                print("扫雷完成，游戏胜利！")
                self.r = 100.0 + (self.settings["width"] * self.settings["height"] - self.t) * 0.05  # 快速胜利加成
                correct_marks = sum(1 for row in self.cells for c in row if c.mark_type == 1 and c.is_mine)
                wrong_marks = sum(1 for row in self.cells for c in row if c.mark_type == 1 and not c.is_mine)
                self.r = 100.0 + (self.settings["width"] * self.settings["height"] - self.t) * 0.1
                self.r += correct_marks * 6.0  # 正确标记奖励
                self.r -= wrong_marks * 4.0  # 错误标记惩罚
                self.condition = False

            # 8. 重复点击惩罚（累积惩罚）
            if self.count[x][y] > 0:
                self.r -= 0.3 * self.count[x][y]  # 重复次数越多，惩罚越重（如第2次-0.6，第3次-0.9）

        # 9. 步数限制（防止无限循环，适配难度）
        self.t += 1
        self.count[x][y] += 1  # 记录点击次数
        max_steps = max(100, total_safe * 1)  # 最大步数=安全格子数×2（保证足够探索时间）
        if self.t >= max_steps:
            self.r = -5.0
            self.condition = False

        # 记录奖励和动作
        self.R.append(self.r)
        self.actions.append((x, y))

        return self._agent_return()

    def _agent_return(self):
        """智能体返回格式封装（状态+奖励+是否结束）"""
        return (
            torch.tensor(self.get_status(), dtype=torch.float32),
            self.r,
            not self.condition
        )

    # ---------------------- 主循环 ----------------------
    def run(self):
        """玩家模式主循环"""
        if not self.window:
            print("请使用window=True启用玩家模式")
            return

        running = True
        while running:
            self.clock.tick(30)  # 30FPS
            self.update_time()

            # 绘制界面并获取按钮区域
            easy_rect, medium_rect, hard_rect, restart_btn = self.draw()

            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()

                    # 游戏结束状态：处理重新开始按钮
                    if (self.game_over or self.victory) and restart_btn and restart_btn.collidepoint(x, y):
                        self.reset()
                        continue

                    # 游戏中：处理难度按钮
                    if not self.game_over and not self.victory and y < self.header_height:
                        if easy_rect and easy_rect.collidepoint(x, y):
                            self.set_difficulty("简单")
                        elif medium_rect and medium_rect.collidepoint(x, y):
                            self.set_difficulty("中等")
                        elif hard_rect and hard_rect.collidepoint(x, y):
                            self.set_difficulty("困难")
                        continue

                    # 游戏中：处理格子点击
                    if not self.game_over and not self.victory:
                        grid_x = x // self.cell_size
                        grid_y = (y - self.header_height) // self.cell_size

                        if event.button == 1:  # 左键：翻开格子
                            if not self.mines_placed:
                                self.place_mines(grid_x, grid_y)
                            self.reveal_cell(grid_x, grid_y)
                            self.check_victory()  # 检查是否胜利
                        elif event.button == 3:  # 右键：标记格子
                            self.toggle_mark(grid_x, grid_y)

        pygame.quit()
        sys.exit()


# 运行示例
if __name__ == "__main__":
    # 玩家模式（默认）
    game = Minesweeper(difficulty="简单", window=True)
    game.run()

    # 智能体模式示例（可用于强化学习训练）
    # agent_game = Minesweeper(difficulty="简单", window=False)
    # print("智能体模式初始化完成")
    # print("初始状态 shape:", agent_game.get_status().shape)
    #
    # # 模拟智能体随机探索
    # step = 0
    # while agent_game.condition and step < 10:
    #     # 随机选择一个未点击的格子
    #     x = random.randint(0, agent_game.settings["width"] - 1)
    #     y = random.randint(0, agent_game.settings["height"] - 1)
    #     state, reward, done = agent_game.agent_step((x, y))
    #     print(f"步骤 {step}: 点击({x},{y})，奖励: {reward:.1f}，是否结束: {done}")
    #     step += 1
