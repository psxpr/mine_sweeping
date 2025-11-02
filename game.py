# import pygame
# import random
# import sys
# import time
#
# # 初始化pygame
# pygame.init()
#
# # 颜色定义
# WHITE = (255, 255, 255)
# GRAY = (192, 192, 192)
# LIGHT_GRAY = (220, 220, 220)
# DARK_GRAY = (100, 100, 100)
# BLACK = (0, 0, 0)
# RED = (255, 0, 0)
# BLUE = (0, 0, 255)
# GREEN = (0, 180, 0)
# YELLOW = (255, 200, 0)
# PURPLE = (128, 0, 128)
# BUTTON_BG = (240, 240, 240)
# BUTTON_HOVER = (220, 220, 255)
# RESTART_BTN_NORMAL = (0, 160, 230)  # 重新开始按钮默认色
# RESTART_BTN_HOVER = (0, 190, 255)  # 重新开始按钮悬停色
#
# # 游戏难度设置
# DIFFICULTIES = {
#     "简单": {"width": 10, "height": 10, "mines": 10},
#     "中等": {"width": 16, "height": 16, "mines": 40},
#     "困难": {"width": 30, "height": 16, "mines": 99}
# }
#
# # 设置字体 - 优先使用支持中文的字体
# pygame.font.init()
#
#
# # 尝试加载中文字体，若失败则使用默认字体
# def get_chinese_font(size):
#     # 常见的中文字体列表，根据系统情况调整
#     chinese_fonts = ["SimHei", "Microsoft YaHei", "Heiti TC", "WenQuanYi Micro Hei", "SimSun"]
#
#     for font_name in chinese_fonts:
#         try:
#             return pygame.font.SysFont(font_name, size)
#         except:
#             continue
#
#     # 如果所有中文字体都加载失败，使用默认字体
#     return pygame.font.SysFont(None, size)
#
#
# # 初始化不同大小的字体
# small_font = get_chinese_font(16)
# medium_font = get_chinese_font(20)
# large_font = get_chinese_font(32)
#
#
# class Cell:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#         self.is_mine = False
#         self.is_revealed = False
#         self.mark_type = 0  # 0: 无标记, 1: 地雷标记, 2: 问号标记
#         self.adjacent_mines = 0
#
#
# class Minesweeper:
#     def __init__(self):
#         # 默认难度为简单
#         self.difficulty = "简单"
#         self.settings = DIFFICULTIES[self.difficulty]
#
#         # 计算窗口大小
#         self.cell_size = 30
#         self.header_height = 100  # 头部信息栏高度
#         self.window_width = self.settings["width"] * self.cell_size
#         self.window_height = self.settings["height"] * self.cell_size + self.header_height
#
#         # 初始化窗口
#         self.screen = pygame.display.set_mode((self.window_width, self.window_height))
#         pygame.display.set_caption("扫雷")
#
#         # 游戏状态变量
#         self.cells = []
#         self.mines_placed = False
#         self.game_over = False
#         self.victory = False
#         self.start_time = None
#         self.elapsed_time = 0
#         self.hovered_button = None  # 用于跟踪鼠标悬停的难度按钮
#         self.hovered_restart_btn = False  # 跟踪重新开始按钮悬停状态
#         self.reset_game()
#
#         # 时钟
#         self.clock = pygame.time.Clock()
#
#     def reset_game(self):
#         """重置游戏状态"""
#         # 初始化格子
#         self.cells = [
#             [Cell(x, y) for y in range(self.settings["height"])]
#             for x in range(self.settings["width"])
#         ]
#
#         # 重置游戏状态
#         self.mines_placed = False
#         self.game_over = False
#         self.victory = False
#         self.start_time = None
#         self.elapsed_time = 0
#         self.hovered_button = None
#         self.hovered_restart_btn = False
#
#     def set_difficulty(self, difficulty):
#         """更改游戏难度"""
#         if difficulty in DIFFICULTIES and difficulty != self.difficulty:
#             self.difficulty = difficulty
#             self.settings = DIFFICULTIES[difficulty]
#
#             # 重新计算窗口大小
#             self.window_width = self.settings["width"] * self.cell_size
#             self.window_height = self.settings["height"] * self.cell_size + self.header_height
#
#             # 重新设置窗口
#             self.screen = pygame.display.set_mode((self.window_width, self.window_height))
#
#             # 重置游戏
#             self.reset_game()
#
#     def place_mines(self, first_click_x, first_click_y):
#         """在第一次点击后放置地雷，确保第一次点击不是地雷"""
#         mines_placed = 0
#
#         # 确保首次点击位置及其周围没有地雷
#         safe_zone = []
#         for dx in [-1, 0, 1]:
#             for dy in [-1, 0, 1]:
#                 nx, ny = first_click_x + dx, first_click_y + dy
#                 if 0 <= nx < self.settings["width"] and 0 <= ny < self.settings["height"]:
#                     safe_zone.append((nx, ny))
#
#         while mines_placed < self.settings["mines"]:
#             x = random.randint(0, self.settings["width"] - 1)
#             y = random.randint(0, self.settings["height"] - 1)
#
#             if (x, y) not in safe_zone and not self.cells[x][y].is_mine:
#                 self.cells[x][y].is_mine = True
#                 mines_placed += 1
#
#         self.calculate_adjacent_mines()
#         self.mines_placed = True
#         self.start_time = time.time()  # 开始计时
#
#     def calculate_adjacent_mines(self):
#         """计算每个格子周围的地雷数量"""
#         for x in range(self.settings["width"]):
#             for y in range(self.settings["height"]):
#                 if not self.cells[x][y].is_mine:
#                     count = 0
#                     for dx in [-1, 0, 1]:
#                         for dy in [-1, 0, 1]:
#                             nx, ny = x + dx, y + dy
#                             if 0 <= nx < self.settings["width"] and 0 <= ny < self.settings["height"]:
#                                 if self.cells[nx][ny].is_mine:
#                                     count += 1
#                     self.cells[x][y].adjacent_mines = count
#
#     def reveal_cell(self, x, y):
#         """翻开一个格子，如果是空白则递归翻开周围格子"""
#         if not 0 <= x < self.settings["width"] or not 0 <= y < self.settings["height"]:
#             return
#
#         cell = self.cells[x][y]
#
#         if cell.is_revealed or cell.mark_type != 0:  # 已标记的格子不能翻开
#             return
#
#         cell.is_revealed = True
#
#         # 如果踩中地雷，游戏结束
#         if cell.is_mine:
#             self.game_over = True
#             return
#
#         # 如果是空白格子（周围没有地雷），递归翻开周围格子
#         if cell.adjacent_mines == 0:
#             for dx in [-1, 0, 1]:
#                 for dy in [-1, 0, 1]:
#                     if dx == 0 and dy == 0:
#                         continue
#                     self.reveal_cell(x + dx, y + dy)
#
#     def toggle_mark(self, x, y):
#         """切换格子的标记类型：无→地雷→问号→无"""
#         if not 0 <= x < self.settings["width"] or not 0 <= y < self.settings["height"]:
#             return
#
#         cell = self.cells[x][y]
#
#         if cell.is_revealed:  # 已翻开的格子不能标记
#             return
#
#         # 循环切换标记类型
#         cell.mark_type = (cell.mark_type + 1) % 3
#
#     def check_victory(self):
#         """检查是否获胜（所有非地雷格子都被翻开）"""
#         if self.game_over:
#             return False
#
#         for x in range(self.settings["width"]):
#             for y in range(self.settings["height"]):
#                 cell = self.cells[x][y]
#                 if not cell.is_mine and not cell.is_revealed:
#                     return False
#
#         self.victory = True
#         return True
#
#     def count_remaining_mines(self):
#         """计算剩余地雷数量（总地雷数减去已标记的数量）"""
#         flagged_count = sum(1 for x in range(self.settings["width"])
#                             for y in range(self.settings["height"])
#                             if self.cells[x][y].mark_type == 1)
#         return self.settings["mines"] - flagged_count
#
#     def update_time(self):
#         """更新游戏用时"""
#         if self.start_time and not self.game_over and not self.victory:
#             self.elapsed_time = int(time.time() - self.start_time)
#
#     def draw_header(self):
#         """绘制头部信息栏（分为上下两部分）"""
#         # 绘制头部背景
#         header_rect = pygame.Rect(0, 0, self.window_width, self.header_height)
#         pygame.draw.rect(self.screen, LIGHT_GRAY, header_rect)
#         pygame.draw.line(self.screen, DARK_GRAY,
#                          (0, self.header_height),
#                          (self.window_width, self.header_height), 2)
#
#         # 下半部分 - 信息和按钮
#         # 绘制分隔线
#         pygame.draw.line(self.screen, DARK_GRAY,
#                          (0, 60),
#                          (self.window_width, 60), 1)
#
#         # 显示剩余地雷数
#         mine_count = self.count_remaining_mines()
#         mine_text = medium_font.render(f"剩余地雷: {max(0, mine_count)}", True, BLACK)
#         mine_rect = mine_text.get_rect(topleft=(20, 20))
#         self.screen.blit(mine_text, mine_rect)
#
#         # 显示用时
#         time_text = medium_font.render(f"用时: {self.elapsed_time}s", True, BLACK)
#         time_rect = time_text.get_rect(topright=(self.window_width - 20, 20))
#         self.screen.blit(time_text, time_rect)
#
#         # 按钮配置常量
#         BUTTON_WIDTH = 70
#         BUTTON_HEIGHT = 30
#         BUTTON_Y = 65
#         BUTTON_SPACING = BUTTON_WIDTH * 1.2  # 按钮之间的间距
#
#         # 计算难度按钮位置（三个按钮水平居中分布）
#         easy_rect = pygame.Rect(
#             self.window_width // 2 - BUTTON_WIDTH * 1.5 - 10,  # 左侧按钮
#             BUTTON_Y,
#             BUTTON_WIDTH,
#             BUTTON_HEIGHT
#         )
#
#         medium_rect = pygame.Rect(
#             self.window_width // 2 - BUTTON_WIDTH // 2,  # 中央按钮
#             BUTTON_Y,
#             BUTTON_WIDTH,
#             BUTTON_HEIGHT
#         )
#
#         hard_rect = pygame.Rect(
#             self.window_width // 2 + BUTTON_WIDTH * 0.5 + 10,  # 右侧按钮
#             BUTTON_Y,
#             BUTTON_WIDTH,
#             BUTTON_HEIGHT
#         )
#
#         # 检查鼠标悬停状态（难度按钮）
#         mouse_pos = pygame.mouse.get_pos()
#         self.hovered_button = None
#         if easy_rect.collidepoint(mouse_pos):
#             self.hovered_button = "简单"
#         elif medium_rect.collidepoint(mouse_pos):
#             self.hovered_button = "中等"
#         elif hard_rect.collidepoint(mouse_pos):
#             self.hovered_button = "困难"
#
#         # 绘制难度按钮
#         difficulty_buttons = [
#             (easy_rect, "简单"),
#             (medium_rect, "中等"),
#             (hard_rect, "困难")
#         ]
#
#         for rect, text in difficulty_buttons:
#             # 根据状态设置按钮颜色
#             if self.difficulty == text:
#                 color = GREEN
#             elif self.hovered_button == text:
#                 color = BUTTON_HOVER
#             else:
#                 color = BUTTON_BG
#
#             pygame.draw.rect(self.screen, color, rect, border_radius=5)
#             pygame.draw.rect(self.screen, DARK_GRAY, rect, 1, border_radius=5)
#             button_text = small_font.render(text, True, BLACK)
#             text_rect = button_text.get_rect(center=rect.center)
#             self.screen.blit(button_text, text_rect)
#
#         return easy_rect, medium_rect, hard_rect
#
#     def draw_grid(self):
#         """绘制游戏网格"""
#         # 绘制网格背景
#         grid_bg_rect = pygame.Rect(
#             0, self.header_height,
#             self.window_width, self.window_height - self.header_height
#         )
#         pygame.draw.rect(self.screen, GRAY, grid_bg_rect)
#
#         for x in range(self.settings["width"]):
#             for y in range(self.settings["height"]):
#                 cell = self.cells[x][y]
#                 # 计算格子在屏幕上的位置（考虑头部高度）
#                 rect = pygame.Rect(
#                     x * self.cell_size,
#                     y * self.cell_size + self.header_height,
#                     self.cell_size - 1,
#                     self.cell_size - 1
#                 )
#
#                 if cell.is_revealed:
#                     pygame.draw.rect(self.screen, WHITE, rect)
#
#                     if cell.is_mine:
#                         # 绘制地雷
#                         pygame.draw.circle(self.screen, RED,
#                                            (x * self.cell_size + self.cell_size // 2,
#                                             y * self.cell_size + self.header_height + self.cell_size // 2),
#                                            self.cell_size // 3)
#                     elif cell.adjacent_mines > 0:
#                         # 绘制周围地雷数量，不同数字不同颜色
#                         color = BLACK
#                         if cell.adjacent_mines == 1:
#                             color = BLUE
#                         elif cell.adjacent_mines == 2:
#                             color = GREEN
#                         elif cell.adjacent_mines == 3:
#                             color = RED
#                         elif cell.adjacent_mines == 4:
#                             color = (0, 0, 128)  # 深蓝色
#                         elif cell.adjacent_mines == 5:
#                             color = (128, 0, 0)  # 棕色
#                         elif cell.adjacent_mines == 6:
#                             color = (0, 128, 128)  # 青色
#                         elif cell.adjacent_mines == 7:
#                             color = BLACK
#                         elif cell.adjacent_mines == 8:
#                             color = GRAY
#
#                         text = small_font.render(str(cell.adjacent_mines), True, color)
#                         text_rect = text.get_rect(center=rect.center)
#                         self.screen.blit(text, text_rect)
#                 else:
#                     # 未翻开的格子 - 添加立体效果
#                     pygame.draw.rect(self.screen, LIGHT_GRAY, rect)
#                     pygame.draw.line(self.screen, WHITE, rect.topleft, rect.topright, 2)
#                     pygame.draw.line(self.screen, WHITE, rect.topleft, rect.bottomleft, 2)
#                     pygame.draw.line(self.screen, DARK_GRAY, rect.bottomright, rect.topright, 1)
#                     pygame.draw.line(self.screen, DARK_GRAY, rect.bottomright, rect.bottomleft, 1)
#
#                     # 地雷标记
#                     if cell.mark_type == 1:
#                         # 绘制旗帜形状而非方块
#                         flag_points = [
#                             (x * self.cell_size + 5, y * self.cell_size + self.header_height + 5),
#                             (x * self.cell_size + 5, y * self.cell_size + self.header_height + self.cell_size - 10),
#                             (x * self.cell_size + self.cell_size - 10,
#                              y * self.cell_size + self.header_height + self.cell_size // 2)
#                         ]
#                         pygame.draw.polygon(self.screen, RED, flag_points)
#                         pygame.draw.rect(self.screen, YELLOW,
#                                          (x * self.cell_size + 5,
#                                           y * self.cell_size + self.header_height + 5,
#                                           3, self.cell_size - 5))
#                     # 问号标记
#                     elif cell.mark_type == 2:
#                         question = small_font.render("?", True, PURPLE)
#                         question_rect = question.get_rect(center=rect.center)
#                         self.screen.blit(question, question_rect)
#
#     def draw_game_result(self):
#         """绘制游戏结束/胜利信息（含重新开始按钮）"""
#         if not self.game_over and not self.victory:
#             return
#
#         # 结果文本
#         result_text = large_font.render("游戏结束!", True, RED) if self.game_over else large_font.render("恭喜胜利!",
#                                                                                                          True, GREEN)
#
#         # 绘制半透明背景
#         overlay = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
#         overlay.fill((255, 255, 255, 200))
#         self.screen.blit(overlay, (0, 0))
#
#         # 绘制文本背景
#         text_bg = pygame.Rect(0, 0, 300, 150)
#         text_bg.center = (self.window_width // 2, self.window_height // 2)
#         pygame.draw.rect(self.screen, WHITE, text_bg, border_radius=10)
#         pygame.draw.rect(self.screen, DARK_GRAY, text_bg, 2, border_radius=10)
#
#         # 绘制结果文本
#         text_rect = result_text.get_rect(center=(self.window_width // 2, self.window_height // 2 - 30))
#         self.screen.blit(result_text, text_rect)
#
#         # 绘制重新开始按钮
#         restart_btn = pygame.Rect(0, 0, 150, 40)
#         restart_btn.center = (self.window_width // 2, self.window_height // 2 + 20)
#
#         # 检查鼠标是否悬停在按钮上
#         mouse_pos = pygame.mouse.get_pos()
#         self.hovered_restart_btn = restart_btn.collidepoint(mouse_pos)
#
#         # 根据状态设置按钮颜色
#         btn_color = RESTART_BTN_HOVER if self.hovered_restart_btn else RESTART_BTN_NORMAL
#
#         pygame.draw.rect(self.screen, btn_color, restart_btn, border_radius=5)
#         pygame.draw.rect(self.screen, DARK_GRAY, restart_btn, 2, border_radius=5)
#
#         restart_text = medium_font.render("重新开始", True, WHITE)
#         restart_text_rect = restart_text.get_rect(center=restart_btn.center)
#         self.screen.blit(restart_text, restart_text_rect)
#
#         return restart_btn
#
#     def draw(self):
#         """绘制整个游戏界面"""
#         self.screen.fill(WHITE)
#
#         # 绘制头部信息栏并获取按钮区域
#         easy_rect, medium_rect, hard_rect = self.draw_header()
#
#         # 绘制网格
#         self.draw_grid()
#
#         # 如果游戏结束或胜利，显示相应信息和重新开始按钮
#         restart_btn = None
#         if self.game_over or self.victory:
#             restart_btn = self.draw_game_result()
#
#         pygame.display.flip()
#         return easy_rect, medium_rect, hard_rect, restart_btn
#
#     def run(self):
#         """游戏主循环"""
#         running = True
#         while running:
#             self.clock.tick(30)
#             self.update_time()
#
#             # 获取按钮区域
#             easy_rect, medium_rect, hard_rect, restart_btn = self.draw()
#
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     running = False
#
#                 if event.type == pygame.MOUSEBUTTONDOWN:
#                     x, y = pygame.mouse.get_pos()
#
#                     # 游戏结束或胜利状态下，检查是否点击重新开始按钮
#                     if (self.game_over or self.victory) and restart_btn and restart_btn.collidepoint(x, y):
#                         self.reset_game()
#                         continue
#
#                     # 游戏进行中，处理难度按钮点击
#                     if not self.game_over and not self.victory and y < self.header_height:
#                         if easy_rect.collidepoint(x, y):
#                             self.set_difficulty("简单")
#                         elif medium_rect.collidepoint(x, y):
#                             self.set_difficulty("中等")
#                         elif hard_rect.collidepoint(x, y):
#                             self.set_difficulty("困难")
#                         continue
#
#                     # 游戏进行中，处理格子点击
#                     if not self.game_over and not self.victory:
#                         # 计算点击的格子坐标（调整头部高度）
#                         grid_x = x // self.cell_size
#                         grid_y = (y - self.header_height) // self.cell_size
#
#                         # 左键点击：翻开格子
#                         if event.button == 1:
#                             # 第一次点击时放置地雷
#                             if not self.mines_placed:
#                                 self.place_mines(grid_x, grid_y)
#                             self.reveal_cell(grid_x, grid_y)
#                             self.check_victory()
#
#                         # 右键点击：切换标记类型
#                         elif event.button == 3:
#                             self.toggle_mark(grid_x, grid_y)
#
#         pygame.quit()
#         sys.exit()
#
#
# if __name__ == "__main__":
#     game = Minesweeper()
#     game.run()