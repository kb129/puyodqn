"""定数定義"""

from enum import IntEnum

# 盤面サイズ
BOARD_WIDTH = 6
BOARD_HEIGHT = 13
VISIBLE_HEIGHT = 12

# 色定義
class Color(IntEnum):
    EMPTY = 0
    RED = 1
    BLUE = 2
    GREEN = 3
    YELLOW = 4
    OJAMA = 5

# 行動定義
class Action(IntEnum):
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    ROTATE_LEFT = 2
    ROTATE_RIGHT = 3
    SOFT_DROP = 4
    NO_ACTION = 5

ACTION_NAMES = [
    "move_left",
    "move_right", 
    "rotate_left",
    "rotate_right",
    "soft_drop",
    "no_action"
]

# ゲーム設定
PUYO_COLORS = [Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW]
CHAIN_BONUS = [0, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512]
CONNECTION_BONUS = [0, 2, 3, 4, 5, 6, 7, 10]  # 4, 5, 6, 7, 8, 9, 10, 11+
COLOR_BONUS = [0, 3, 6, 12, 24]  # 1, 2, 3, 4, 5色

# おじゃまぷよ設定
OJAMA_RATE = 70  # 70点で1個
SMALL_OJAMA = 1
LARGE_OJAMA = 6
ROCK_OJAMA = 30

# ゲームオーバーライン
GAMEOVER_COLUMNS = [2, 3]
GAMEOVER_ROW = 12