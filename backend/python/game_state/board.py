"""盤面管理クラス"""

from typing import List, Set, Tuple, Optional
from utils.constants import *
from game_state.puyo import PuyoPair
import random

class Board:
    """ぷよぷよ盤面管理"""
    
    def __init__(self):
        # 6x13の盤面（13行目は隠し段）
        self.grid = [[Color.EMPTY for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
        
    def is_empty(self, x: int, y: int) -> bool:
        """指定位置が空かチェック"""
        if x < 0 or x >= BOARD_WIDTH or y < 0 or y >= BOARD_HEIGHT:
            return False
        return self.grid[y][x] == Color.EMPTY
    
    def set_puyo(self, x: int, y: int, color: Color):
        """指定位置にぷよを配置"""
        if 0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT:
            self.grid[y][x] = color
    
    def get_puyo(self, x: int, y: int) -> Color:
        """指定位置のぷよ色を取得"""
        if 0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT:
            return self.grid[y][x]
        return Color.EMPTY
    
    def place_puyo_pair(self, pair: PuyoPair) -> bool:
        """ぷよペアを盤面に配置"""
        positions = pair.get_positions()
        
        # 配置可能かチェック
        for px, py, _ in positions:
            if not self.is_empty(px, py):
                return False
        
        # 実際に配置
        for px, py, color in positions:
            self.set_puyo(px, py, color)
        
        return True
    
    def apply_gravity(self) -> bool:
        """重力を適用（落下処理）"""
        changed = False
        
        for x in range(BOARD_WIDTH):
            # 各列で下から詰める
            write_y = BOARD_HEIGHT - 1
            
            for read_y in range(BOARD_HEIGHT - 1, -1, -1):
                if self.grid[read_y][x] != Color.EMPTY:
                    if read_y != write_y:
                        self.grid[write_y][x] = self.grid[read_y][x]
                        self.grid[read_y][x] = Color.EMPTY
                        changed = True
                    write_y -= 1
        
        return changed
    
    def find_connected_group(self, start_x: int, start_y: int, color: Color, visited: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """指定位置から連結するぷよ群を探索"""
        if (start_x, start_y) in visited:
            return set()
        
        if (start_x < 0 or start_x >= BOARD_WIDTH or 
            start_y < 0 or start_y >= BOARD_HEIGHT or 
            self.grid[start_y][start_x] != color or
            color == Color.EMPTY or color == Color.OJAMA):
            return set()
        
        group = {(start_x, start_y)}
        visited.add((start_x, start_y))
        
        # 4方向に再帰探索
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dx, dy in directions:
            nx, ny = start_x + dx, start_y + dy
            group.update(self.find_connected_group(nx, ny, color, visited))
        
        return group
    
    def find_chains(self) -> List[Set[Tuple[int, int]]]:
        """消去可能な連結群を全て検出"""
        visited = set()
        chains = []
        
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                if (x, y) not in visited and self.grid[y][x] not in [Color.EMPTY, Color.OJAMA]:
                    group = self.find_connected_group(x, y, self.grid[y][x], visited)
                    if len(group) >= 4:  # 4個以上で消去
                        chains.append(group)
        
        return chains
    
    def find_adjacent_ojama(self, color_positions: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """色付きぷよに隣接するおじゃまぷよを検出"""
        ojama_positions = set()
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for x, y in color_positions:
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < BOARD_WIDTH and 0 <= ny < BOARD_HEIGHT and 
                    self.grid[ny][nx] == Color.OJAMA):
                    ojama_positions.add((nx, ny))
        
        return ojama_positions
    
    def remove_puyos(self, positions: Set[Tuple[int, int]]) -> int:
        """指定位置のぷよを消去（隣接おじゃまぷよも含む）"""
        # まず隣接するおじゃまぷよを検出
        adjacent_ojama = self.find_adjacent_ojama(positions)
        
        # 色付きぷよとおじゃまぷよを合わせて消去
        all_positions = positions | adjacent_ojama
        
        count = 0
        for x, y in all_positions:
            if self.grid[y][x] != Color.EMPTY:
                self.grid[y][x] = Color.EMPTY
                count += 1
        return count
    
    def check_gameover(self) -> bool:
        """ゲームオーバー判定"""
        for col in GAMEOVER_COLUMNS:
            if self.grid[GAMEOVER_ROW][col] != Color.EMPTY:
                return True
        return False
    
    def get_column_height(self, x: int) -> int:
        """指定列の高さ（下からの積み上げ数）を取得"""
        for y in range(BOARD_HEIGHT - 1, -1, -1):
            if self.grid[y][x] != Color.EMPTY:
                return BOARD_HEIGHT - y
        return 0
    
    def get_drop_position(self, x: int) -> int:
        """指定列でぷよが落ちる位置を取得"""
        for y in range(BOARD_HEIGHT - 1, -1, -1):
            if self.grid[y][x] == Color.EMPTY:
                return y
        return -1  # 列が満杯
    
    def add_ojama(self, count: int):
        """おじゃまぷよを追加（最大5段まで）"""
        if count <= 0:
            return 0
        
        # 最大5段分（30個）まで制限
        max_drop = 5 * BOARD_WIDTH  # 5段 × 6列 = 30個
        actual_drop = min(count, max_drop)
        
        # 各列に均等分配
        per_column = actual_drop // BOARD_WIDTH
        remainder = actual_drop % BOARD_WIDTH
        
        for x in range(BOARD_WIDTH):
            ojama_for_this_column = per_column
            if x < remainder:
                ojama_for_this_column += 1
            
            # 上から順に配置
            for _ in range(ojama_for_this_column):
                drop_y = self.get_drop_position(x)
                if drop_y >= 0:
                    self.set_puyo(x, drop_y, Color.OJAMA)
        
        # 降らせた分を返す
        return actual_drop
    
    def to_list(self) -> List[List[int]]:
        """盤面を2次元リストとして返す"""
        return [[int(cell) for cell in row] for row in self.grid]
    
    def from_list(self, grid: List[List[int]]):
        """2次元リストから盤面を復元"""
        for y in range(min(len(grid), BOARD_HEIGHT)):
            for x in range(min(len(grid[y]), BOARD_WIDTH)):
                self.grid[y][x] = Color(grid[y][x])
    
    def copy(self) -> 'Board':
        """盤面のコピーを作成"""
        new_board = Board()
        new_board.from_list(self.to_list())
        return new_board