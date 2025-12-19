"""ぷよ関連のクラス定義"""

from dataclasses import dataclass
from typing import Tuple, List
from utils.constants import Color

@dataclass
class PuyoPair:
    """ぷよペア"""
    colors: Tuple[Color, Color]  # (上, 下)
    x: int                       # x座標 (0-5)
    y: int                       # y座標 (0-12)
    rotation: int                # 回転状態 (0-3)
    
    def get_positions(self) -> List[Tuple[int, int, Color]]:
        """回転状態に応じた各ぷよの位置と色を返す"""
        positions = []
        
        if self.rotation == 0:  # 縦向き（上が上、下が下）
            positions.append((self.x, self.y, self.colors[0]))      # 上
            positions.append((self.x, self.y + 1, self.colors[1]))  # 下
        elif self.rotation == 1:  # 右向き（上が右、下が左）
            positions.append((self.x, self.y, self.colors[1]))      # 左
            positions.append((self.x + 1, self.y, self.colors[0]))  # 右
        elif self.rotation == 2:  # 逆縦（下が上、上が下）
            positions.append((self.x, self.y, self.colors[1]))      # 上
            positions.append((self.x, self.y + 1, self.colors[0]))  # 下
        elif self.rotation == 3:  # 左向き（下が右、上が左）
            positions.append((self.x, self.y, self.colors[0]))      # 左
            positions.append((self.x + 1, self.y, self.colors[1]))  # 右
            
        return positions
    
    def can_rotate(self, board: List[List[Color]], direction: int = 1) -> bool:
        """回転可能かチェック"""
        new_rotation = (self.rotation + direction) % 4
        temp_pair = PuyoPair(self.colors, self.x, self.y, new_rotation)
        
        for px, py, _ in temp_pair.get_positions():
            # 盤面外チェック
            if px < 0 or px >= 6 or py < 0 or py >= 13:
                return False
            # 既存ぷよとの衝突チェック
            if board[py][px] != Color.EMPTY:
                return False
                
        return True
    
    def can_move(self, board: List[List[Color]], dx: int) -> bool:
        """移動可能かチェック"""
        temp_pair = PuyoPair(self.colors, self.x + dx, self.y, self.rotation)
        
        for px, py, _ in temp_pair.get_positions():
            # 盤面外チェック
            if px < 0 or px >= 6 or py < 0 or py >= 13:
                return False
            # 既存ぷよとの衝突チェック
            if board[py][px] != Color.EMPTY:
                return False
                
        return True
    
    def can_fall(self, board: List[List[Color]]) -> bool:
        """落下可能かチェック"""
        temp_pair = PuyoPair(self.colors, self.x, self.y + 1, self.rotation)
        
        for px, py, _ in temp_pair.get_positions():
            # 盤面外チェック
            if py >= 13:
                return False
            # 既存ぷよとの衝突チェック
            if board[py][px] != Color.EMPTY:
                return False
                
        return True