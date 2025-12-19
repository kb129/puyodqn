"""無能CPU実装"""

from typing import Dict, Any, List
from ai.base_player import BasePlayer
import random

class WeakCPU(BasePlayer):
    """最も深い谷に回転なしで落とす単純AI"""
    
    def __init__(self, player_id: str):
        super().__init__(player_id)
        self.name = "WeakCPU"
        self.target_column = None
        self.reaction_delay = 0
    
    def get_action(self, game_state: Dict[str, Any]) -> str:
        """行動決定"""
        player_state = game_state.get('players', {}).get(self.player_id)
        if not player_state or not player_state.get('current_puyo'):
            return "no_action"
        
        current_puyo = player_state['current_puyo']
        board = player_state['board']
        current_x = current_puyo['x']
        
        # 目標列が未設定、または新しいぷよの場合は再計算
        if self.target_column is None:
            self.target_column = self._find_deepest_column(board)
        
        # 人間らしい反応遅延（簡易版）
        if self.reaction_delay > 0:
            self.reaction_delay -= 1
            return "no_action"
        
        # 目標位置への移動
        if current_x < self.target_column:
            return "move_right"
        elif current_x > self.target_column:
            return "move_left"
        else:
            # 目標位置に到達したら落下
            self.target_column = None  # 次のぷよ用にリセット
            self.reaction_delay = random.randint(1, 3)  # 少しの遅延
            return "soft_drop"
    
    def _find_deepest_column(self, board: List[List[int]]) -> int:
        """最も深い（空きが多い）列を見つける"""
        depths = []
        
        for col in range(6):  # 6列
            depth = 0
            # 下から上へ空セルを数える
            for row in range(12, -1, -1):  # 13行（隠し段含む）
                if board[row][col] == 0:  # 空セル
                    depth += 1
                else:
                    break
            depths.append(depth)
        
        # 最も深い列を返す（同じ深さの場合は左優先）
        max_depth = max(depths)
        for col, depth in enumerate(depths):
            if depth == max_depth:
                return col
        
        return 2  # フォールバック（中央）
    
    def on_game_start(self):
        """ゲーム開始時の初期化"""
        self.target_column = None
        self.reaction_delay = 0
    
    def get_debug_info(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """デバッグ情報を返す"""
        player_state = game_state.get('players', {}).get(self.player_id)
        if not player_state:
            return {}
        
        board = player_state['board']
        depths = []
        
        for col in range(6):
            depth = 0
            for row in range(12, -1, -1):
                if board[row][col] == 0:
                    depth += 1
                else:
                    break
            depths.append(depth)
        
        return {
            'column_depths': depths,
            'target_column': self.target_column,
            'reaction_delay': self.reaction_delay,
            'deepest_column': self._find_deepest_column(board)
        }