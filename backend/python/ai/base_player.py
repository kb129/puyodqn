"""AIプレイヤーの基底クラス"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from game_state.game_state import PlayerState

class BasePlayer(ABC):
    """AIプレイヤーの基底クラス"""
    
    def __init__(self, player_id: str):
        self.player_id = player_id
        self.name = "BasePlayer"
    
    @abstractmethod
    def get_action(self, game_state: Dict[str, Any]) -> str:
        """
        ゲーム状態を受け取り、次の行動を返す
        
        Args:
            game_state: ゲーム状態辞書
            
        Returns:
            行動文字列 ('move_left', 'move_right', 'rotate_left', 'rotate_right', 'soft_drop', 'no_action')
        """
        pass
    
    def on_game_start(self):
        """ゲーム開始時の初期化処理"""
        pass
    
    def on_game_end(self, result: Dict[str, Any]):
        """ゲーム終了時の処理"""
        pass