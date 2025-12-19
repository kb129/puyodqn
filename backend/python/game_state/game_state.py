"""ゲーム状態管理"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from game_state.board import Board
from game_state.puyo import PuyoPair
from utils.constants import *
import random

@dataclass
class PlayerState:
    """プレイヤー状態"""
    player_id: str
    board: Board
    current_puyo: Optional[PuyoPair]
    next_puyos: List[PuyoPair]
    score: int
    is_chaining: bool
    chain_count: int
    ojama_pending: int
    
    def to_dict(self) -> dict:
        """辞書形式で返す"""
        return {
            'player_id': self.player_id,
            'board': self.board.to_list(),
            'current_puyo': {
                'colors': list(self.current_puyo.colors),
                'x': self.current_puyo.x,
                'y': self.current_puyo.y,
                'rotation': self.current_puyo.rotation
            } if self.current_puyo else None,
            'next_puyos': [list(pair.colors) for pair in self.next_puyos],
            'score': self.score,
            'is_chaining': self.is_chaining,
            'chain_count': self.chain_count,
            'ojama_pending': self.ojama_pending
        }

class PuyoGenerator:
    """ぷよ生成器"""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
    
    def generate_pair(self) -> PuyoPair:
        """ランダムなぷよペアを生成"""
        color1 = self.rng.choice(PUYO_COLORS)
        color2 = self.rng.choice(PUYO_COLORS)
        return PuyoPair((color1, color2), 2, 0, 0)  # 初期位置は(2,0)、回転0

class GameState:
    """ゲーム状態管理"""
    
    def __init__(self, mode: str = 'single', seed: Optional[int] = None):
        self.mode = mode
        self.seed = seed or random.randint(0, 1000000)
        self.generator = PuyoGenerator(self.seed)
        
        # プレイヤー初期化
        self.players = {}
        if mode == 'single':
            self.players['A'] = self._create_player('A')
        elif mode == 'versus':
            self.players['A'] = self._create_player('A')
            self.players['B'] = self._create_player('B')
        
        self.game_over = False
        self.winner = None
        self.turn = 0
    
    def _create_player(self, player_id: str) -> PlayerState:
        """プレイヤー状態を初期化"""
        board = Board()
        
        # 初期ネクスト生成
        next_puyos = [
            self.generator.generate_pair(),
            self.generator.generate_pair()
        ]
        
        # 最初のぷよペア
        current_puyo = self.generator.generate_pair()
        
        return PlayerState(
            player_id=player_id,
            board=board,
            current_puyo=current_puyo,
            next_puyos=next_puyos,
            score=0,
            is_chaining=False,
            chain_count=0,
            ojama_pending=0
        )
    
    def get_player(self, player_id: str) -> Optional[PlayerState]:
        """プレイヤー状態取得"""
        return self.players.get(player_id)
    
    def apply_action(self, player_id: str, action: str) -> bool:
        """プレイヤーの行動を適用"""
        player = self.get_player(player_id)
        if not player or not player.current_puyo or player.is_chaining:
            return False
        
        puyo = player.current_puyo
        
        if action == "move_left":
            if puyo.can_move(player.board.grid, -1):
                puyo.x -= 1
                return True
                
        elif action == "move_right":
            if puyo.can_move(player.board.grid, 1):
                puyo.x += 1
                return True
                
        elif action == "rotate_left":
            if puyo.can_rotate(player.board.grid, -1):
                puyo.rotation = (puyo.rotation - 1) % 4
                return True
                
        elif action == "rotate_right":
            if puyo.can_rotate(player.board.grid, 1):
                puyo.rotation = (puyo.rotation + 1) % 4
                return True
                
        elif action == "soft_drop":
            if puyo.can_fall(player.board.grid):
                puyo.y += 1
                return True
                
        return False
    
    def update_falling_puyo(self, player_id: str) -> bool:
        """落下中ぷよの自動落下処理"""
        player = self.get_player(player_id)
        if not player or not player.current_puyo or player.is_chaining:
            return False
        
        puyo = player.current_puyo
        
        # 自動落下
        if puyo.can_fall(player.board.grid):
            puyo.y += 1
            return True
        else:
            # 着地処理
            return self._land_puyo(player_id)
    
    def _land_puyo(self, player_id: str) -> bool:
        """ぷよ着地処理"""
        player = self.get_player(player_id)
        if not player or not player.current_puyo:
            return False
        
        # 盤面に配置
        if not player.board.place_puyo_pair(player.current_puyo):
            # 配置失敗（ゲームオーバー）
            self.game_over = True
            if self.mode == 'versus':
                other_id = 'B' if player_id == 'A' else 'A'
                self.winner = other_id
            return False
        
        # 着地後におじゃまぷよを適用（5段分ずつ）
        if self.mode == 'versus' and player.ojama_pending > 0:
            # 5段分（30個）を上限として降らせる
            max_drop_per_landing = 5 * BOARD_WIDTH  # 5段 × 6列 = 30個
            drop_amount = min(player.ojama_pending, max_drop_per_landing)
            
            dropped = player.board.add_ojama(drop_amount)
            player.ojama_pending = max(0, player.ojama_pending - dropped)
            
            # おじゃまぷよ配置後のゲームオーバーチェック
            if player.board.check_gameover():
                self.game_over = True
                other_id = 'B' if player_id == 'A' else 'A'
                self.winner = other_id
                return False
        
        # 次のぷよペア準備
        player.current_puyo = player.next_puyos.pop(0)
        player.next_puyos.append(self.generator.generate_pair())
        
        # 連鎖チェック開始
        self._start_chain_check(player_id)
        
        return True
    
    def _start_chain_check(self, player_id: str):
        """連鎖チェック開始"""
        player = self.get_player(player_id)
        if not player:
            return
        
        player.is_chaining = True
        player.chain_count = 0
        self._process_chain(player_id)
    
    def _process_chain(self, player_id: str) -> int:
        """連鎖処理（再帰的）"""
        player = self.get_player(player_id)
        if not player:
            return 0
        
        # 重力適用
        player.board.apply_gravity()
        
        # 消去可能群検出
        chains = player.board.find_chains()
        
        if not chains:
            # 連鎖終了
            player.is_chaining = False
            
            # ゲームオーバーチェック
            if player.board.check_gameover():
                self.game_over = True
                if self.mode == 'versus':
                    other_id = 'B' if player_id == 'A' else 'A'
                    self.winner = other_id
            
            return 0
        
        # 連鎖継続
        player.chain_count += 1
        total_removed = 0
        chain_score = 0
        
        # 各群を消去してスコア計算
        group_counts = []
        colors_used = set()
        
        for group in chains:
            removed_count = player.board.remove_puyos(group)
            total_removed += removed_count
            group_counts.append(removed_count)
            
            # 使用された色を記録
            if group:
                sample_pos = group[0]
                color = player.board.grid[sample_pos[1]][sample_pos[0]]
                if color != Color.EMPTY and color != Color.OJAMA:
                    colors_used.add(color)
        
        # 正式なぷよぷよスコア計算
        if total_removed > 0:
            # 基本点
            base_score = total_removed * 10
            
            # 連鎖ボーナス
            chain_bonus = CHAIN_BONUS[min(player.chain_count - 1, len(CHAIN_BONUS) - 1)]
            
            # 連結ボーナス (同時消し群の最大サイズに基づく)
            max_group_size = max(group_counts) if group_counts else 0
            connection_bonus = CONNECTION_BONUS[min(max_group_size - 4, len(CONNECTION_BONUS) - 1)] if max_group_size >= 4 else 0
            
            # 色数ボーナス
            color_count = len(colors_used)
            color_bonus = COLOR_BONUS[min(color_count - 1, len(COLOR_BONUS) - 1)] if color_count > 0 else 0
            
            # 総ボーナス（最低でも1倍）
            total_bonus = max(1, chain_bonus + connection_bonus + color_bonus)
            
            # 最終スコア
            chain_score = base_score * total_bonus
        
        player.score += int(chain_score)
        
        # おじゃまぷよ計算（対戦時）
        if self.mode == 'versus' and chain_score > 0:
            ojama_count = int(chain_score // OJAMA_RATE)
            if ojama_count > 0:
                other_id = 'B' if player_id == 'A' else 'A'
                other_player = self.get_player(other_id)
                if other_player:
                    other_player.ojama_pending += ojama_count
        
        # 次の連鎖をチェック
        return total_removed + self._process_chain(player_id)
    
    def apply_pending_ojama(self, player_id: str):
        """待機中のおじゃまぷよを適用（5段分ずつ）"""
        player = self.get_player(player_id)
        if not player or player.ojama_pending <= 0:
            return
        
        # 5段分（30個）を上限として降らせる
        max_drop_per_apply = 5 * BOARD_WIDTH  # 5段 × 6列 = 30個
        drop_amount = min(player.ojama_pending, max_drop_per_apply)
        
        dropped = player.board.add_ojama(drop_amount)
        player.ojama_pending = max(0, player.ojama_pending - dropped)
        
        # ゲームオーバーチェック
        if player.board.check_gameover():
            self.game_over = True
            if self.mode == 'versus':
                other_id = 'B' if player_id == 'A' else 'A'
                self.winner = other_id
    
    def to_dict(self) -> dict:
        """ゲーム状態を辞書で返す"""
        return {
            'mode': self.mode,
            'players': {pid: player.to_dict() for pid, player in self.players.items()},
            'game_over': self.game_over,
            'winner': self.winner,
            'turn': self.turn,
            'seed': self.seed
        }