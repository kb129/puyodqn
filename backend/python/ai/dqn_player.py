"""DQNプレイヤー実装 - 位置ベース行動空間版"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from ai.base_player import BasePlayer
from game_state.puyo import PuyoPair
from utils.constants import Color
import copy
import itertools


@dataclass
class PlacementAction:
    """配置行動クラス"""
    action_id: int       # 0-23の行動ID
    column: int          # 着地列 (0-5)
    rotation: int        # 回転状態 (0-3)
    final_positions: List[Tuple[int, int, Color]]  # 最終配置位置
    
    @classmethod
    def from_id(cls, action_id: int, puyo_pair: PuyoPair, board: List[List[Color]]) -> 'PlacementAction':
        """行動IDから配置行動を生成"""
        column = action_id // 4
        rotation = action_id % 4
        
        # 落下シミュレーションで最終位置計算
        final_positions = cls._simulate_drop(puyo_pair, board, column, rotation)
        
        return cls(
            action_id=action_id,
            column=column,
            rotation=rotation,
            final_positions=final_positions
        )
    
    @staticmethod
    def _simulate_drop(puyo_pair: PuyoPair, board: List[List[Color]], 
                      target_column: int, target_rotation: int) -> List[Tuple[int, int, Color]]:
        """落下シミュレーション"""
        # 回転を適用
        temp_pair = PuyoPair(puyo_pair.colors, target_column, 0, target_rotation)
        
        # 可能な限り落下
        while temp_pair.can_fall(board):
            temp_pair.y += 1
        
        return temp_pair.get_positions()


class ActionSpace:
    """動的行動空間管理"""
    
    MAX_ACTIONS = 24  # 固定サイズ (6列 × 4回転)
    
    @staticmethod
    def get_valid_actions(puyo_pair: PuyoPair, board: List[List[Color]]) -> List[PlacementAction]:
        """現在の盤面で有効な全配置行動を生成"""
        valid_actions = []
        
        for action_id in range(ActionSpace.MAX_ACTIONS):
            try:
                action = PlacementAction.from_id(action_id, puyo_pair, board)
                if ActionSpace._is_valid_placement(action, board):
                    valid_actions.append(action)
            except:
                # 配置不可能（盤面外・衝突等）
                continue
                
        return valid_actions
    
    @staticmethod
    def _is_valid_placement(action: PlacementAction, board: List[List[Color]]) -> bool:
        """配置の妥当性チェック"""
        for px, py, _ in action.final_positions:
            # 盤面範囲チェック
            if px < 0 or px >= 6 or py < 0 or py >= 13:
                return False
            # 既存ぷよとの衝突チェック
            if board[py][px] != Color.EMPTY:
                return False
        return True
    
    @staticmethod
    def create_action_mask(valid_actions: List[PlacementAction]) -> torch.Tensor:
        """有効行動マスクを生成"""
        mask = torch.zeros(ActionSpace.MAX_ACTIONS, dtype=torch.bool)
        for action in valid_actions:
            mask[action.action_id] = True
        return mask


@dataclass
class TacticalFeatures:
    """戦術的状況の特徴量"""
    
    # 連鎖可能性分析
    max_chain_potential: int           # 現在盤面から可能な最大連鎖数
    immediate_chain_count: int         # 1手で発動できる連鎖数
    chain_trigger_positions: List[Tuple[int, int]]  # 連鎖発動可能位置
    
    # ネクスト活用分析  
    next_puyo_synergy: float          # ネクストぷよとの相性度 (0-1)
    next2_puyo_synergy: float         # ネクスト2との相性度 (0-1)
    optimal_placement_exists: bool    # 理想的配置が可能か
    
    # 防御状況
    height_danger_level: float        # 積み上げ危険度 (0-1)
    column_balance_score: float       # 列バランス評価 (0-1)
    
    # 対戦時の相対状況
    opponent_max_chain: int           # 相手の最大連鎖可能数
    opponent_immediate_threat: int    # 相手の即座脅威レベル
    speed_advantage: float            # 速度的優位性 (-1 to 1)


class PuyoDQN(nn.Module):
    """位置ベース行動空間対応のDQNネットワーク"""
    
    def __init__(self, input_size=444, hidden_size=512, output_size=24):
        super().__init__()
        
        # 盤面特徴抽出 (CNN)
        self.board_conv = nn.Sequential(
            # 第1畳み込み層: 色チャネル処理
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 第2畳み込み層: 空間パターン抽出  
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 第3畳み込み層: 高次特徴
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 固定サイズプーリング（MPS対応）
            nn.AvgPool2d(kernel_size=2, stride=2),  # 6×13 -> 3×6（端数切り捨て）
            nn.AvgPool2d(kernel_size=(3, 2), stride=(3, 2))  # 3×6 -> 1×3
        )
        
        # 戦術特徴処理
        self.tactical_fc = nn.Sequential(
            nn.Linear(20, 64),  # 戦術特徴(12) + 相対情報(8) = 20
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # メイン決定ネットワーク
        self.main_fc = nn.Sequential(
            nn.Linear(128*1*3 + 64, hidden_size),  # 盤面特徴(128×1×3) + 戦術特徴(64)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, output_size)    # 24行動 (6列×4回転)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 盤面特徴抽出 (390次元)
        board_features = x[:, :390].view(batch_size, 5, 13, 6)  # 5色×13行×6列
        board_features = board_features.permute(0, 1, 3, 2)     # 5色×6列×13行に変換
        board_out = self.board_conv(board_features).flatten(1)   # バッチ×(128×1×3)
        
        # 戦術・相対特徴処理 (20次元)
        tactical_features = x[:, 424:]  # 基本情報(4) + 戦術(12) + 相対(8) = 24次元のうち後20次元
        tactical_out = self.tactical_fc(tactical_features)
        
        # 結合して最終決定
        combined = torch.cat([board_out, tactical_out], dim=1)
        q_values = self.main_fc(combined)
        
        return q_values


class GameOverException(Exception):
    """有効行動が存在しない場合の例外"""
    pass


class DQNPlayer(BasePlayer):
    """深層Q学習によるぷよぷよAI - 位置ベース行動空間"""
    
    def __init__(self, player_id: str, model_path: Optional[str] = None):
        super().__init__(player_id)
        self.name = "DQNPlayer"
        # デバイス選択（MPS > CUDA > CPU）
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.model = PuyoDQN(input_size=444, output_size=24).to(self.device)
        self.epsilon = 0.1  # 探索率
        
        if model_path:
            self.load_model(model_path)
        
    def get_action(self, game_state: Dict[str, Any]) -> str:
        """位置ベース行動選択（動的マスキング）"""
        try:
            player_state = game_state['players'][self.player_id]
            puyo_pair = self._parse_puyo_pair(player_state['current_puyo'])
            board = player_state['board']
            
            # 有効行動を取得
            valid_actions = ActionSpace.get_valid_actions(puyo_pair, board)
            
            if not valid_actions:
                raise GameOverException("No valid actions available")
            
            # ε-greedy選択
            if random.random() < self.epsilon:
                selected_action = random.choice(valid_actions)  # 探索
            else:
                # Q値計算 + 動的マスキング
                state_tensor = self.encode_state(game_state).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    q_values = self.model(state_tensor).squeeze(0)
                
                # 無効行動をマスク（MPS対応）
                action_mask = ActionSpace.create_action_mask(valid_actions).to(self.device)
                masked_q_values = q_values.clone()
                masked_q_values[~action_mask] = -1e9
                
                # 最適行動選択
                best_action_id = masked_q_values.argmax().item()
                
                selected_action = None
                for action in valid_actions:
                    if action.action_id == best_action_id:
                        selected_action = action
                        break
                
                if selected_action is None:
                    selected_action = valid_actions[0]  # フォールバック
            
            # 位置ベース行動を従来の行動形式に変換
            return self._convert_to_legacy_action(selected_action, player_state)
            
        except Exception as e:
            # エラー時は無能CPUにフォールバック
            return self._fallback_action(game_state)
    
    def _parse_puyo_pair(self, current_puyo: Dict[str, Any]) -> PuyoPair:
        """辞書形式のぷよをPuyoPairに変換"""
        return PuyoPair(
            colors=(Color(current_puyo['colors'][0]), Color(current_puyo['colors'][1])),
            x=current_puyo['x'],
            y=current_puyo['y'],
            rotation=current_puyo['rotation']
        )
    
    def _convert_to_legacy_action(self, placement_action: PlacementAction, player_state: Dict[str, Any]) -> str:
        """PlacementActionを従来の行動文字列に変換"""
        current_puyo = player_state['current_puyo']
        target_x = placement_action.column
        target_rotation = placement_action.rotation
        current_x = current_puyo['x']
        current_rotation = current_puyo['rotation']
        
        # 回転が必要な場合
        if current_rotation != target_rotation:
            # 最短回転方向を選択
            rotation_diff = (target_rotation - current_rotation) % 4
            if rotation_diff == 1 or rotation_diff == -3:
                return "rotate_right"
            else:
                return "rotate_left"
        
        # 移動が必要な場合
        if current_x < target_x:
            return "move_right"
        elif current_x > target_x:
            return "move_left"
        
        # 目標位置・回転に到達していれば落下
        return "soft_drop"
    
    def _fallback_action(self, game_state: Dict[str, Any]) -> str:
        """エラー時のフォールバック行動（無能CPUロジック）"""
        player_state = game_state.get('players', {}).get(self.player_id)
        if not player_state or not player_state.get('current_puyo'):
            return "no_action"
        
        current_puyo = player_state['current_puyo']
        board = player_state['board']
        current_x = current_puyo['x']
        
        # 最も深い列を見つける
        target_column = self._find_deepest_column(board)
        
        # 目標位置への移動
        if current_x < target_column:
            return "move_right"
        elif current_x > target_column:
            return "move_left"
        else:
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
    
    def encode_state(self, game_state: Dict[str, Any]) -> torch.Tensor:
        """
        戦術的特徴量を含む拡張状態表現
        
        構成要素:
        - 盤面状態: 6x13x5 (390次元) - 各セルの色one-hot
        - 現在ぷよ: 2x5 (10次元) - 2個のぷよone-hot  
        - ネクスト: 4x5 (20次元) - next1, next2のone-hot
        - 基本情報: 4次元 - [score, ojama_pending, chain_count, is_chaining]
        - 戦術特徴: 12次元 - 連鎖可能性、ネクスト相性、脅威レベル等
        - 相対情報: 8次元 - 対戦時の相手との比較情報
        
        合計: 390 + 10 + 20 + 4 + 12 + 8 = 444次元
        """
        
        player_state = game_state['players'][self.player_id]
        
        # 基本盤面エンコード (390次元)
        board_tensor = self._encode_board(player_state['board'])
        
        # ぷよエンコード (30次元)
        puyo_tensor = self._encode_puyos(player_state)
        
        # 基本情報 (4次元)
        basic_info = torch.tensor([
            player_state['score'] / 100000.0,        # スコア正規化
            player_state.get('ojama_pending', 0) / 50.0,    # おじゃま正規化  
            player_state.get('chain_count', 0) / 10.0,      # 連鎖数正規化
            float(player_state.get('is_chaining', False))   # 連鎖中フラグ
        ])
        
        # 戦術特徴抽出 (12次元) - 簡易版
        tactical_tensor = self._encode_tactical_features(game_state)
        
        # 相対情報 (8次元) - 対戦時のみ
        relative_info = torch.zeros(8)
        if len(game_state['players']) > 1:
            relative_info = self._encode_relative_state(game_state)
        
        return torch.cat([
            board_tensor.flatten(),
            puyo_tensor.flatten(),
            basic_info,
            tactical_tensor, 
            relative_info
        ])
    
    def _encode_board(self, board: List[List[int]]) -> torch.Tensor:
        """盤面をone-hotエンコード"""
        # 6x13x5 のone-hot表現
        board_tensor = torch.zeros(6, 13, 5)
        
        for row in range(13):
            for col in range(6):
                color = board[row][col]
                if 0 <= color <= 4:  # 0:空, 1-4:色, 5:おじゃま
                    board_tensor[col, row, color] = 1.0
        
        return board_tensor
    
    def _encode_puyos(self, player_state: Dict[str, Any]) -> torch.Tensor:
        """現在ぷよ・ネクストぷよをエンコード"""
        puyo_tensor = torch.zeros(30)  # 2+4 = 6個のぷよ × 5色
        
        # 現在ぷよ (10次元)
        if player_state.get('current_puyo'):
            colors = player_state['current_puyo']['colors']
            for i, color in enumerate(colors[:2]):
                if 1 <= color <= 4:
                    puyo_tensor[i * 5 + (color - 1)] = 1.0
        
        # ネクストぷよ (20次元)
        next_puyos = player_state.get('next_puyos', [])
        for puyo_idx, next_puyo in enumerate(next_puyos[:2]):
            colors = next_puyo.get('colors', [])
            for i, color in enumerate(colors[:2]):
                if 1 <= color <= 4:
                    offset = 10 + puyo_idx * 10 + i * 5
                    puyo_tensor[offset + (color - 1)] = 1.0
        
        return puyo_tensor
    
    def _encode_tactical_features(self, game_state: Dict[str, Any]) -> torch.Tensor:
        """戦術特徴の簡易エンコード"""
        # 簡易版：基本的な特徴のみ
        player_state = game_state['players'][self.player_id]
        board = player_state['board']
        
        # 高さ計算
        heights = []
        for col in range(6):
            height = 0
            for row in range(13):
                if board[row][col] != 0:
                    height = 13 - row
                    break
            heights.append(height)
        
        avg_height = sum(heights) / 6.0
        max_height = max(heights)
        height_variance = sum((h - avg_height) ** 2 for h in heights) / 6.0
        
        return torch.tensor([
            0.0,  # max_chain_potential (未実装)
            0.0,  # immediate_chain_count (未実装)
            0.5,  # next_puyo_synergy (デフォルト値)
            0.5,  # next2_puyo_synergy (デフォルト値)
            1.0,  # optimal_placement_exists (デフォルト値)
            min(max_height / 13.0, 1.0),  # height_danger_level
            min(height_variance / 10.0, 1.0),  # column_balance_score
            0.0,  # opponent_max_chain (未実装)
            0.0,  # opponent_immediate_threat (未実装)
            0.0,  # speed_advantage (未実装)
            0.0,  # chain_trigger_positions (未実装)
            0.0   # tempo_advantage (未実装)
        ])
    
    def _encode_relative_state(self, game_state: Dict[str, Any]) -> torch.Tensor:
        """対戦時の相対状態エンコード（簡易版）"""
        my_id = self.player_id
        opp_id = 'B' if my_id == 'A' else 'A'
        
        my_player = game_state['players'][my_id]
        opp_player = game_state['players'][opp_id]
        
        return torch.tensor([
            (my_player['score'] - opp_player['score']) / 100000.0,  # スコア差
            0.0,  # おじゃま差 (未実装)
            0.0,  # 連鎖数差 (未実装)
            0.0,  # 高さ比較 (未実装)
            0.0,  # 脅威レベル差 (未実装)
            0.0,  # 安定性差 (未実装)
            0.0,  # 連鎖準備度比較 (未実装)
            0.0   # 連鎖状態差 (未実装)
        ])
    
    def load_model(self, model_path: str):
        """モデルの読み込み"""
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
    
    def save_model(self, model_path: str):
        """モデルの保存"""
        try:
            torch.save(self.model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
        except Exception as e:
            print(f"Failed to save model: {e}")