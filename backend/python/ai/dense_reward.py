"""
密な報酬システムの実装
- 毎手でリッチなフィードバック
- 戦術的特徴量に基づく報酬
- 動的難易度調整
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import copy

from utils.constants import Color


@dataclass
class RewardComponents:
    """密な報酬の構成要素"""
    
    # 即座報酬（毎手）
    survival_reward: float = 0.01           # 生存報酬
    tempo_reward: float = 0.0               # テンポ優位報酬
    position_quality: float = 0.0           # 配置品質報酬
    
    # 戦術報酬
    chain_setup_reward: float = 0.0         # 連鎖セットアップ報酬
    defense_reward: float = 0.0             # 防御行動報酬
    threat_mitigation: float = 0.0          # 脅威軽減報酬
    
    # 結果報酬
    chain_execution_reward: float = 0.0     # 連鎖実行報酬
    damage_dealt: float = 0.0               # 与ダメージ報酬
    damage_received: float = 0.0            # 被ダメージペナルティ
    
    # 終了報酬
    win_reward: float = 0.0                 # 勝利報酬
    loss_penalty: float = 0.0               # 敗北ペナルティ
    
    def total(self) -> float:
        """総報酬計算"""
        return (
            self.survival_reward + self.tempo_reward + self.position_quality +
            self.chain_setup_reward + self.defense_reward + self.threat_mitigation +
            self.chain_execution_reward + self.damage_dealt + self.damage_received +
            self.win_reward + self.loss_penalty
        )
    
    def to_dict(self) -> Dict[str, float]:
        """辞書形式で返す（ログ用）"""
        return {
            'survival': self.survival_reward,
            'tempo': self.tempo_reward,
            'position': self.position_quality,
            'chain_setup': self.chain_setup_reward,
            'defense': self.defense_reward,
            'threat': self.threat_mitigation,
            'chain_exec': self.chain_execution_reward,
            'damage_dealt': self.damage_dealt,
            'damage_received': self.damage_received,
            'win': self.win_reward,
            'loss': self.loss_penalty,
            'total': self.total()
        }


@dataclass
class RewardConfig:
    """報酬計算設定"""
    
    # 基本報酬スケール
    survival_reward: float = 0.01           # 毎手生存報酬
    tempo_reward_scale: float = 0.1         # テンポ優位報酬スケール
    position_quality_scale: float = 0.15    # 配置品質報酬スケール
    chain_setup_scale: float = 0.2          # 連鎖セットアップ報酬
    defense_scale: float = 0.1              # 防御報酬スケール
    chain_execution_base: float = 0.1       # 連鎖実行基本報酬
    damage_reward_scale: float = 0.02       # ダメージ報酬スケール
    win_reward: float = 10.0                # 勝利報酬
    loss_penalty: float = -10.0             # 敗北ペナルティ
    
    # 高さペナルティ
    height_penalty_threshold: int = 10      # 高さペナルティ閾値
    height_penalty_scale: float = -0.1      # 高さペナルティスケール
    
    # 連鎖報酬調整
    chain_multiplier: float = 1.5           # 連鎖数乗数
    max_chain_reward: float = 2.0           # 最大連鎖報酬
    
    # 動的調整
    adaptive_scaling: bool = True           # 動的スケール調整
    performance_window: int = 1000          # 性能評価ウィンドウ


class BoardAnalyzer:
    """盤面分析器"""
    
    @staticmethod
    def calculate_height_distribution(board: List[List[int]]) -> Dict[str, float]:
        """高さ分布の計算"""
        heights = []
        for col in range(6):
            height = 0
            for row in range(13):
                if board[row][col] != 0:
                    height = 13 - row
                    break
            heights.append(height)
        
        return {
            'max_height': max(heights),
            'min_height': min(heights),
            'avg_height': sum(heights) / 6.0,
            'variance': np.var(heights),
            'heights': heights
        }
    
    @staticmethod
    def count_color_connections(board: List[List[int]]) -> Dict[str, int]:
        """色別の連結数をカウント"""
        connections = {'total': 0, 'by_color': {1: 0, 2: 0, 3: 0, 4: 0}}
        
        for row in range(13):
            for col in range(6):
                color = board[row][col]
                if color in [1, 2, 3, 4]:
                    # 右隣チェック
                    if col < 5 and board[row][col + 1] == color:
                        connections['total'] += 1
                        connections['by_color'][color] += 1
                    # 下隣チェック
                    if row < 12 and board[row + 1][col] == color:
                        connections['total'] += 1
                        connections['by_color'][color] += 1
        
        return connections
    
    @staticmethod
    def analyze_chain_potential(board: List[List[int]]) -> Dict[str, float]:
        """連鎖可能性の分析"""
        # 簡易版：4個以上の同色グループを探す
        visited = [[False] * 6 for _ in range(13)]
        groups = []
        
        def dfs(row: int, col: int, color: int, group: List[Tuple[int, int]]):
            if (row < 0 or row >= 13 or col < 0 or col >= 6 or 
                visited[row][col] or board[row][col] != color):
                return
            
            visited[row][col] = True
            group.append((row, col))
            
            # 4方向探索
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                dfs(row + dr, col + dc, color, group)
        
        for row in range(13):
            for col in range(6):
                color = board[row][col]
                if color in [1, 2, 3, 4] and not visited[row][col]:
                    group = []
                    dfs(row, col, color, group)
                    if len(group) >= 4:
                        groups.append((color, len(group)))
        
        potential_chain_length = len(groups)
        total_puyos_in_groups = sum(size for _, size in groups)
        
        return {
            'potential_chains': potential_chain_length,
            'total_group_puyos': total_puyos_in_groups,
            'largest_group': max((size for _, size in groups), default=0),
            'colors_ready': len(set(color for color, _ in groups))
        }
    
    @staticmethod
    def evaluate_balance_score(board: List[List[int]]) -> float:
        """盤面バランスの評価"""
        height_info = BoardAnalyzer.calculate_height_distribution(board)
        
        # 高さの分散が小さいほど良い
        variance_penalty = height_info['variance'] / 30.0  # 正規化
        
        # 極端な高さペナルティ
        max_height_penalty = 0.0
        if height_info['max_height'] >= 10:
            max_height_penalty = (height_info['max_height'] - 9) ** 2 / 10.0
        
        # バランススコア（0-1, 高いほど良い）
        balance_score = max(0.0, 1.0 - variance_penalty - max_height_penalty)
        
        return balance_score


class DenseRewardCalculator:
    """密な報酬計算器"""
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.performance_history = []
        self.adaptive_multiplier = 1.0
        
    def calculate_reward(self, 
                        prev_state: Dict[str, Any], 
                        action_taken: str,
                        new_state: Dict[str, Any],
                        player_id: str) -> RewardComponents:
        """1手の総合報酬を計算"""
        
        components = RewardComponents()
        
        # 基本生存報酬
        if not new_state.get('game_over', False):
            components.survival_reward = self.config.survival_reward
        
        # 各種報酬成分を計算
        components.tempo_reward = self._calculate_tempo_reward(prev_state, new_state, player_id)
        components.position_quality = self._calculate_position_quality_reward(prev_state, new_state, player_id)
        components.chain_setup_reward = self._calculate_chain_setup_reward(prev_state, new_state, player_id)
        components.defense_reward = self._calculate_defense_reward(prev_state, new_state, player_id)
        components.threat_mitigation = self._calculate_threat_mitigation_reward(prev_state, new_state, player_id)
        components.chain_execution_reward = self._calculate_chain_execution_reward(new_state, player_id)
        components.damage_dealt, components.damage_received = self._calculate_damage_rewards(prev_state, new_state, player_id)
        
        # 終了報酬
        if new_state.get('game_over', False):
            winner = new_state.get('winner')
            if winner == player_id:
                components.win_reward = self.config.win_reward
            elif winner is not None:  # 相手が勝利
                components.loss_penalty = self.config.loss_penalty
        
        # 動的スケール調整
        if self.config.adaptive_scaling:
            self._apply_adaptive_scaling(components)
        
        return components
    
    def _calculate_tempo_reward(self, prev_state: Dict, new_state: Dict, player_id: str) -> float:
        """テンポ優位性による報酬"""
        if len(new_state['players']) == 1:
            return 0.0  # シングルプレイでは無効
        
        my_player = new_state['players'][player_id]
        opp_id = 'B' if player_id == 'A' else 'A'
        opp_player = new_state['players'].get(opp_id)
        
        if not opp_player:
            return 0.0
        
        # 高さ差による優位性
        my_heights = BoardAnalyzer.calculate_height_distribution(my_player['board'])
        opp_heights = BoardAnalyzer.calculate_height_distribution(opp_player['board'])
        
        height_advantage = (opp_heights['avg_height'] - my_heights['avg_height']) / 12.0
        
        # 連鎖準備度差
        my_chain_potential = BoardAnalyzer.analyze_chain_potential(my_player['board'])
        opp_chain_potential = BoardAnalyzer.analyze_chain_potential(opp_player['board'])
        
        setup_advantage = (
            my_chain_potential['potential_chains'] - opp_chain_potential['potential_chains']
        ) / 10.0
        
        # テンポ報酬
        tempo_score = (height_advantage * 0.6 + setup_advantage * 0.4) * self.config.tempo_reward_scale
        return np.clip(tempo_score, -0.1, 0.1)
    
    def _calculate_position_quality_reward(self, prev_state: Dict, new_state: Dict, player_id: str) -> float:
        """配置の品質による報酬"""
        prev_player = prev_state['players'][player_id]
        new_player = new_state['players'][player_id]
        
        # 前後での改善を評価
        prev_connections = BoardAnalyzer.count_color_connections(prev_player['board'])
        new_connections = BoardAnalyzer.count_color_connections(new_player['board'])
        
        connection_improvement = (new_connections['total'] - prev_connections['total']) / 20.0
        
        # バランス改善を評価
        prev_balance = BoardAnalyzer.evaluate_balance_score(prev_player['board'])
        new_balance = BoardAnalyzer.evaluate_balance_score(new_player['board'])
        
        balance_improvement = (new_balance - prev_balance) * 2.0
        
        quality_score = (connection_improvement + balance_improvement) * self.config.position_quality_scale
        return np.clip(quality_score, -0.15, 0.15)
    
    def _calculate_chain_setup_reward(self, prev_state: Dict, new_state: Dict, player_id: str) -> float:
        """連鎖セットアップ進展による報酬"""
        prev_player = prev_state['players'][player_id]
        new_player = new_state['players'][player_id]
        
        # 連鎖可能性の変化
        prev_potential = BoardAnalyzer.analyze_chain_potential(prev_player['board'])
        new_potential = BoardAnalyzer.analyze_chain_potential(new_player['board'])
        
        chain_improvement = (
            new_potential['potential_chains'] - prev_potential['potential_chains']
        )
        
        color_diversity_bonus = 0.0
        if new_potential['colors_ready'] > prev_potential['colors_ready']:
            color_diversity_bonus = 0.1
        
        setup_score = (chain_improvement * 0.1 + color_diversity_bonus) * self.config.chain_setup_scale
        return np.clip(setup_score, -0.2, 0.2)
    
    def _calculate_defense_reward(self, prev_state: Dict, new_state: Dict, player_id: str) -> float:
        """防御行動による報酬"""
        prev_player = prev_state['players'][player_id]
        new_player = new_state['players'][player_id]
        
        # 高さ危険度の改善
        prev_heights = BoardAnalyzer.calculate_height_distribution(prev_player['board'])
        new_heights = BoardAnalyzer.calculate_height_distribution(new_player['board'])
        
        height_danger_reduction = 0.0
        if prev_heights['max_height'] >= self.config.height_penalty_threshold:
            if new_heights['max_height'] < prev_heights['max_height']:
                height_danger_reduction = (prev_heights['max_height'] - new_heights['max_height']) * 0.1
        
        # おじゃまぷよ対応
        ojama_reduction = 0.0
        if 'ojama_pending' in prev_player and 'ojama_pending' in new_player:
            prev_ojama = prev_player.get('ojama_pending', 0)
            new_ojama = new_player.get('ojama_pending', 0)
            if new_ojama < prev_ojama:
                ojama_reduction = (prev_ojama - new_ojama) * 0.01
        
        defense_score = (height_danger_reduction + ojama_reduction) * self.config.defense_scale
        return np.clip(defense_score, -0.1, 0.1)
    
    def _calculate_threat_mitigation_reward(self, prev_state: Dict, new_state: Dict, player_id: str) -> float:
        """脅威軽減報酬"""
        # 現在は簡易実装
        return 0.0
    
    def _calculate_chain_execution_reward(self, new_state: Dict, player_id: str) -> float:
        """連鎖実行による報酬"""
        player = new_state['players'][player_id]
        
        chain_count = player.get('chain_count', 0)
        
        if chain_count >= 2:
            # 連鎖数に応じた指数的報酬
            base_reward = self.config.chain_execution_base * (self.config.chain_multiplier ** (chain_count - 1))
            return min(self.config.max_chain_reward, base_reward)
        
        return 0.0
    
    def _calculate_damage_rewards(self, prev_state: Dict, new_state: Dict, player_id: str) -> Tuple[float, float]:
        """与ダメージ・被ダメージ報酬"""
        if len(new_state['players']) == 1:
            return 0.0, 0.0
        
        opp_id = 'B' if player_id == 'A' else 'A'
        
        damage_dealt = 0.0
        damage_received = 0.0
        
        # おじゃまぷよ変化による与ダメージ・被ダメージ
        if opp_id in prev_state['players'] and opp_id in new_state['players']:
            prev_opp = prev_state['players'][opp_id]
            new_opp = new_state['players'][opp_id]
            
            if 'ojama_pending' in prev_opp and 'ojama_pending' in new_opp:
                opp_ojama_increase = new_opp['ojama_pending'] - prev_opp['ojama_pending']
                damage_dealt = opp_ojama_increase * self.config.damage_reward_scale
        
        prev_my = prev_state['players'][player_id]
        new_my = new_state['players'][player_id]
        
        if 'ojama_pending' in prev_my and 'ojama_pending' in new_my:
            my_ojama_increase = new_my['ojama_pending'] - prev_my['ojama_pending']
            damage_received = -my_ojama_increase * self.config.damage_reward_scale
        
        return damage_dealt, damage_received
    
    def _apply_adaptive_scaling(self, components: RewardComponents):
        """動的スケール調整の適用"""
        # 性能履歴に基づいてスケールを調整
        if len(self.performance_history) > self.config.performance_window:
            recent_performance = np.mean(self.performance_history[-self.config.performance_window:])
            
            # 性能が低い場合は報酬を増強
            if recent_performance < -1.0:
                self.adaptive_multiplier = min(1.5, self.adaptive_multiplier * 1.01)
            elif recent_performance > 1.0:
                self.adaptive_multiplier = max(0.5, self.adaptive_multiplier * 0.99)
        
        # 全報酬成分にスケールを適用（勝敗報酬以外）
        components.survival_reward *= self.adaptive_multiplier
        components.tempo_reward *= self.adaptive_multiplier
        components.position_quality *= self.adaptive_multiplier
        components.chain_setup_reward *= self.adaptive_multiplier
        components.defense_reward *= self.adaptive_multiplier
        components.threat_mitigation *= self.adaptive_multiplier
        components.chain_execution_reward *= self.adaptive_multiplier
        components.damage_dealt *= self.adaptive_multiplier
        components.damage_received *= self.adaptive_multiplier
    
    def update_performance_history(self, total_reward: float):
        """性能履歴の更新"""
        self.performance_history.append(total_reward)
        if len(self.performance_history) > self.config.performance_window * 2:
            # 古い履歴を削除
            self.performance_history = self.performance_history[-self.config.performance_window:]