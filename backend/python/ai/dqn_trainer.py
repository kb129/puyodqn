"""DQN学習システム - 自己対戦学習環境"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import time
import asyncio
import logging

from ai.dqn_player import DQNPlayer, ActionSpace, PlacementAction, PuyoDQN
from ai.weak_cpu import WeakCPU
from ai.advanced_loss import PuyoDQNLoss, LossConfig
from ai.dense_reward import DenseRewardCalculator, RewardConfig as DenseRewardConfig
from game_state.game_state import GameState
from utils.constants import Color, PUYO_COLORS


@dataclass
class Experience:
    """経験データ"""
    state: torch.Tensor              # 状態
    action_id: int                   # 選択した行動ID
    reward: float                    # 報酬
    next_state: torch.Tensor         # 次状態
    done: bool                       # 終了フラグ
    valid_actions_mask: torch.Tensor # 有効行動マスク
    next_valid_actions_mask: torch.Tensor  # 次状態の有効行動マスク


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
    win_reward: float = 10.0                # 勝利報酬
    loss_penalty: float = -10.0             # 敗北ペナルティ
    
    def total(self) -> float:
        return (self.survival_reward + self.tempo_reward + self.position_quality +
                self.chain_setup_reward + self.defense_reward + self.threat_mitigation +
                self.chain_execution_reward + self.damage_dealt + self.damage_received +
                self.win_reward + self.loss_penalty)


@dataclass  
class TrainingConfig:
    """学習設定"""
    # ハイパーパラメータ
    learning_rate: float = 1e-4
    batch_size: int = 32
    memory_size: int = 100000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 50000
    target_update: int = 2000
    gamma: float = 0.99
    
    # 密な報酬設計
    survival_reward: float = 0.01           # 毎手生存報酬
    tempo_reward_scale: float = 0.1         # テンポ優位報酬スケール
    position_quality_scale: float = 0.15    # 配置品質報酬スケール
    chain_setup_scale: float = 0.2          # 連鎖セットアップ報酬
    defense_scale: float = 0.1              # 防御報酬スケール
    chain_execution_base: float = 0.1       # 連鎖実行基本報酬
    damage_reward_scale: float = 0.02       # ダメージ報酬スケール
    win_reward: float = 10.0                # 勝利報酬
    loss_penalty: float = -10.0             # 敗北ペナルティ
    
    # 自己対戦設定
    opponent_update_interval: int = 1000    # 相手更新頻度
    game_speed_multiplier: float = 50.0     # 学習用ゲーム速度
    
    # 学習制御
    episodes: int = 100000
    save_interval: int = 5000
    eval_interval: int = 10000
    device: str = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


class PrioritizedReplayBuffer:
    """優先度付き経験再生バッファ"""
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        
    def push(self, experience: Experience, priority: float = 1.0):
        """経験の保存"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Experience], torch.Tensor, np.ndarray]:
        """優先度付きサンプリング"""
        if len(self.buffer) < batch_size:
            return [], torch.tensor([]), np.array([])
        
        prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]
        
        # 重要度重み計算
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return experiences, torch.FloatTensor(weights), indices
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """優先度の更新"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # 数値安定性のため
    
    def __len__(self):
        return len(self.buffer)


class RewardCalculator:
    """密な報酬システム"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def calculate_dense_reward(self, prev_state: Dict[str, Any], action: PlacementAction, 
                              new_state: Dict[str, Any], player_id: str) -> RewardComponents:
        """密な報酬の計算"""
        reward = RewardComponents()
        
        prev_player = prev_state['players'][player_id]
        new_player = new_state['players'][player_id]
        
        # 基本生存報酬
        reward.survival_reward = self.config.survival_reward
        
        # スコア増加による報酬
        score_diff = new_player['score'] - prev_player['score']
        if score_diff > 0:
            reward.chain_execution_reward = min(score_diff / 10000.0, 1.0) * self.config.chain_execution_base
        
        # 連鎖実行報酬
        if new_player.get('chain_count', 0) > prev_player.get('chain_count', 0):
            chain_bonus = new_player['chain_count'] * self.config.chain_setup_scale
            reward.chain_execution_reward += chain_bonus
        
        # 配置品質評価
        reward.position_quality = self._evaluate_position_quality(prev_player, new_player) * self.config.position_quality_scale
        
        # 高さペナルティ
        height_penalty = self._calculate_height_penalty(new_player['board'])
        reward.position_quality -= height_penalty * 0.1
        
        # ゲーム終了時の報酬
        if new_state.get('game_over', False):
            if new_state.get('winner') == player_id:
                reward.win_reward = self.config.win_reward
            else:
                reward.loss_penalty = self.config.loss_penalty
        
        # 対戦時の相対報酬
        if len(new_state['players']) > 1:
            opponent_id = 'B' if player_id == 'A' else 'A'
            reward.tempo_reward = self._calculate_tempo_advantage(new_player, new_state['players'][opponent_id]) * self.config.tempo_reward_scale
        
        return reward
    
    def _evaluate_position_quality(self, prev_player: Dict[str, Any], new_player: Dict[str, Any]) -> float:
        """配置品質の評価"""
        prev_board = prev_player['board']
        new_board = new_player['board']
        
        # 連結数の変化を評価
        prev_connections = self._count_connections(prev_board)
        new_connections = self._count_connections(new_board)
        
        connection_improvement = (new_connections - prev_connections) / 10.0
        
        # バランス改善を評価
        prev_balance = self._calculate_balance(prev_board)
        new_balance = self._calculate_balance(new_board)
        
        balance_improvement = (new_balance - prev_balance) * 2.0
        
        return connection_improvement + balance_improvement
    
    def _count_connections(self, board: List[List[int]]) -> int:
        """同色連結数をカウント"""
        connections = 0
        
        for row in range(13):
            for col in range(6):
                color = board[row][col]
                if color in [1, 2, 3, 4]:  # 色ぷよのみ
                    # 右隣チェック
                    if col < 5 and board[row][col + 1] == color:
                        connections += 1
                    # 下隣チェック
                    if row < 12 and board[row + 1][col] == color:
                        connections += 1
        
        return connections
    
    def _calculate_balance(self, board: List[List[int]]) -> float:
        """盤面バランスの評価"""
        heights = []
        for col in range(6):
            height = 0
            for row in range(13):
                if board[row][col] != 0:
                    height = 13 - row
                    break
            heights.append(height)
        
        avg_height = sum(heights) / 6.0
        variance = sum((h - avg_height) ** 2 for h in heights) / 6.0
        
        # 分散が小さいほど良い（最大分散で正規化）
        return 1.0 - min(variance / 30.0, 1.0)
    
    def _calculate_height_penalty(self, board: List[List[int]]) -> float:
        """高さペナルティ計算"""
        max_height = 0
        for col in range(6):
            for row in range(13):
                if board[row][col] != 0:
                    height = 13 - row
                    max_height = max(max_height, height)
                    break
        
        # 10段以上で急激にペナルティ
        if max_height >= 10:
            return (max_height - 9) ** 2 / 10.0
        return 0.0
    
    def _calculate_tempo_advantage(self, my_player: Dict[str, Any], opp_player: Dict[str, Any]) -> float:
        """テンポ優位性の計算"""
        my_score = my_player['score']
        opp_score = opp_player['score']
        
        # スコア差によるテンポ優位
        score_advantage = (my_score - opp_score) / max(my_score + opp_score, 1000)
        
        return np.tanh(score_advantage)  # -1 to 1


class DQNTrainer:
    """DQN学習エンジン"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # メインとターゲットネットワーク
        self.main_net = PuyoDQN().to(self.device)
        self.target_net = PuyoDQN().to(self.device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        
        # 最適化
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=config.learning_rate)
        
        # 経験再生
        self.memory = PrioritizedReplayBuffer(config.memory_size)
        
        # 密な報酬計算器
        dense_reward_config = DenseRewardConfig(
            survival_reward=config.survival_reward,
            tempo_reward_scale=config.tempo_reward_scale,
            position_quality_scale=config.position_quality_scale,
            chain_setup_scale=config.chain_setup_scale,
            defense_scale=config.defense_scale,
            chain_execution_base=config.chain_execution_base,
            damage_reward_scale=config.damage_reward_scale,
            win_reward=config.win_reward,
            loss_penalty=config.loss_penalty
        )
        self.reward_calculator = DenseRewardCalculator(dense_reward_config)
        
        # 高度な損失関数
        loss_config = LossConfig(
            gamma=config.gamma,
            use_double_dqn=True,
            use_huber_loss=True,
            n_step=3
        )
        self.loss_calculator = PuyoDQNLoss(loss_config)
        
        # 学習進捗
        self.step_count = 0
        self.episode_count = 0
        self.epsilon = config.epsilon_start
        self.current_performance = 0.0
        
        # プレイヤー
        self.learner = DQNPlayer("A")
        self.learner.model = self.main_net
        self.learner.epsilon = self.epsilon
        
        self.opponent = WeakCPU("B")  # 初期は無能CPU
        
        # ログ
        self.logger = logging.getLogger(__name__)
    
    def train(self):
        """メイン学習ループ"""
        self.logger.info(f"Starting training for {self.config.episodes} episodes")
        
        for episode in range(self.config.episodes):
            self.episode_count = episode
            
            # エピソード実行
            episode_reward = self._run_episode()
            
            # 学習実行
            if len(self.memory) > self.config.batch_size:
                loss = self._update_model()
            else:
                loss = 0.0
            
            # ε減衰
            self._update_epsilon()
            
            # ターゲットネット更新
            if self.step_count % self.config.target_update == 0:
                self.target_net.load_state_dict(self.main_net.state_dict())
            
            # 相手プレイヤー更新
            if episode % self.config.opponent_update_interval == 0:
                self._update_opponent()
            
            # 定期保存・評価
            if episode % self.config.save_interval == 0:
                self._save_checkpoint(episode)
            
            if episode % self.config.eval_interval == 0:
                self._evaluate_performance(episode)
            
            # 性能更新（損失関数のカリキュラム学習用）
            if episode % 100 == 0:
                # 最近のエピソード報酬を性能指標とする
                self.current_performance = episode_reward
                self.logger.info(f"Episode {episode}, Reward: {episode_reward:.4f}, Loss: {loss:.4f}, Epsilon: {self.epsilon:.4f}, Performance: {self.current_performance:.4f}")
    
    def _run_episode(self) -> float:
        """1エピソードの実行"""
        # ゲーム初期化（簡易版）
        game_state = self._create_initial_state()
        total_reward = 0.0
        
        while not game_state.get('game_over', False):
            # 現在状態の保存
            prev_state = self._copy_state(game_state)
            
            # プレイヤーA（学習エージェント）の行動
            action_a = self._get_learner_action(game_state)
            
            # プレイヤーB（相手）の行動
            action_b = self.opponent.get_action(game_state)
            
            # ゲーム状態更新（簡易版）
            game_state = self._update_game_state(game_state, action_a, action_b)
            
            # 報酬計算（密な報酬）
            reward_components = self.reward_calculator.calculate_reward(prev_state, action_a, game_state, "A")
            step_reward = reward_components.total()
            total_reward += step_reward
            
            # 性能履歴更新
            self.reward_calculator.update_performance_history(step_reward)
            
            # 経験保存
            if hasattr(self, '_last_state') and hasattr(self, '_last_action'):
                self._store_experience(self._last_state, self._last_action, step_reward, game_state)
            
            self._last_state = prev_state
            self._last_action = action_a
            
            self.step_count += 1
        
        return total_reward
    
    def _get_learner_action(self, game_state: Dict[str, Any]) -> str:
        """学習エージェントの行動取得"""
        # ε-greedy
        if random.random() < self.epsilon:
            # 探索行動（ランダム）
            actions = ["move_left", "move_right", "rotate_left", "rotate_right", "soft_drop"]
            return random.choice(actions)
        else:
            # 活用行動
            return self.learner.get_action(game_state)
    
    def _update_model(self) -> float:
        """高度な損失関数を使ったモデル更新"""
        experiences, weights, indices = self.memory.sample(self.config.batch_size)
        
        if not experiences:
            return 0.0
        
        weights = weights.to(self.device)
        
        # 高度な損失計算
        total_loss, td_errors = self.loss_calculator.compute_total_loss(
            main_net=self.main_net,
            target_net=self.target_net,
            experiences=experiences,
            weights=weights,
            device=self.device,
            current_performance=self.current_performance
        )
        
        # バックプロパゲーション
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_net.parameters(), 1.0)
        self.optimizer.step()
        
        # 優先度更新
        priorities = td_errors + 1e-6
        self.memory.update_priorities(indices, priorities)
        
        # 損失計算器の更新
        self.loss_calculator.update()
        
        return total_loss.item()
    
    def _store_experience(self, state: Dict[str, Any], action: str, reward: float, next_state: Dict[str, Any]):
        """経験の保存（改良版）"""
        try:
            state_tensor = self.learner.encode_state(state)
            next_state_tensor = self.learner.encode_state(next_state)
            
            # アクション文字列をIDに変換（簡易版）
            action_mapping = {
                "move_left": 0, "move_right": 1, "rotate_left": 2, 
                "rotate_right": 3, "soft_drop": 4, "no_action": 5
            }
            action_id = action_mapping.get(action, 5)
            
            # 有効行動マスクの生成（実際の実装では各状態で計算）
            valid_mask = torch.ones(24, dtype=torch.bool)
            next_valid_mask = torch.ones(24, dtype=torch.bool)
            
            # TODO: 実際のゲーム状態から有効行動を計算
            # current_puyo = state['players']['A'].get('current_puyo')
            # if current_puyo:
            #     valid_actions = ActionSpace.get_valid_actions(current_puyo, state['players']['A']['board'])
            #     valid_mask = ActionSpace.create_action_mask(valid_actions)
            
            exp = Experience(
                state=state_tensor,
                action_id=action_id,
                reward=reward,
                next_state=next_state_tensor,
                done=next_state.get('game_over', False),
                valid_actions_mask=valid_mask,
                next_valid_actions_mask=next_valid_mask
            )
            
            self.memory.push(exp)
        except Exception as e:
            self.logger.warning(f"Failed to store experience: {e}")
    
    def _update_epsilon(self):
        """ε値の更新"""
        self.epsilon = max(
            self.config.epsilon_end,
            self.config.epsilon_start - (self.step_count / self.config.epsilon_decay)
        )
        self.learner.epsilon = self.epsilon
    
    def _update_opponent(self):
        """相手プレイヤーの更新"""
        # 自己対戦：現在のモデルをコピー
        self.opponent = DQNPlayer("B")
        self.opponent.model.load_state_dict(self.main_net.state_dict())
        self.opponent.epsilon = 0.1  # 少しのランダム性を保持
        
        self.logger.info("Opponent updated with current model")
    
    def _create_initial_state(self) -> Dict[str, Any]:
        """初期ゲーム状態の作成（簡易版）"""
        return {
            'game_over': False,
            'players': {
                'A': {
                    'board': [[0] * 6 for _ in range(13)],
                    'current_puyo': {
                        'colors': [random.choice([1, 2, 3, 4]), random.choice([1, 2, 3, 4])],
                        'x': 2, 'y': 0, 'rotation': 0
                    },
                    'next_puyos': [],
                    'score': 0,
                    'chain_count': 0,
                    'is_chaining': False
                },
                'B': {
                    'board': [[0] * 6 for _ in range(13)],
                    'current_puyo': {
                        'colors': [random.choice([1, 2, 3, 4]), random.choice([1, 2, 3, 4])],
                        'x': 2, 'y': 0, 'rotation': 0
                    },
                    'next_puyos': [],
                    'score': 0,
                    'chain_count': 0,
                    'is_chaining': False
                }
            }
        }
    
    def _copy_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """状態のディープコピー"""
        import copy
        return copy.deepcopy(state)
    
    def _update_game_state(self, state: Dict[str, Any], action_a: str, action_b: str) -> Dict[str, Any]:
        """ゲーム状態の更新（簡易版）"""
        # 実際の実装では、ゲームエンジンを呼び出す
        # ここでは簡易的に一定確率でゲーム終了
        if random.random() < 0.01:
            state['game_over'] = True
            state['winner'] = random.choice(['A', 'B'])
        
        return state
    
    def _save_checkpoint(self, episode: int):
        """チェックポイント保存"""
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.main_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }
        
        checkpoint_path = f"models/checkpoint_episode_{episode}.pth"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _evaluate_performance(self, episode: int):
        """性能評価"""
        # 無能CPUとの対戦で評価
        eval_wins = 0
        eval_games = 10
        
        for _ in range(eval_games):
            # 評価ゲーム実行（簡易版）
            winner = random.choice(['A', 'B'])  # 実際はゲーム実行
            if winner == 'A':
                eval_wins += 1
        
        win_rate = eval_wins / eval_games
        self.logger.info(f"Episode {episode}: Win rate vs CPU: {win_rate:.2f}")


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO)
    
    # 学習実行
    config = TrainingConfig()
    trainer = DQNTrainer(config)
    trainer.train()