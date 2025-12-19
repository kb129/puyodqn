"""
高度なDQN損失関数の実装
- Prioritized Experience Replay対応
- Double DQN
- Dueling DQN対応
- Rainbow DQN要素統合
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass 
class LossConfig:
    """損失関数設定"""
    
    # 基本設定
    gamma: float = 0.99                     # 割引率
    clip_grad_norm: float = 1.0            # 勾配クリッピング
    
    # Double DQN
    use_double_dqn: bool = True            # Double DQN使用
    
    # Priority設定  
    priority_alpha: float = 0.6            # 優先度の重要度
    priority_beta_start: float = 0.4       # 重要度サンプリング開始値
    priority_beta_end: float = 1.0         # 重要度サンプリング終了値
    priority_beta_frames: int = 100000     # βの線形増加フレーム数
    
    # Huber Loss
    use_huber_loss: bool = True           # Huber損失使用
    huber_delta: float = 1.0              # Huber損失の閾値
    
    # Multi-step learning
    n_step: int = 3                       # N-step学習のN
    
    # Noisy Networks (将来拡張用)
    use_noisy_networks: bool = False


class AdvancedLoss:
    """高度なDQN損失関数"""
    
    def __init__(self, config: LossConfig):
        self.config = config
        self.frame_count = 0
        
    def compute_loss(self, 
                    main_net: torch.nn.Module,
                    target_net: torch.nn.Module, 
                    experiences: List,
                    weights: torch.Tensor,
                    device: torch.device) -> Tuple[torch.Tensor, np.ndarray]:
        """
        高度な損失計算
        
        Returns:
            loss: 重み付き損失
            td_errors: TD誤差（優先度更新用）
        """
        
        # バッチデータの準備
        batch_data = self._prepare_batch(experiences, device)
        
        # Q値計算
        current_q, next_q, target_q = self._compute_q_values(
            main_net, target_net, batch_data
        )
        
        # TD誤差計算
        td_errors = self._compute_td_errors(current_q, target_q)
        
        # 損失計算
        loss = self._compute_weighted_loss(td_errors, weights)
        
        return loss, td_errors.detach().cpu().numpy()
    
    def _prepare_batch(self, experiences: List, device: torch.device) -> dict:
        """バッチデータの準備"""
        states = torch.stack([exp.state for exp in experiences]).to(device)
        actions = torch.tensor([exp.action_id for exp in experiences]).long().to(device)
        rewards = torch.tensor([exp.reward for exp in experiences]).float().to(device)
        next_states = torch.stack([exp.next_state for exp in experiences]).to(device)
        dones = torch.tensor([exp.done for exp in experiences]).bool().to(device)
        
        # 有効行動マスクの処理
        valid_masks = torch.stack([exp.valid_actions_mask for exp in experiences]).to(device)
        next_valid_masks = torch.stack([exp.next_valid_actions_mask for exp in experiences]).to(device)
        
        return {
            'states': states,
            'actions': actions, 
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'valid_masks': valid_masks,
            'next_valid_masks': next_valid_masks
        }
    
    def _compute_q_values(self, main_net: torch.nn.Module, target_net: torch.nn.Module, 
                         batch_data: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Q値の計算"""
        
        # 現在のQ値
        current_q_values = main_net(batch_data['states'])
        current_q = current_q_values.gather(1, batch_data['actions'].unsqueeze(1)).squeeze(1)
        
        # 次状態のQ値計算
        with torch.no_grad():
            if self.config.use_double_dqn:
                # Double DQN: メインネットで行動選択、ターゲットネットで評価
                next_q_main = main_net(batch_data['next_states'])
                
                # 有効行動のみを考慮（MPS対応：大きな負の値を使用）
                large_negative = -1e9
                masked_q_main = next_q_main.masked_fill(~batch_data['next_valid_masks'], large_negative)
                next_actions = masked_q_main.argmax(1)
                
                next_q_target = target_net(batch_data['next_states'])
                next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # 標準DQN
                next_q_values = target_net(batch_data['next_states'])
                large_negative = -1e9
                masked_next_q = next_q_values.masked_fill(~batch_data['next_valid_masks'], large_negative)
                next_q = masked_next_q.max(1)[0]
        
        # ターゲットQ値
        target_q = batch_data['rewards'] + (self.config.gamma * next_q * ~batch_data['dones'])
        
        return current_q, next_q, target_q
    
    def _compute_td_errors(self, current_q: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
        """TD誤差の計算"""
        if self.config.use_huber_loss:
            # Huber損失（外れ値に対してロバスト）
            td_errors = F.smooth_l1_loss(current_q, target_q, reduction='none', beta=self.config.huber_delta)
        else:
            # MSE損失
            td_errors = F.mse_loss(current_q, target_q, reduction='none')
        
        return td_errors
    
    def _compute_weighted_loss(self, td_errors: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """重み付き損失の計算"""
        # 重要度サンプリングによる重み付け
        weighted_loss = (td_errors * weights).mean()
        
        return weighted_loss
    
    def get_priority_beta(self) -> float:
        """現在の重要度サンプリングβ値を取得"""
        progress = min(1.0, self.frame_count / self.config.priority_beta_frames)
        beta = self.config.priority_beta_start + progress * (
            self.config.priority_beta_end - self.config.priority_beta_start
        )
        return beta
    
    def update_frame_count(self):
        """フレームカウントを更新"""
        self.frame_count += 1


class MultiStepLoss:
    """N-step学習対応損失"""
    
    def __init__(self, n_step: int = 3, gamma: float = 0.99):
        self.n_step = n_step
        self.gamma = gamma
    
    def compute_n_step_targets(self, experiences: List) -> torch.Tensor:
        """N-stepターゲットの計算"""
        batch_size = len(experiences)
        n_step_returns = torch.zeros(batch_size)
        
        for i, exp in enumerate(experiences):
            # N-step収益の計算
            n_step_return = 0.0
            gamma_pow = 1.0
            
            for step in range(min(self.n_step, len(exp.n_step_rewards))):
                n_step_return += gamma_pow * exp.n_step_rewards[step]
                gamma_pow *= self.gamma
            
            n_step_returns[i] = n_step_return
        
        return n_step_returns


class RegularizedLoss:
    """正則化付き損失"""
    
    def __init__(self, l1_weight: float = 0.0, l2_weight: float = 1e-4):
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
    
    def compute_regularization_loss(self, model: torch.nn.Module) -> torch.Tensor:
        """正則化損失の計算"""
        l1_loss = 0.0
        l2_loss = 0.0
        
        for param in model.parameters():
            if param.requires_grad:
                l1_loss += torch.sum(torch.abs(param))
                l2_loss += torch.sum(param ** 2)
        
        return self.l1_weight * l1_loss + self.l2_weight * l2_loss


class CurriculumLoss:
    """カリキュラム学習対応損失"""
    
    def __init__(self):
        self.difficulty_level = 0
        self.performance_history = []
    
    def adapt_loss_weights(self, current_performance: float) -> dict:
        """性能に基づく損失重み調整"""
        self.performance_history.append(current_performance)
        
        # 最近の性能向上を評価
        if len(self.performance_history) > 100:
            recent_improvement = (
                np.mean(self.performance_history[-20:]) - 
                np.mean(self.performance_history[-100:-80])
            )
            
            # 性能停滞時は難易度調整
            if recent_improvement < 0.01:
                self.difficulty_level = min(self.difficulty_level + 0.1, 2.0)
            elif recent_improvement > 0.05:
                self.difficulty_level = max(self.difficulty_level - 0.1, 0.0)
        
        return {
            'exploration_weight': 1.0 + self.difficulty_level * 0.2,
            'exploitation_weight': 1.0 - self.difficulty_level * 0.1,
            'regularization_weight': 1.0 + self.difficulty_level * 0.1
        }


# 統合損失クラス
class PuyoDQNLoss:
    """ぷよぷよDQN専用の統合損失関数"""
    
    def __init__(self, config: LossConfig):
        self.config = config
        self.advanced_loss = AdvancedLoss(config)
        self.multi_step_loss = MultiStepLoss(config.n_step, config.gamma) if config.n_step > 1 else None
        self.regularized_loss = RegularizedLoss()
        self.curriculum_loss = CurriculumLoss()
        
    def compute_total_loss(self, 
                          main_net: torch.nn.Module,
                          target_net: torch.nn.Module,
                          experiences: List,
                          weights: torch.Tensor,
                          device: torch.device,
                          current_performance: float = 0.0) -> Tuple[torch.Tensor, np.ndarray]:
        """総合損失の計算"""
        
        # 基本損失
        base_loss, td_errors = self.advanced_loss.compute_loss(
            main_net, target_net, experiences, weights, device
        )
        
        # 正則化損失
        reg_loss = self.regularized_loss.compute_regularization_loss(main_net)
        
        # カリキュラム学習重み
        curriculum_weights = self.curriculum_loss.adapt_loss_weights(current_performance)
        
        # 総合損失
        total_loss = (
            base_loss * curriculum_weights['exploitation_weight'] + 
            reg_loss * curriculum_weights['regularization_weight']
        )
        
        return total_loss, td_errors
    
    def update(self):
        """フレーム更新"""
        self.advanced_loss.update_frame_count()