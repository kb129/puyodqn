#!/usr/bin/env python3
"""DQNプレイヤー学習スクリプト"""

import os
import sys
import asyncio
import logging
import argparse
from datetime import datetime
from pathlib import Path

# プロジェクトルートをPATHに追加
sys.path.append(str(Path(__file__).parent.parent))

from ai.dqn_trainer import DQNTrainer, TrainingConfig
from ai.dqn_player import DQNPlayer
from ai.weak_cpu import WeakCPU
from ai.game_adapter import AIGameRunner


def setup_logging(log_dir: str = "logs"):
    """ログ設定"""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"dqn_training_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    return logging.getLogger(__name__)


async def evaluate_against_cpu(dqn_player: DQNPlayer, games: int = 100) -> dict:
    """無能CPUとの評価対戦"""
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating against WeakCPU ({games} games)...")

    cpu_opponent = WeakCPU("B")
    runner = AIGameRunner()

    # 評価時はε=0（完全活用）
    original_epsilon = dqn_player.epsilon
    dqn_player.epsilon = 0.0

    try:
        results = await runner.run_ai_vs_ai(
            dqn_player,
            cpu_opponent,
            game_count=games,
            speed_multiplier=50.0,  # 高速評価
        )

        # 統計計算
        wins = sum(1 for r in results if r.winner == "A")
        losses = sum(1 for r in results if r.winner == "B")
        draws = sum(1 for r in results if r.winner is None)

        win_rate = wins / games
        avg_score = sum(r.final_scores.get("A", 0) for r in results) / games
        avg_game_length = sum(r.game_length for r in results) / games
        avg_max_chain = sum(r.max_chain.get("A", 0) for r in results) / games

        return {
            "win_rate": win_rate,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "avg_score": avg_score,
            "avg_game_length": avg_game_length,
            "avg_max_chain": avg_max_chain,
        }

    finally:
        # ε値復元
        dqn_player.epsilon = original_epsilon


class EnhancedDQNTrainer(DQNTrainer):
    """評価機能付きDQNトレーナー"""

    def __init__(self, config: TrainingConfig, output_dir: str = "models"):
        super().__init__(config)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 評価履歴
        self.evaluation_history = []

    async def train_with_evaluation(self):
        """評価付き学習ループ"""
        self.logger.info(
            f"Starting enhanced training for {self.config.episodes} episodes"
        )

        # 初期評価
        await self._run_evaluation(0)

        for episode in range(self.config.episodes):
            self.episode_count = episode

            # エピソード実行（非同期版）
            episode_reward = await self._run_episode_async()

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
            if episode % self.config.opponent_update_interval == 0 and episode > 0:
                self._update_opponent()

            # 定期保存・評価
            if episode % self.config.save_interval == 0 and episode > 0:
                await self._save_checkpoint_async(episode)

            if episode % self.config.eval_interval == 0 and episode > 0:
                await self._run_evaluation(episode)

            # プログレス表示
            if episode % 100 == 0:
                self.logger.info(
                    f"Episode {episode:6d} | "
                    f"Reward: {episode_reward:8.4f} | "
                    f"Loss: {loss:8.4f} | "
                    f"Epsilon: {self.epsilon:6.4f} | "
                    f"Memory: {len(self.memory):6d}"
                )

    async def _run_episode_async(self) -> float:
        """非同期エピソード実行"""
        from ai.game_adapter import GameEngine

        # ゲーム初期化
        runner = AIGameRunner()
        engine = GameEngine(mode="versus", speed_multiplier=10.0)

        # 対戦実行
        result = await runner._run_single_game(engine, self.learner, self.opponent)

        # 報酬計算（簡易版）
        if result.winner == "A":
            return 10.0  # 勝利報酬
        elif result.winner == "B":
            return -10.0  # 敗北ペナルティ
        else:
            return 0.0  # 引き分け

    async def _run_evaluation(self, episode: int):
        """評価実行"""
        self.logger.info(f"Running evaluation at episode {episode}")

        # DQNプレイヤー作成（現在のモデル使用）
        eval_player = DQNPlayer("A")
        eval_player.model.load_state_dict(self.main_net.state_dict())

        # 評価実行
        eval_results = await evaluate_against_cpu(eval_player, games=50)

        # 結果記録
        eval_data = {"episode": episode, "timestamp": datetime.now(), **eval_results}
        self.evaluation_history.append(eval_data)

        # 結果表示
        self.logger.info(
            f"Evaluation Results (Episode {episode}): "
            f"Win Rate: {eval_results['win_rate']:.3f} | "
            f"Avg Score: {eval_results['avg_score']:.0f} | "
            f"Avg Chain: {eval_results['avg_max_chain']:.1f}"
        )

        # ベストモデル保存
        if eval_results["win_rate"] > 0.7:  # 70%以上の勝率
            best_model_path = os.path.join(
                self.output_dir,
                f"best_model_ep{episode}_wr{eval_results['win_rate']:.3f}.pth",
            )
            eval_player.save_model(best_model_path)
            self.logger.info(f"New best model saved: {best_model_path}")

    async def _save_checkpoint_async(self, episode: int):
        """非同期チェックポイント保存"""
        checkpoint = {
            "episode": episode,
            "model_state_dict": self.main_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self.step_count,
            "evaluation_history": self.evaluation_history,
            "config": self.config,
        }

        checkpoint_path = os.path.join(
            self.output_dir, f"checkpoint_episode_{episode}.pth"
        )

        # 非同期保存
        import torch

        await asyncio.get_event_loop().run_in_executor(
            None, torch.save, checkpoint, checkpoint_path
        )

        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

        # 古いチェックポイント削除（最新5個のみ保持）
        await self._cleanup_old_checkpoints()

    async def _cleanup_old_checkpoints(self):
        """古いチェックポイントの削除"""
        checkpoint_files = []
        for file in os.listdir(self.output_dir):
            if file.startswith("checkpoint_episode_") and file.endswith(".pth"):
                checkpoint_files.append(file)

        # エピソード番号でソート
        checkpoint_files.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))

        # 古いファイルを削除（最新5個以外）
        for old_file in checkpoint_files[:-5]:
            old_path = os.path.join(self.output_dir, old_file)
            os.remove(old_path)
            self.logger.debug(f"Removed old checkpoint: {old_path}")


def create_training_config(args, device: str) -> TrainingConfig:
    """コマンドライン引数から学習設定を作成"""
    config = TrainingConfig()

    # コマンドライン引数で上書き
    if args.episodes:
        config.episodes = args.episodes
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.memory_size:
        config.memory_size = args.memory_size
    if args.epsilon_decay:
        config.epsilon_decay = args.epsilon_decay

    # デバイス設定
    config.device = device
    
    # MPSの場合はバッチサイズを最適化
    if device == "mps":
        if not args.batch_size:  # コマンドライン指定がない場合
            config.batch_size = 128  # MPSでは128以上が効率的
        config.game_speed_multiplier = 100.0  # 高速化
    elif device == "cpu":
        # CPUでは小さなバッチサイズが効率的
        if not args.batch_size:
            config.batch_size = 32

    return config


async def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="DQN Puyo Player Training")

    # 学習パラメータ
    parser.add_argument(
        "--episodes", type=int, default=100000, help="Training episodes"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--memory-size", type=int, default=100000, help="Replay buffer size"
    )
    parser.add_argument(
        "--epsilon-decay", type=int, default=50000, help="Epsilon decay steps"
    )

    # 出力設定
    parser.add_argument(
        "--output-dir", type=str, default="models", help="Output directory"
    )
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")

    # 実行設定
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")

    # 評価モード
    parser.add_argument("--eval-only", action="store_true", help="Evaluation mode only")
    parser.add_argument("--eval-model", type=str, help="Model path for evaluation")
    parser.add_argument(
        "--eval-games", type=int, default=100, help="Number of evaluation games"
    )

    args = parser.parse_args()

    # ログ設定
    logger = setup_logging(args.log_dir)
    logger.info("Starting DQN training script")
    logger.info(f"Arguments: {vars(args)}")

    # デバイス情報表示
    import torch

    # デバイス選択（MPS > CUDA > CPU）
    if args.cpu:
        device = "cpu"
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    logger.info(f"Using device: {device}")
    if device == "mps":
        logger.info("Using Apple Metal Performance Shaders (MPS) acceleration")
    elif device == "cuda":
        logger.info("Using NVIDIA CUDA acceleration")
    else:
        logger.info("Using CPU (no hardware acceleration)")

    if args.eval_only:
        # 評価モードのみ
        if not args.eval_model:
            logger.error("--eval-model is required for evaluation mode")
            return

        logger.info(f"Evaluation mode: {args.eval_model}")

        # モデル読み込み
        player = DQNPlayer("A")
        player.load_model(args.eval_model)

        # 評価実行
        results = await evaluate_against_cpu(player, args.eval_games)

        # 結果表示
        print("\n=== Evaluation Results ===")
        print(f"Win Rate: {results['win_rate']:.1%}")
        print(
            f"Wins/Losses/Draws: {results['wins']}/{results['losses']}/{results['draws']}"
        )
        print(f"Average Score: {results['avg_score']:.1f}")
        print(f"Average Game Length: {results['avg_game_length']:.1f} turns")
        print(f"Average Max Chain: {results['avg_max_chain']:.2f}")

    else:
        # 学習モード
        config = create_training_config(args, device)
        trainer = EnhancedDQNTrainer(config, args.output_dir)

        # チェックポイントから復帰
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            # TODO: チェックポイント読み込み実装

        # 学習開始
        try:
            await trainer.train_with_evaluation()
            logger.info("Training completed successfully")
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            # 緊急保存
            emergency_path = os.path.join(args.output_dir, "emergency_checkpoint.pth")
            trainer._save_checkpoint(trainer.episode_count)
            logger.info(f"Emergency checkpoint saved: {emergency_path}")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise


if __name__ == "__main__":
    asyncio.run(main())
