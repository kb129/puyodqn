#!/usr/bin/env python3
"""PuyoDQN CLI Tool - AI管理・学習・評価ツール"""

import asyncio
import argparse
import sys
from pathlib import Path

# プロジェクトルートをPATHに追加
sys.path.append(str(Path(__file__).parent))

from ai.weak_cpu import WeakCPU
from ai.dqn_player import DQNPlayer
from ai.game_adapter import AIGameRunner


async def run_ai_battle(args):
    """AI対戦の実行"""
    print(f"🎮 AI Battle: {args.player_a} vs {args.player_b}")
    
    # プレイヤーA作成
    if args.player_a == "weak":
        player_a = WeakCPU("A")
    elif args.player_a == "dqn":
        if args.model_a:
            player_a = DQNPlayer("A", model_path=args.model_a)
        else:
            print("❌ DQN player requires --model-a parameter")
            return
        player_a.epsilon = 0.0  # 評価モード
    else:
        print(f"❌ Unknown player type: {args.player_a}")
        return
    
    # プレイヤーB作成
    if args.player_b == "weak":
        player_b = WeakCPU("B")
    elif args.player_b == "dqn":
        if args.model_b:
            player_b = DQNPlayer("B", model_path=args.model_b)
        else:
            print("❌ DQN player requires --model-b parameter")
            return
        player_b.epsilon = 0.0  # 評価モード
    else:
        print(f"❌ Unknown player type: {args.player_b}")
        return
    
    # 対戦実行
    runner = AIGameRunner()
    print(f"⚔️  Running {args.games} games...")
    
    try:
        results = await runner.run_ai_vs_ai(
            player_a, player_b, 
            game_count=args.games, 
            speed_multiplier=args.speed
        )
        
        # 結果集計
        wins_a = sum(1 for r in results if r.winner == 'A')
        wins_b = sum(1 for r in results if r.winner == 'B')
        draws = sum(1 for r in results if r.winner is None)
        
        print("\n📊 Battle Results:")
        print(f"   {args.player_a.upper()}: {wins_a} wins ({wins_a/args.games:.1%})")
        print(f"   {args.player_b.upper()}: {wins_b} wins ({wins_b/args.games:.1%})")
        print(f"   Draws: {draws} ({draws/args.games:.1%})")
        
        if results:
            avg_length = sum(r.game_length for r in results) / len(results)
            avg_score_a = sum(r.final_scores.get('A', 0) for r in results) / len(results)
            avg_score_b = sum(r.final_scores.get('B', 0) for r in results) / len(results)
            
            print(f"\n📈 Statistics:")
            print(f"   Average game length: {avg_length:.1f} turns")
            print(f"   Average scores: A={avg_score_a:.0f}, B={avg_score_b:.0f}")
            
            # 最大連鎖
            if any(r.max_chain.get('A', 0) > 0 for r in results):
                max_chain_a = max(r.max_chain.get('A', 0) for r in results)
                max_chain_b = max(r.max_chain.get('B', 0) for r in results)
                print(f"   Max chains: A={max_chain_a}, B={max_chain_b}")
        
        print("✅ Battle completed!")
        
    except Exception as e:
        print(f"❌ Battle failed: {e}")
        import traceback
        traceback.print_exc()


def list_models():
    """利用可能なモデル一覧表示"""
    import os
    import glob
    from datetime import datetime
    
    models_dir = Path("models")
    if not models_dir.exists():
        print("📂 No models directory found")
        return
    
    model_files = list(models_dir.glob("*.pth"))
    
    if not model_files:
        print("📂 No models found in models/ directory")
        return
    
    print("🤖 Available Models:")
    print("=" * 60)
    
    for model_file in sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True):
        stat = model_file.stat()
        size_mb = stat.st_size / (1024 * 1024)
        modified = datetime.fromtimestamp(stat.st_mtime)
        
        print(f"   {model_file.name}")
        print(f"     Size: {size_mb:.1f} MB")
        print(f"     Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")
        print()


def test_ai():
    """AI基本テスト"""
    print("🧪 Testing AI Components...")
    
    try:
        # WeakCPU テスト
        weak = WeakCPU("A")
        print(f"✅ WeakCPU: {weak.name}")
        
        # DQN テスト
        dqn = DQNPlayer("A")
        print(f"✅ DQNPlayer: {dqn.name}")
        
        # 簡単な状態エンコーディングテスト
        test_state = {
            'players': {
                'A': {
                    'board': [[0] * 6 for _ in range(13)],
                    'current_puyo': {'colors': [1, 2], 'x': 2, 'y': 0, 'rotation': 0},
                    'next_puyos': [{'colors': [3, 4]}, {'colors': [1, 3]}],
                    'score': 0, 'chain_count': 0, 'is_chaining': False
                }
            }
        }
        
        encoded = dqn.encode_state(test_state)
        print(f"✅ State encoding: {encoded.shape}")
        
        action = dqn.get_action(test_state)
        print(f"✅ Action generation: {action}")
        
        print("🎉 All AI tests passed!")
        
    except Exception as e:
        print(f"❌ AI test failed: {e}")
        import traceback
        traceback.print_exc()


async def run_training_demo():
    """学習のデモンストレーション"""
    print("🎓 Training Demo (Short Version)")
    print("Note: This is a simplified demo. Use train_dqn.py for full training.")
    
    try:
        from ai.dqn_trainer import DQNTrainer, TrainingConfig
        
        # 短時間学習設定
        config = TrainingConfig()
        config.episodes = 100  # 短縮
        config.save_interval = 50
        config.eval_interval = 50
        
        trainer = DQNTrainer(config)
        
        print("🏃‍♂️ Running 100 episodes...")
        # 実際の学習は時間がかかるため、ここでは設定表示のみ
        print(f"   Learning rate: {config.learning_rate}")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Memory size: {config.memory_size}")
        print(f"   Device: {config.device}")
        
        print("💡 To run full training, use:")
        print("   uv run python ai/train_dqn.py --episodes 10000")
        
    except Exception as e:
        print(f"❌ Training demo failed: {e}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="PuyoDQN CLI - AI Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # AI対戦
  python cli.py battle weak dqn --model-b models/best_model.pth --games 10
  
  # モデル一覧
  python cli.py list-models
  
  # AIテスト
  python cli.py test
  
  # 学習デモ
  python cli.py train-demo
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Battle command
    battle_parser = subparsers.add_parser('battle', help='Run AI vs AI battle')
    battle_parser.add_argument('player_a', choices=['weak', 'dqn'], help='Player A type')
    battle_parser.add_argument('player_b', choices=['weak', 'dqn'], help='Player B type')
    battle_parser.add_argument('--model-a', type=str, help='Model path for player A (if DQN)')
    battle_parser.add_argument('--model-b', type=str, help='Model path for player B (if DQN)')
    battle_parser.add_argument('--games', type=int, default=10, help='Number of games to play')
    battle_parser.add_argument('--speed', type=float, default=10.0, help='Game speed multiplier')
    
    # List models command
    subparsers.add_parser('list-models', help='List available models')
    
    # Test command
    subparsers.add_parser('test', help='Run AI component tests')
    
    # Training demo command
    subparsers.add_parser('train-demo', help='Show training demo')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # コマンド実行
    if args.command == 'battle':
        asyncio.run(run_ai_battle(args))
    elif args.command == 'list-models':
        list_models()
    elif args.command == 'test':
        test_ai()
    elif args.command == 'train-demo':
        asyncio.run(run_training_demo())


if __name__ == "__main__":
    main()