#!/usr/bin/env python3
"""AIプレイヤーのテストスクリプト"""

import asyncio
import sys
from pathlib import Path

# プロジェクトルートをPATHに追加
sys.path.append(str(Path(__file__).parent))

from ai.weak_cpu import WeakCPU
from ai.dqn_player import DQNPlayer
from ai.game_adapter import AIGameRunner


async def test_weak_cpu_vs_weak_cpu():
    """WeakCPU同士の対戦テスト"""
    print("Testing WeakCPU vs WeakCPU...")
    
    player_a = WeakCPU("A")
    player_b = WeakCPU("B")
    
    runner = AIGameRunner()
    results = await runner.run_ai_vs_ai(player_a, player_b, game_count=5, speed_multiplier=10.0)
    
    wins_a = sum(1 for r in results if r.winner == 'A')
    wins_b = sum(1 for r in results if r.winner == 'B')
    draws = sum(1 for r in results if r.winner is None)
    
    print(f"Results: A:{wins_a}, B:{wins_b}, Draw:{draws}")
    
    if results:
        avg_length = sum(r.game_length for r in results) / len(results)
        avg_score_a = sum(r.final_scores.get('A', 0) for r in results) / len(results)
        avg_score_b = sum(r.final_scores.get('B', 0) for r in results) / len(results)
        
        print(f"Average game length: {avg_length:.1f} turns")
        print(f"Average scores: A:{avg_score_a:.0f}, B:{avg_score_b:.0f}")


async def test_dqn_vs_weak_cpu():
    """DQN vs WeakCPUの対戦テスト（モデル未学習版）"""
    print("\nTesting DQN vs WeakCPU (untrained model)...")
    
    try:
        dqn_player = DQNPlayer("A")  # モデル未読み込み
        dqn_player.epsilon = 0.0  # 完全活用モード
        
        weak_cpu = WeakCPU("B")
        
        runner = AIGameRunner()
        results = await runner.run_ai_vs_ai(dqn_player, weak_cpu, game_count=3, speed_multiplier=10.0)
        
        wins_a = sum(1 for r in results if r.winner == 'A')
        wins_b = sum(1 for r in results if r.winner == 'B')
        draws = sum(1 for r in results if r.winner is None)
        
        print(f"Results: DQN:{wins_a}, WeakCPU:{wins_b}, Draw:{draws}")
        
        if results:
            avg_length = sum(r.game_length for r in results) / len(results)
            avg_score_a = sum(r.final_scores.get('A', 0) for r in results) / len(results)
            avg_score_b = sum(r.final_scores.get('B', 0) for r in results) / len(results)
            
            print(f"Average game length: {avg_length:.1f} turns")
            print(f"Average scores: DQN:{avg_score_a:.0f}, WeakCPU:{avg_score_b:.0f}")
        
    except Exception as e:
        print(f"DQN test failed: {e}")


def test_dqn_action_encoding():
    """DQNの行動エンコーディングテスト"""
    print("\nTesting DQN action encoding...")
    
    # 簡易ゲーム状態
    test_game_state = {
        'players': {
            'A': {
                'board': [[0] * 6 for _ in range(13)],
                'current_puyo': {
                    'colors': [1, 2],  # 赤、青
                    'x': 2, 'y': 0, 'rotation': 0
                },
                'next_puyos': [
                    {'colors': [3, 4]},  # 緑、黄
                    {'colors': [1, 3]}   # 赤、緑
                ],
                'score': 0,
                'chain_count': 0,
                'is_chaining': False
            }
        }
    }
    
    try:
        dqn_player = DQNPlayer("A")
        
        # 状態エンコーディングテスト
        state_tensor = dqn_player.encode_state(test_game_state)
        print(f"State encoding shape: {state_tensor.shape}")
        print(f"Expected shape: (444,)")
        
        # 行動取得テスト
        action = dqn_player.get_action(test_game_state)
        print(f"Action: {action}")
        
        # フォールバック行動テスト
        fallback_action = dqn_player._fallback_action(test_game_state)
        print(f"Fallback action: {fallback_action}")
        
        print("DQN action encoding test passed!")
        
    except Exception as e:
        print(f"DQN action encoding test failed: {e}")
        import traceback
        traceback.print_exc()


def test_action_space():
    """行動空間テスト"""
    print("\nTesting action space...")
    
    from ai.dqn_player import ActionSpace, PlacementAction
    from game_state.puyo import PuyoPair
    from utils.constants import Color
    
    try:
        # テスト用ぷよペア
        puyo_pair = PuyoPair(
            colors=(Color.RED, Color.BLUE),
            x=2, y=0, rotation=0
        )
        
        # 空盤面
        empty_board = [[Color.EMPTY] * 6 for _ in range(13)]
        
        # 有効行動取得
        valid_actions = ActionSpace.get_valid_actions(puyo_pair, empty_board)
        
        print(f"Valid actions count: {len(valid_actions)}")
        print(f"Expected: 24 (6 columns × 4 rotations)")
        
        # いくつかの行動を表示
        for i, action in enumerate(valid_actions[:5]):
            print(f"Action {i}: ID={action.action_id}, Column={action.column}, Rotation={action.rotation}")
        
        # 行動マスク生成
        action_mask = ActionSpace.create_action_mask(valid_actions)
        print(f"Action mask shape: {action_mask.shape}")
        print(f"True count in mask: {action_mask.sum().item()}")
        
        print("Action space test passed!")
        
    except Exception as e:
        print(f"Action space test failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """メインテスト関数"""
    print("=== AI Player Tests ===")
    
    # 基本機能テスト
    test_dqn_action_encoding()
    test_action_space()
    
    # 対戦テスト
    await test_weak_cpu_vs_weak_cpu()
    await test_dqn_vs_weak_cpu()
    
    print("\n=== All tests completed ===")


if __name__ == "__main__":
    asyncio.run(main())