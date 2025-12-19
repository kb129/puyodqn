"""ゲームエンジンとAIの統合アダプター"""

import asyncio
import random
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import copy

from ai.base_player import BasePlayer
from ai.weak_cpu import WeakCPU
from ai.dqn_player import DQNPlayer
from game_state.game_state import GameState
from game_state.puyo import PuyoPair
from utils.constants import Color, PUYO_COLORS, BOARD_WIDTH, BOARD_HEIGHT


@dataclass
class GameResult:
    """ゲーム結果"""
    winner: Optional[str]
    final_scores: Dict[str, int]
    game_length: int
    total_chains: Dict[str, int]
    max_chain: Dict[str, int]


class GameEngine:
    """簡易ゲームエンジン（AI学習用）"""
    
    def __init__(self, mode: str = 'versus', speed_multiplier: float = 1.0):
        self.mode = mode
        self.speed_multiplier = speed_multiplier
        self.turn_count = 0
        
        # ぷよ生成器
        self.puyo_generator = PuyoGenerator()
        
        # ゲーム状態初期化（ぷよ生成器作成後）
        self.game_state = self._create_initial_state()
        
    def _create_initial_state(self) -> Dict[str, Any]:
        """初期ゲーム状態の作成"""
        return {
            'mode': self.mode,
            'game_over': False,
            'winner': None,
            'turn_count': 0,
            'players': {
                'A': self._create_player_state('A'),
                'B': self._create_player_state('B') if self.mode == 'versus' else None
            }
        }
    
    def _create_player_state(self, player_id: str) -> Dict[str, Any]:
        """プレイヤー状態の初期化"""
        return {
            'id': player_id,
            'board': [[Color.EMPTY] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)],
            'current_puyo': self.puyo_generator.generate_puyo_pair(),
            'next_puyos': [
                self.puyo_generator.generate_puyo_pair(),
                self.puyo_generator.generate_puyo_pair()
            ],
            'score': 0,
            'chain_count': 0,
            'is_chaining': False,
            'ojama_pending': 0,
            'total_chains': 0,
            'max_chain': 0
        }
    
    def get_state(self, player_id: Optional[str] = None) -> Dict[str, Any]:
        """現在のゲーム状態を取得"""
        if player_id:
            # 特定プレイヤー視点の状態
            state = copy.deepcopy(self.game_state)
            # 相手の情報を制限することも可能
            return state
        return copy.deepcopy(self.game_state)
    
    def is_game_over(self) -> bool:
        """ゲーム終了判定"""
        return self.game_state.get('game_over', False)
    
    async def step(self, action_a: str, action_b: Optional[str] = None) -> GameResult:
        """1ステップの実行"""
        if self.is_game_over():
            return self._create_result()
        
        # 行動実行
        self._execute_action('A', action_a)
        if action_b and self.game_state['players']['B']:
            self._execute_action('B', action_b)
        
        # ゲーム状態更新
        self._update_game_logic()
        
        # 終了判定
        self._check_game_over()
        
        self.turn_count += 1
        self.game_state['turn_count'] = self.turn_count
        
        if self.is_game_over():
            return self._create_result()
        
        return None
    
    async def step_simultaneous(self, action_a: str, action_b: str) -> Dict[str, Any]:
        """同時行動実行"""
        # 同時に行動を実行
        self._execute_action('A', action_a)
        self._execute_action('B', action_b)
        
        # ゲーム状態更新
        self._update_game_logic()
        
        # 終了判定
        self._check_game_over()
        
        return self.get_state()
    
    def _execute_action(self, player_id: str, action: str):
        """プレイヤーの行動を実行"""
        player_state = self.game_state['players'][player_id]
        current_puyo = player_state['current_puyo']
        
        if not current_puyo or player_state.get('is_chaining', False):
            return
        
        # 行動に応じて処理
        if action == "move_left":
            if self._can_move(player_state, -1):
                current_puyo['x'] -= 1
        elif action == "move_right":
            if self._can_move(player_state, 1):
                current_puyo['x'] += 1
        elif action == "rotate_left":
            if self._can_rotate(player_state, -1):
                current_puyo['rotation'] = (current_puyo['rotation'] - 1) % 4
        elif action == "rotate_right":
            if self._can_rotate(player_state, 1):
                current_puyo['rotation'] = (current_puyo['rotation'] + 1) % 4
        elif action == "soft_drop":
            self._drop_puyo(player_state)
    
    def _can_move(self, player_state: Dict[str, Any], dx: int) -> bool:
        """移動可能判定"""
        current_puyo = player_state['current_puyo']
        board = player_state['board']
        
        new_x = current_puyo['x'] + dx
        
        # 境界チェック
        if new_x < 0 or new_x >= BOARD_WIDTH:
            return False
        
        # 回転状態を考慮した位置チェック
        puyo_pair = PuyoPair(
            colors=(Color(current_puyo['colors'][0]), Color(current_puyo['colors'][1])),
            x=new_x,
            y=current_puyo['y'],
            rotation=current_puyo['rotation']
        )
        
        return puyo_pair.can_move(board, 0)  # 実際は移動後の衝突判定
    
    def _can_rotate(self, player_state: Dict[str, Any], direction: int) -> bool:
        """回転可能判定"""
        current_puyo = player_state['current_puyo']
        board = player_state['board']
        
        puyo_pair = PuyoPair(
            colors=(Color(current_puyo['colors'][0]), Color(current_puyo['colors'][1])),
            x=current_puyo['x'],
            y=current_puyo['y'],
            rotation=current_puyo['rotation']
        )
        
        return puyo_pair.can_rotate(board, direction)
    
    def _drop_puyo(self, player_state: Dict[str, Any]):
        """ぷよを落下させて固定"""
        current_puyo = player_state['current_puyo']
        board = player_state['board']
        
        # 最下位置まで落下
        puyo_pair = PuyoPair(
            colors=(Color(current_puyo['colors'][0]), Color(current_puyo['colors'][1])),
            x=current_puyo['x'],
            y=current_puyo['y'],
            rotation=current_puyo['rotation']
        )
        
        # 落下シミュレーション
        while puyo_pair.can_fall(board):
            puyo_pair.y += 1
        
        # 盤面に配置
        positions = puyo_pair.get_positions()
        for px, py, color in positions:
            if 0 <= px < BOARD_WIDTH and 0 <= py < BOARD_HEIGHT:
                board[py][px] = color
        
        # 次のぷよを生成
        self._generate_next_puyo(player_state)
    
    def _generate_next_puyo(self, player_state: Dict[str, Any]):
        """次のぷよを生成"""
        # ネクストからカレントへ
        next_puyos = player_state['next_puyos']
        if next_puyos:
            player_state['current_puyo'] = next_puyos.pop(0)
            player_state['current_puyo']['x'] = 2  # 初期位置
            player_state['current_puyo']['y'] = 0
            player_state['current_puyo']['rotation'] = 0
            
            # 新しいネクストを生成
            next_puyos.append(self.puyo_generator.generate_puyo_pair())
        else:
            player_state['current_puyo'] = self.puyo_generator.generate_puyo_pair()
    
    def _update_game_logic(self):
        """ゲームロジック更新"""
        for player_id, player_state in self.game_state['players'].items():
            if player_state is None:
                continue
                
            # 連鎖判定・処理
            chain_count = self._process_chains(player_state)
            
            if chain_count > 0:
                player_state['total_chains'] += 1
                player_state['max_chain'] = max(player_state['max_chain'], chain_count)
                
                # スコア計算（簡易版）
                score_bonus = chain_count * chain_count * 100
                player_state['score'] += score_bonus
    
    def _process_chains(self, player_state: Dict[str, Any]) -> int:
        """連鎖処理"""
        board = player_state['board']
        chain_count = 0
        
        while True:
            # 4個以上の連結を探索
            eliminated = self._find_and_eliminate_groups(board)
            
            if not eliminated:
                break
                
            chain_count += 1
            
            # ぷよ落下処理
            self._apply_gravity(board)
        
        return chain_count
    
    def _find_and_eliminate_groups(self, board: List[List[Color]]) -> bool:
        """4個以上の同色グループを消去"""
        visited = [[False] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        eliminated = False
        
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                color = board[row][col]
                
                if (not visited[row][col] and 
                    color in [Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW]):
                    
                    group = []
                    self._dfs_group(board, visited, row, col, color, group)
                    
                    if len(group) >= 4:
                        # グループ消去
                        for r, c in group:
                            board[r][c] = Color.EMPTY
                        eliminated = True
        
        return eliminated
    
    def _dfs_group(self, board: List[List[Color]], visited: List[List[bool]], 
                   row: int, col: int, color: Color, group: List[Tuple[int, int]]):
        """深さ優先探索でグループを探索"""
        if (row < 0 or row >= BOARD_HEIGHT or 
            col < 0 or col >= BOARD_WIDTH or
            visited[row][col] or board[row][col] != color):
            return
        
        visited[row][col] = True
        group.append((row, col))
        
        # 4方向探索
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dr, dc in directions:
            self._dfs_group(board, visited, row + dr, col + dc, color, group)
    
    def _apply_gravity(self, board: List[List[Color]]):
        """重力適用（ぷよを下に落とす）"""
        for col in range(BOARD_WIDTH):
            # 各列で上から下へスキャン
            write_row = BOARD_HEIGHT - 1
            
            for read_row in range(BOARD_HEIGHT - 1, -1, -1):
                if board[read_row][col] != Color.EMPTY:
                    board[write_row][col] = board[read_row][col]
                    if write_row != read_row:
                        board[read_row][col] = Color.EMPTY
                    write_row -= 1
    
    def _check_game_over(self):
        """ゲーム終了判定"""
        # ゲームオーバー条件（簡易版）
        for player_id, player_state in self.game_state['players'].items():
            if player_state is None:
                continue
                
            # 盤面上部にぷよがある場合
            board = player_state['board']
            if any(board[0][col] != Color.EMPTY for col in range(BOARD_WIDTH)):
                self.game_state['game_over'] = True
                # 勝者決定（相手がいる場合）
                if len([p for p in self.game_state['players'].values() if p is not None]) > 1:
                    self.game_state['winner'] = 'A' if player_id == 'B' else 'B'
                return
        
        # ターン数上限
        if self.turn_count > 10000:  # 無限ループ防止
            self.game_state['game_over'] = True
    
    def _create_result(self) -> GameResult:
        """ゲーム結果の作成"""
        scores = {}
        chains = {}
        max_chains = {}
        
        for player_id, player_state in self.game_state['players'].items():
            if player_state:
                scores[player_id] = player_state['score']
                chains[player_id] = player_state['total_chains']
                max_chains[player_id] = player_state['max_chain']
        
        return GameResult(
            winner=self.game_state.get('winner'),
            final_scores=scores,
            game_length=self.turn_count,
            total_chains=chains,
            max_chain=max_chains
        )


class PuyoGenerator:
    """ぷよ生成器"""
    
    def __init__(self, seed: Optional[int] = None):
        if seed:
            random.seed(seed)
    
    def generate_puyo_pair(self) -> Dict[str, Any]:
        """ぷよペアを生成"""
        colors = [
            random.choice([Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW]),
            random.choice([Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW])
        ]
        
        return {
            'colors': colors,
            'x': 2,  # 初期位置
            'y': 0,
            'rotation': 0
        }


class AIGameRunner:
    """AI対戦実行器"""
    
    def __init__(self):
        self.logger = None
    
    async def run_ai_vs_ai(self, player_a: BasePlayer, player_b: BasePlayer, 
                          game_count: int = 1, speed_multiplier: float = 1.0) -> List[GameResult]:
        """AI同士の対戦実行"""
        results = []
        
        for game_idx in range(game_count):
            # ゲームエンジン初期化
            engine = GameEngine(mode='versus', speed_multiplier=speed_multiplier)
            
            # プレイヤー初期化
            player_a.on_game_start()
            player_b.on_game_start()
            
            result = await self._run_single_game(engine, player_a, player_b)
            results.append(result)
            
            # ゲーム終了処理
            player_a.on_game_end({'result': result})
            player_b.on_game_end({'result': result})
        
        return results
    
    async def _run_single_game(self, engine: GameEngine, player_a: BasePlayer, player_b: BasePlayer) -> GameResult:
        """1ゲームの実行"""
        max_steps = 10000  # 無限ループ防止
        step_count = 0
        
        while not engine.is_game_over() and step_count < max_steps:
            game_state = engine.get_state()
            
            # 各プレイヤーの行動取得
            action_a = player_a.get_action(game_state)
            action_b = player_b.get_action(game_state)
            
            # 1ステップ実行
            await engine.step_simultaneous(action_a, action_b)
            
            step_count += 1
            
            # 高速化のための簡易ウェイト
            if engine.speed_multiplier < 10:
                await asyncio.sleep(0.01 / engine.speed_multiplier)
        
        return engine._create_result()


# 使用例
async def example_usage():
    """使用例"""
    # プレイヤー作成
    weak_cpu = WeakCPU("A")
    dqn_player = DQNPlayer("B")
    
    # 対戦実行
    runner = AIGameRunner()
    results = await runner.run_ai_vs_ai(weak_cpu, dqn_player, game_count=10, speed_multiplier=10.0)
    
    # 結果分析
    wins_a = sum(1 for r in results if r.winner == 'A')
    wins_b = sum(1 for r in results if r.winner == 'B')
    draws = sum(1 for r in results if r.winner is None)
    
    print(f"Results: A:{wins_a}, B:{wins_b}, Draw:{draws}")
    
    avg_game_length = sum(r.game_length for r in results) / len(results)
    print(f"Average game length: {avg_game_length:.1f} turns")


if __name__ == "__main__":
    asyncio.run(example_usage())