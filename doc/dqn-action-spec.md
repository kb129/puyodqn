# DQN行動空間仕様書 - 動的マスキング方式

## 概要

キー入力ベースではなく配置可能パターンベースの行動空間を採用。
動的マスキング技術により可変長行動空間を固定サイズネットワークで処理。

## 行動空間設計

### 基本構造
```python
# 行動ID定義
action_id = column * 4 + rotation
# column: 0-5 (6列)
# rotation: 0-3 (4回転)
# 合計: 24通りの行動
```

### 行動エンコーディング表
| Action ID | Column | Rotation | 説明 |
|-----------|---------|----------|------|
| 0-3 | 0 | 0-3 | 0列目の4回転 |
| 4-7 | 1 | 0-3 | 1列目の4回転 |
| 8-11 | 2 | 0-3 | 2列目の4回転 |
| 12-15 | 3 | 0-3 | 3列目の4回転 |
| 16-19 | 4 | 0-3 | 4列目の4回転 |
| 20-23 | 5 | 0-3 | 5列目の4回転 |

### 回転状態定義
```python
# rotation = 0: 縦向き（colors[0]が上、colors[1]が下）
# rotation = 1: 右向き（colors[0]が右、colors[1]が左）
# rotation = 2: 逆縦（colors[0]が下、colors[1]が上）
# rotation = 3: 左向き（colors[0]が左、colors[1]が右）
```

## 実装クラス設計

### 1. PlacementAction
```python
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
```

### 2. ActionSpace
```python
class ActionSpace:
    """動的行動空間管理"""
    
    MAX_ACTIONS = 24  # 固定サイズ
    
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
```

### 3. DQNPlayer (動的マスキング版)
```python
class DQNPlayer(BasePlayer):
    """動的マスキング型DQNプレイヤー"""
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PuyoDQN(input_size=422, output_size=ActionSpace.MAX_ACTIONS)
        self.epsilon = 0.1
        
        if model_path:
            self.load_model(model_path)
    
    def get_action(self, game_state: Dict[str, Any]) -> PlacementAction:
        """行動選択"""
        player_state = game_state['players'][self.player_id]
        puyo_pair = self._parse_puyo_pair(player_state['current_puyo'])
        board = player_state['board']
        
        # 有効行動を取得
        valid_actions = ActionSpace.get_valid_actions(puyo_pair, board)
        
        if not valid_actions:
            raise GameOverException("No valid actions available")
        
        # ε-greedy選択
        if random.random() < self.epsilon:
            return random.choice(valid_actions)  # 探索
        
        # Q値計算
        state_tensor = self.encode_state(game_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze(0)
        
        # 動的マスキング適用
        action_mask = ActionSpace.create_action_mask(valid_actions).to(self.device)
        masked_q_values = q_values.clone()
        masked_q_values[~action_mask] = float('-inf')
        
        # 最適行動選択
        best_action_id = masked_q_values.argmax().item()
        
        # 対応する行動オブジェクトを返す
        for action in valid_actions:
            if action.action_id == best_action_id:
                return action
        
        # フォールバック（通常は到達しない）
        return valid_actions[0]
```

## 学習関連仕様

### Experience Replay
```python
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
```

### 学習ループ
```python
def update_model(self, batch: List[Experience]) -> float:
    """モデル更新"""
    states = torch.stack([exp.state for exp in batch])
    actions = torch.tensor([exp.action_id for exp in batch])
    rewards = torch.tensor([exp.reward for exp in batch])
    next_states = torch.stack([exp.next_state for exp in batch])
    dones = torch.tensor([exp.done for exp in batch])
    valid_masks = torch.stack([exp.valid_actions_mask for exp in batch])
    next_valid_masks = torch.stack([exp.next_valid_actions_mask for exp in batch])
    
    # 現在Q値
    current_q_values = self.model(states)
    current_q = current_q_values.gather(1, actions.unsqueeze(1))
    
    # 次状態Q値（マスキング適用）
    with torch.no_grad():
        next_q_values = self.target_model(next_states)
        next_q_values[~next_valid_masks] = float('-inf')
        next_q = next_q_values.max(1)[0].detach()
        
    # ターゲットQ値
    target_q = rewards + (self.gamma * next_q * ~dones)
    
    # 損失計算・更新
    loss = F.mse_loss(current_q.squeeze(), target_q)
    
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    
    return loss.item()
```

## パフォーマンス最適化

### 1. 事前計算キャッシュ
```python
class ActionCache:
    """行動計算結果のキャッシュ"""
    
    def __init__(self):
        self.cache = {}  # board_hash -> valid_actions
    
    def get_valid_actions(self, puyo_pair: PuyoPair, board: List[List[Color]]) -> List[PlacementAction]:
        board_hash = self._hash_board_state(board, puyo_pair)
        
        if board_hash not in self.cache:
            self.cache[board_hash] = ActionSpace.get_valid_actions(puyo_pair, board)
            
        return self.cache[board_hash]
```

### 2. 並列計算
```python
def get_valid_actions_parallel(puyo_pair: PuyoPair, board: List[List[Color]]) -> List[PlacementAction]:
    """並列処理版の有効行動計算"""
    
    def check_action(action_id):
        try:
            action = PlacementAction.from_id(action_id, puyo_pair, board)
            return action if ActionSpace._is_valid_placement(action, board) else None
        except:
            return None
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(check_action, i) for i in range(ActionSpace.MAX_ACTIONS)]
        results = [f.result() for f in futures]
    
    return [action for action in results if action is not None]
```

## エラーハンドリング

### 例外クラス
```python
class GameOverException(Exception):
    """有効行動が存在しない場合の例外"""
    pass

class InvalidActionException(Exception):
    """無効な行動を実行しようとした場合の例外"""
    pass
```

### ロバスト性対応
```python
def safe_get_action(self, game_state: Dict[str, Any]) -> Optional[PlacementAction]:
    """安全な行動取得（例外処理付き）"""
    try:
        return self.get_action(game_state)
    except GameOverException:
        return None  # ゲーム終了として処理
    except Exception as e:
        logger.error(f"Action selection failed: {e}")
        # フォールバック：ランダム有効行動
        valid_actions = ActionSpace.get_valid_actions(puyo_pair, board)
        return random.choice(valid_actions) if valid_actions else None
```

## 拡張状態表現仕様

### 戦術的特徴量の追加
```python
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

def extract_tactical_features(game_state: Dict[str, Any], player_id: str) -> TacticalFeatures:
    """戦術的特徴量を抽出"""
    player = game_state['players'][player_id]
    board = player['board']
    
    # 連鎖分析
    max_chain = analyze_max_chain_potential(board)
    immediate_chain = analyze_immediate_chain(board, player['current_puyo'])
    
    # ネクスト分析
    next_synergy = calculate_next_synergy(board, player['next_puyos'])
    
    # 対戦分析
    opponent_features = None
    if len(game_state['players']) > 1:
        opponent_id = 'B' if player_id == 'A' else 'A'
        opponent = game_state['players'][opponent_id]
        opponent_features = analyze_opponent_threat(opponent['board'], opponent['current_puyo'])
    
    return TacticalFeatures(
        max_chain_potential=max_chain,
        immediate_chain_count=immediate_chain,
        next_puyo_synergy=next_synergy[0],
        next2_puyo_synergy=next_synergy[1],
        height_danger_level=calculate_height_danger(board),
        column_balance_score=calculate_balance_score(board),
        opponent_max_chain=opponent_features['max_chain'] if opponent_features else 0,
        opponent_immediate_threat=opponent_features['immediate_threat'] if opponent_features else 0,
        speed_advantage=calculate_speed_advantage(player, opponent) if opponent_features else 0.0,
        # その他の計算...
    )
```

### 強化された状態エンコーディング
```python
def encode_state_enhanced(self, game_state: Dict[str, Any]) -> torch.Tensor:
    """
    拡張状態表現のエンコーディング
    
    構成要素:
    - 盤面状態: 6x13x5 (390次元) - 各セルの色one-hot
    - 現在ぷよ: 2x5 (10次元) - 2個のぷよone-hot  
    - ネクスト: 4x5 (20次元) - next1, next2のone-hot
    - 基本情報: 4次元 - [score, ojama_pending, chain_count, is_chaining]
    - 戦術特徴: 12次元 - TacticalFeaturesから抽出
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
        player_state['ojama_pending'] / 50.0,    # おじゃま正規化  
        player_state['chain_count'] / 10.0,      # 連鎖数正規化
        float(player_state['is_chaining'])       # 連鎖中フラグ
    ])
    
    # 戦術特徴抽出 (12次元)
    tactical = extract_tactical_features(game_state, self.player_id)
    tactical_tensor = torch.tensor([
        tactical.max_chain_potential / 15.0,     # 最大連鎖可能数
        tactical.immediate_chain_count / 10.0,   # 即座連鎖数
        tactical.next_puyo_synergy,              # ネクスト相性
        tactical.next2_puyo_synergy,             # ネクスト2相性
        float(tactical.optimal_placement_exists), # 理想配置可能
        tactical.height_danger_level,            # 高さ危険度
        tactical.column_balance_score,           # バランススコア
        tactical.opponent_max_chain / 15.0,      # 相手最大連鎖
        tactical.opponent_immediate_threat / 10.0, # 相手即座脅威
        tactical.speed_advantage,                # 速度優位
        len(tactical.chain_trigger_positions) / 10.0, # 連鎖発動点数
        self._calculate_tempo_advantage(game_state) # テンポ優位
    ])
    
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

def _encode_relative_state(self, game_state: Dict[str, Any]) -> torch.Tensor:
    """対戦時の相対状態エンコード"""
    my_id = self.player_id
    opp_id = 'B' if my_id == 'A' else 'A'
    
    my_player = game_state['players'][my_id]
    opp_player = game_state['players'][opp_id]
    
    return torch.tensor([
        (my_player['score'] - opp_player['score']) / 100000.0,      # スコア差
        (opp_player['ojama_pending'] - my_player['ojama_pending']) / 50.0, # おじゃま差
        (my_player['chain_count'] - opp_player['chain_count']) / 10.0,      # 連鎖数差
        self._height_comparison(my_player['board'], opp_player['board']),    # 高さ比較
        self._threat_level_diff(my_player, opp_player),                     # 脅威レベル差
        self._board_stability_diff(my_player['board'], opp_player['board']), # 安定性差
        self._chain_setup_comparison(my_player, opp_player),                # 連鎖準備度比較
        float(my_player['is_chaining']) - float(opp_player['is_chaining'])  # 連鎖状態差
    ])
```

### 連鎖分析アルゴリズム
```python
def analyze_max_chain_potential(board: List[List[Color]]) -> int:
    """現在盤面から可能な最大連鎖数を分析"""
    max_chain = 0
    
    # 全可能配置パターンを試行
    for test_colors in itertools.product(PUYO_COLORS, repeat=2):
        for col in range(6):
            for rot in range(4):
                # 仮想配置
                test_board = copy.deepcopy(board)
                if place_virtual_puyo(test_board, col, rot, test_colors):
                    chain_count = simulate_chain_cascade(test_board)
                    max_chain = max(max_chain, chain_count)
    
    return max_chain

def analyze_immediate_chain(board: List[List[Color]], current_puyo: Dict) -> int:
    """現在のぷよで即座に発動可能な連鎖数"""
    if not current_puyo:
        return 0
    
    colors = (current_puyo['colors'][0], current_puyo['colors'][1])
    max_immediate = 0
    
    for col in range(6):
        for rot in range(4):
            test_board = copy.deepcopy(board)
            if place_virtual_puyo(test_board, col, rot, colors):
                chain_count = simulate_chain_cascade(test_board)
                max_immediate = max(max_immediate, chain_count)
    
    return max_immediate

def calculate_next_synergy(board: List[List[Color]], next_puyos: List[Dict]) -> Tuple[float, float]:
    """ネクストぷよとの相性度計算"""
    
    def calc_synergy(puyo_colors):
        synergy_score = 0.0
        
        # 現在盤面との色マッチング度
        for color in puyo_colors:
            color_count = count_color_on_board(board, color)
            # 既存の色と合わせやすいほど高スコア
            synergy_score += min(color_count / 20.0, 1.0)
        
        # 連鎖セットアップ可能性
        setup_potential = analyze_chain_setup_potential(board, puyo_colors)
        synergy_score += setup_potential
        
        return min(synergy_score / 2.0, 1.0)
    
    next1_synergy = calc_synergy(next_puyos[0]['colors']) if next_puyos else 0.0
    next2_synergy = calc_synergy(next_puyos[1]['colors']) if len(next_puyos) > 1 else 0.0
    
    return next1_synergy, next2_synergy
```

## 評価指標

### 行動空間効率性
```python
@dataclass
class ActionSpaceMetrics:
    """行動空間の効率性評価"""
    
    total_positions_evaluated: int = 0
    valid_positions_found: int = 0
    average_valid_actions_per_turn: float = 0.0
    action_computation_time_ms: float = 0.0
    
    def efficiency_ratio(self) -> float:
        """有効行動の割合"""
        return self.valid_positions_found / max(1, self.total_positions_evaluated)
```

この拡張された状態表現で戦術的な判断能力が大幅に向上すると期待されます。次は報酬設計を詰めますか？