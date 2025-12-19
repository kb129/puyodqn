# AI・CPU仕様書

## AI プレイヤー概要

### プレイヤータイプ
1. **無能CPU** - シンプルな深度優先思考
2. **DQNプレイヤー** - 機械学習による最適化（将来実装）

## 無能CPU仕様

### 基本思考アルゴリズム
```python
class WeakCPU:
    def get_action(self, game_state: GameState) -> str:
        """
        最も深い谷に回転なしでぷよを落とす単純AI
        """
        current_x = game_state.current_puyo.x
        target_column = self.find_deepest_column(game_state.board)
        
        if current_x < target_column:
            return "move_right"
        elif current_x > target_column:
            return "move_left"
        else:
            return "soft_drop"
    
    def find_deepest_column(self, board: List[List[int]]) -> int:
        """各列の深度を計算し、最も深い列を返す"""
        depths = []
        for col in range(6):
            depth = 0
            for row in range(12, -1, -1):  # 下から上へ
                if board[row][col] == 0:    # 空セル
                    depth += 1
                else:
                    break
            depths.append(depth)
        
        return depths.index(max(depths))
```

### 行動決定フロー
```
1. 現在のぷよペアの位置を取得
2. 全6列の深度を計算
3. 最も深い列を目標として設定
4. 現在位置から目標位置への移動方向を決定
5. 目標位置に到達していれば即座に落下
```

### 特徴・制限
- **回転なし**: ぷよペアの回転は行わない
- **連鎖無視**: 連鎖を狙わず単純に積み上げるだけ
- **即応性**: 計算時間は常に一定（< 10ms）
- **決定論的**: 同じ局面では必ず同じ行動

### パフォーマンス仕様
```python
@dataclass
class CPUPerformanceConfig:
    reaction_time_ms: int = 100      # 人間らしい反応時間
    max_thinking_time_ms: int = 50   # 最大思考時間
    randomness: float = 0.0          # ランダム性（0-1）
    look_ahead_depth: int = 0        # 先読み深度
```

## DQNプレイヤー仕様

### アーキテクチャ概要
```python
class DQNPlayer(BasePlayer):
    """深層Q学習によるぷよぷよAI - 位置ベース行動空間"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PuyoDQN(input_size=444, output_size=24)  # 6列×4回転=24行動
        self.epsilon = 0.1  # 探索率
        
        if model_path:
            self.load_model(model_path)
        
    def get_action(self, game_state: Dict[str, Any]) -> PlacementAction:
        """位置ベース行動選択（動的マスキング）"""
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
        
        # Q値計算 + 動的マスキング
        state_tensor = self.encode_state(game_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze(0)
        
        # 無効行動をマスク
        action_mask = ActionSpace.create_action_mask(valid_actions).to(self.device)
        masked_q_values = q_values.clone()
        masked_q_values[~action_mask] = float('-inf')
        
        # 最適行動選択
        best_action_id = masked_q_values.argmax().item()
        
        for action in valid_actions:
            if action.action_id == best_action_id:
                return action
        
        return valid_actions[0]  # フォールバック
```

### 拡張状態表現設計
```python
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
```

### ネットワーク構造
```python
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
            
            # グローバル平均プーリング
            nn.AdaptiveAvgPool2d((4, 3))
        )
        
        # 戦術特徴処理
        self.tactical_fc = nn.Sequential(
            nn.Linear(20, 64),  # 戦術特徴(12) + 相対情報(8) = 20
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # メイン決定ネットワーク
        self.main_fc = nn.Sequential(
            nn.Linear(128*4*3 + 64, hidden_size),  # 盤面特徴 + 戦術特徴
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
        board_out = self.board_conv(board_features).flatten(1)   # バッチ×(128×4×3)
        
        # 戦術・相対特徴処理 (20次元)
        tactical_features = x[:, 424:]  # 基本情報(4) + 戦術(12) + 相対(8) = 24次元のうち後20次元
        tactical_out = self.tactical_fc(tactical_features)
        
        # 結合して最終決定
        combined = torch.cat([board_out, tactical_out], dim=1)
        q_values = self.main_fc(combined)
        
        return q_values
```

### 学習設定
```python
@dataclass  
class TrainingConfig:
    # ハイパーパラメータ
    learning_rate: float = 1e-4
    batch_size: int = 32
    memory_size: int = 100000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 50000
    target_update: int = 2000
    
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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
```

## 位置ベース行動空間仕様

### 行動定義
```python
@dataclass
class PlacementAction:
    """配置行動クラス"""
    action_id: int       # 0-23の行動ID (column * 4 + rotation)
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
        
        return cls(action_id, column, rotation, final_positions)

class ActionSpace:
    """動的行動空間管理"""
    MAX_ACTIONS = 24  # 6列 × 4回転
    
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
                continue
        return valid_actions
    
    @staticmethod
    def create_action_mask(valid_actions: List[PlacementAction]) -> torch.Tensor:
        """有効行動マスクを生成"""
        mask = torch.zeros(ActionSpace.MAX_ACTIONS, dtype=torch.bool)
        for action in valid_actions:
            mask[action.action_id] = True
        return mask
```

## 密な報酬システム仕様

### 報酬構成要素
```python
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
```

## 自己対戦学習環境仕様

### 自己対戦エンジン
```python
class SelfPlayEngine:
    """自己対戦学習エンジン"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.main_agent = DQNPlayer("main")
        self.opponent_agent = DQNPlayer("opponent")
        self.experience_buffer = PrioritizedReplayBuffer(config.memory_size)
        self.episode_count = 0
        
    async def run_self_play_episode(self) -> Dict[str, Any]:
        """1エピソードの自己対戦実行"""
        
        # 定期的に相手をメインエージェントのコピーで更新
        if self.episode_count % self.config.opponent_update_interval == 0:
            self._update_opponent()
        
        # ゲーム初期化・実行
        game_engine = GameEngine(mode='versus', speed_multiplier=self.config.game_speed_multiplier)
        experiences = []
        
        prev_state = game_engine.get_state()
        
        while not game_engine.is_game_over():
            # 両エージェントの行動選択
            action_a = self.main_agent.get_action(prev_state, 'A')
            action_b = self.opponent_agent.get_action(prev_state, 'B')
            
            # 同時実行
            new_state = await game_engine.step_simultaneous(action_a, action_b)
            
            # 密な報酬計算
            reward_a = self._calculate_dense_reward(prev_state, action_a, new_state, 'A')
            
            # 経験保存（メインエージェントのみ）
            experiences.append(Experience(
                state=self.main_agent.encode_state(prev_state, 'A'),
                action=action_a.action_id,
                reward=reward_a.total(),
                next_state=self.main_agent.encode_state(new_state, 'A'),
                done=new_state['game_over'],
                valid_actions_mask=ActionSpace.create_action_mask(
                    ActionSpace.get_valid_actions(prev_state['players']['A']['current_puyo'], 
                                                prev_state['players']['A']['board'])
                )
            ))
            
            prev_state = new_state
        
        # 学習実行
        if len(self.experience_buffer) > self.config.batch_size:
            self._update_main_agent()
        
        self.episode_count += 1
        return {'episode': self.episode_count, 'winner': new_state['winner']}

class SelfPlayTraining:
    def __init__(self, config: TrainingConfig):
        self.agent = DQNPlayer()
        self.opponent = DQNPlayer()  # 同じモデル
        self.memory = ReplayBuffer(config.memory_size)
        
    async def train_episode(self):
        """1ゲームの学習実行"""
        game_engine = GameEngine(mode='versus')
        
        # ゲーム実行
        while not game_engine.is_game_over():
            # エージェント行動
            state_a = game_engine.get_state('A') 
            action_a = self.agent.get_action(state_a)
            
            state_b = game_engine.get_state('B')
            action_b = self.opponent.get_action(state_b)
            
            # 同時実行
            results = await game_engine.step(action_a, action_b)
            
            # 経験保存
            for result in results:
                self.memory.push(result.transition)
            
            # 学習実行
            if len(self.memory) > self.config.batch_size:
                self.update_model()
```

### 評価指標
```python
@dataclass
class EvaluationMetrics:
    """AI性能評価指標"""
    
    # 基本性能
    win_rate: float                    # 勝率（vs 無能CPU）
    average_score: float               # 平均スコア
    survival_time: float               # 平均生存時間
    
    # 戦術指標  
    average_chain_length: float        # 平均連鎖数
    max_chain_achieved: int            # 最大連鎖数
    chain_frequency: float             # 連鎖発生頻度
    
    # 効率指標
    actions_per_second: float          # 行動頻度
    thinking_time_ms: float            # 平均思考時間
    memory_usage_mb: float             # メモリ使用量
    
    # 学習指標
    training_episodes: int             # 学習エピソード数
    convergence_episode: int           # 収束エピソード数
    final_epsilon: float               # 最終探索率
```

## AI vs AI 対戦システム

### 自動トーナメント
```python
class AITournament:
    """複数AI間のトーナメント戦"""
    
    def __init__(self, participants: List[BasePlayer]):
        self.participants = participants
        self.results = TournamentResults()
        
    async def run_round_robin(self, games_per_match: int = 100):
        """総当り戦実行"""
        for i, player_a in enumerate(self.participants):
            for j, player_b in enumerate(self.participants[i+1:], i+1):
                
                wins_a = wins_b = 0
                
                for game in range(games_per_match):
                    winner = await self.play_match(player_a, player_b)
                    if winner == 'A':
                        wins_a += 1
                    else:
                        wins_b += 1
                
                self.results.record_match(
                    player_a.name, player_b.name, 
                    wins_a, wins_b
                )
        
        return self.results.generate_ranking()
```

### ゲームスピード制御
```python
class HighSpeedGameEngine(GameEngine):
    """学習用高速ゲームエンジン"""
    
    def __init__(self, speed_multiplier: float = 100.0):
        super().__init__()
        self.speed_multiplier = speed_multiplier
        self.disable_animations = True
        self.disable_audio = True
        
    def update_timing(self):
        """アニメーション時間を短縮"""
        self.puyo_fall_time = 0.001      # 通常: 0.1s
        self.chain_pause_time = 0.001    # 通常: 0.1s
        self.ojama_fall_time = 0.001     # 通常: 0.2s
```

## デバッグ・分析ツール

### 思考過程可視化
```python
class AIDebugger:
    def visualize_thinking(self, ai_player: DQNPlayer, game_state: GameState):
        """AIの思考過程を可視化"""
        
        # Q値分析
        q_values = ai_player.get_q_values(game_state)
        action_names = ['left', 'right', 'rot_l', 'rot_r', 'drop', 'wait']
        
        plt.figure(figsize=(10, 6))
        plt.bar(action_names, q_values.cpu().numpy())
        plt.title('Q-Values for Current State')
        plt.ylabel('Q-Value')
        plt.show()
        
        # 注目領域分析（将来：Attention機構）
        if hasattr(ai_player.model, 'attention_weights'):
            self.visualize_attention(game_state, ai_player.model.attention_weights)
```

### パフォーマンス分析
```python
class PerformanceProfiler:
    def profile_ai_performance(self, ai_player: BasePlayer, num_games: int = 100):
        """AI性能の詳細分析"""
        
        metrics = []
        
        for game_id in range(num_games):
            start_time = time.time()
            
            # ゲーム実行
            result = self.play_single_game(ai_player)
            
            # メトリクス収集
            metrics.append({
                'game_id': game_id,
                'score': result.final_score,
                'survival_time': result.survival_time,
                'max_chain': result.max_chain,
                'total_actions': result.total_actions,
                'game_duration': time.time() - start_time
            })
        
        return self.analyze_metrics(metrics)
```