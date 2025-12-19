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

## DQNプレイヤー仕様（将来実装）

### アーキテクチャ概要
```python
class DQNPlayer(BasePlayer):
    """深層Q学習によるぷよぷよAI"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path) if model_path else None
        self.epsilon = 0.1  # 探索率
        
    def get_action(self, game_state: GameState) -> str:
        if self.model is None:
            return self.random_action()
            
        state_tensor = self.encode_state(game_state)
        with torch.no_grad():
            q_values = self.model(state_tensor)
            
        if random.random() < self.epsilon:
            return self.random_action()  # 探索
        else:
            return self.decode_action(q_values.argmax().item())  # 活用
```

### 状態表現設計
```python
def encode_state(self, game_state: GameState) -> torch.Tensor:
    """
    ゲーム状態をニューラルネットワーク入力用テンソルに変換
    
    状態表現:
    - 盤面状態: 6x13x5 (各セルの色をone-hot)  
    - 落下ぷよ: 2x5 (2個のぷよの色をone-hot)
    - ネクスト: 4x5 (next, next2の色をone-hot)
    - スコア: 1 (正規化済み)
    - 相手おじゃま: 1 (正規化済み)
    
    合計: 6*13*5 + 2*5 + 4*5 + 1 + 1 = 422次元
    """
    # 盤面エンコード
    board_tensor = F.one_hot(torch.tensor(game_state.board), 5).float()
    
    # 現在ぷよエンコード  
    current_puyo = torch.zeros(2, 5)
    if game_state.current_puyo:
        current_puyo[0, game_state.current_puyo.colors[0]] = 1
        current_puyo[1, game_state.current_puyo.colors[1]] = 1
    
    # その他特徴量
    features = torch.tensor([
        game_state.score / 100000.0,      # スコア正規化
        game_state.ojama_pending / 30.0   # おじゃま正規化
    ])
    
    return torch.cat([
        board_tensor.flatten(),
        current_puyo.flatten(), 
        features
    ])
```

### ネットワーク構造
```python
class PuyoDQN(nn.Module):
    def __init__(self, input_size=422, hidden_size=512, output_size=6):
        super().__init__()
        
        # 盤面特徴抽出
        self.board_conv = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 3))
        )
        
        # 全結合層
        self.fc = nn.Sequential(
            nn.Linear(64*4*3 + 32, hidden_size),  # 盤面 + その他特徴
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, output_size)   # 6行動
        )
        
    def forward(self, x):
        # 盤面とその他特徴量を分離
        board_features = x[:, :390].view(-1, 5, 6, 13)  # 6x13x5
        other_features = x[:, 390:]                      # その他
        
        # 盤面畳み込み
        board_out = self.board_conv(board_features).flatten(1)
        
        # 結合して全結合層
        combined = torch.cat([board_out, other_features], dim=1)
        return self.fc(combined)
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
    target_update: int = 1000
    
    # 報酬設計
    score_reward_scale: float = 1e-5     # スコア報酬
    chain_bonus: float = 0.1             # 連鎖ボーナス  
    game_over_penalty: float = -1.0      # ゲームオーバー
    survival_reward: float = 0.001       # 生存報酬
    
    # 学習制御
    episodes: int = 100000
    save_interval: int = 1000
    eval_interval: int = 5000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
```

## 学習環境仕様

### 自己対戦学習
```python
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