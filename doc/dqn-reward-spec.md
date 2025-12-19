# DQN報酬設計仕様書 - 密な報酬＋自己対戦

## 概要

密な報酬設計により毎手でリッチなフィードバックを提供。
自己対戦学習により相対的な戦略を効率的に学習。

## 密な報酬システム設計

### 基本報酬構造
```python
@dataclass
class RewardComponents:
    """報酬の構成要素"""
    
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
        """総報酬計算"""
        return (self.survival_reward + self.tempo_reward + self.position_quality +
                self.chain_setup_reward + self.defense_reward + self.threat_mitigation +
                self.chain_execution_reward + self.damage_dealt + self.damage_received +
                self.win_reward + self.loss_penalty)

class DenseRewardCalculator:
    """密な報酬計算器"""
    
    def __init__(self):
        self.config = RewardConfig()
        self.previous_state = None
        
    def calculate_reward(self, 
                        prev_state: Dict[str, Any], 
                        action: PlacementAction,
                        new_state: Dict[str, Any],
                        player_id: str) -> RewardComponents:
        """1手の報酬を計算"""
        
        components = RewardComponents()
        
        # 基本生存報酬
        if not new_state.get('game_over', False):
            components.survival_reward = 0.01
        
        # 各種報酬を計算
        components.tempo_reward = self._calculate_tempo_reward(prev_state, new_state, player_id)
        components.position_quality = self._calculate_position_quality(action, new_state, player_id)
        components.chain_setup_reward = self._calculate_chain_setup_reward(prev_state, new_state, player_id)
        components.defense_reward = self._calculate_defense_reward(prev_state, new_state, player_id)
        components.threat_mitigation = self._calculate_threat_mitigation(prev_state, new_state, player_id)
        components.chain_execution_reward = self._calculate_chain_execution_reward(new_state, player_id)
        components.damage_dealt, components.damage_received = self._calculate_damage_rewards(prev_state, new_state, player_id)
        
        # 終了報酬
        if new_state.get('game_over', False):
            if new_state.get('winner') == player_id:
                components.win_reward = 10.0
            else:
                components.loss_penalty = -10.0
        
        return components
```

### 詳細報酬計算

#### 1. テンポ優位報酬
```python
def _calculate_tempo_reward(self, prev_state: Dict, new_state: Dict, player_id: str) -> float:
    """テンポ優位性による報酬"""
    if len(new_state['players']) == 1:
        return 0.0  # シングルプレイでは無効
    
    my_player = new_state['players'][player_id]
    opp_id = 'B' if player_id == 'A' else 'A'
    opp_player = new_state['players'][opp_id]
    
    # 高さ差による優位性
    my_avg_height = self._calculate_average_height(my_player['board'])
    opp_avg_height = self._calculate_average_height(opp_player['board'])
    height_advantage = (opp_avg_height - my_avg_height) / 12.0  # 正規化
    
    # 連鎖準備度差
    my_chain_setup = self._evaluate_chain_setup_progress(my_player['board'])
    opp_chain_setup = self._evaluate_chain_setup_progress(opp_player['board'])
    setup_advantage = (my_chain_setup - opp_chain_setup)
    
    # テンポ報酬
    tempo_score = (height_advantage * 0.02) + (setup_advantage * 0.03)
    return max(-0.1, min(0.1, tempo_score))  # クリッピング

def _calculate_average_height(self, board: List[List[int]]) -> float:
    """盤面の平均高さ計算"""
    heights = []
    for col in range(6):
        height = 0
        for row in range(12, -1, -1):
            if board[row][col] != 0:  # 空でない
                height = 12 - row
                break
        heights.append(height)
    return sum(heights) / len(heights)
```

#### 2. 配置品質報酬
```python
def _calculate_position_quality(self, action: PlacementAction, new_state: Dict, player_id: str) -> float:
    """配置の品質による報酬"""
    player = new_state['players'][player_id]
    board = player['board']
    
    quality_score = 0.0
    
    # 色の集約度評価
    color_clustering = self._evaluate_color_clustering(board, action.final_positions)
    quality_score += color_clustering * 0.05
    
    # バランス維持評価
    balance_score = self._evaluate_column_balance(board)
    quality_score += balance_score * 0.03
    
    # 連鎖可能性向上評価
    chain_potential_delta = self._calculate_chain_potential_improvement(action, board)
    quality_score += chain_potential_delta * 0.02
    
    # 危険回避評価
    danger_reduction = self._calculate_danger_reduction(action, board)
    quality_score += danger_reduction * 0.04
    
    return max(-0.15, min(0.15, quality_score))

def _evaluate_color_clustering(self, board: List[List[int]], placed_positions: List[Tuple[int, int, int]]) -> float:
    """色の集約度を評価"""
    clustering_score = 0.0
    
    for pos_x, pos_y, color in placed_positions:
        # 周囲8方向の同色ぷよ数をカウント
        adjacent_same_color = 0
        for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            nx, ny = pos_x + dx, pos_y + dy
            if 0 <= nx < 6 and 0 <= ny < 13:
                if board[ny][nx] == color:
                    adjacent_same_color += 1
        
        # 隣接同色数に基づくスコア
        clustering_score += adjacent_same_color / 8.0
    
    return clustering_score / max(1, len(placed_positions))
```

#### 3. 連鎖セットアップ報酬
```python
def _calculate_chain_setup_reward(self, prev_state: Dict, new_state: Dict, player_id: str) -> float:
    """連鎖セットアップ進展による報酬"""
    prev_player = prev_state['players'][player_id]
    new_player = new_state['players'][player_id]
    
    # 連鎖可能数の変化
    prev_chain_potential = analyze_max_chain_potential(prev_player['board'])
    new_chain_potential = analyze_max_chain_potential(new_player['board'])
    chain_improvement = new_chain_potential - prev_chain_potential
    
    # 連鎖発動点の変化
    prev_triggers = len(analyze_chain_triggers(prev_player['board']))
    new_triggers = len(analyze_chain_triggers(new_player['board']))
    trigger_improvement = new_triggers - prev_triggers
    
    # セットアップ報酬
    setup_reward = (chain_improvement * 0.1) + (trigger_improvement * 0.05)
    return max(-0.2, min(0.2, setup_reward))

def analyze_chain_triggers(board: List[List[int]]) -> List[Tuple[int, int]]:
    """連鎖発動可能位置を分析"""
    triggers = []
    
    # 各空位置で連鎖が発動するかテスト
    for row in range(13):
        for col in range(6):
            if board[row][col] == 0:  # 空セル
                for test_color in [1, 2, 3, 4]:  # 4色テスト
                    test_board = copy.deepcopy(board)
                    test_board[row][col] = test_color
                    
                    if simulate_chain_cascade(test_board) > 0:
                        triggers.append((col, row))
                        break
    
    return triggers
```

#### 4. 防御・脅威軽減報酬
```python
def _calculate_defense_reward(self, prev_state: Dict, new_state: Dict, player_id: str) -> float:
    """防御行動による報酬"""
    if len(new_state['players']) == 1:
        return 0.0
    
    # 高さ危険度の改善
    prev_danger = calculate_height_danger(prev_state['players'][player_id]['board'])
    new_danger = calculate_height_danger(new_state['players'][player_id]['board'])
    danger_reduction = prev_danger - new_danger
    
    # おじゃまぷよ対応
    prev_ojama = prev_state['players'][player_id]['ojama_pending']
    new_ojama = new_state['players'][player_id]['ojama_pending']
    ojama_cleared = prev_ojama - new_ojama
    
    defense_score = (danger_reduction * 0.1) + (ojama_cleared * 0.01)
    return max(-0.1, min(0.1, defense_score))

def _calculate_threat_mitigation(self, prev_state: Dict, new_state: Dict, player_id: str) -> float:
    """相手脅威に対する対応報酬"""
    if len(new_state['players']) == 1:
        return 0.0
    
    opp_id = 'B' if player_id == 'A' else 'A'
    opp_prev = prev_state['players'][opp_id]
    opp_new = new_state['players'][opp_id]
    
    # 相手の即座脅威レベルの変化を監視
    prev_threat = analyze_immediate_threat(opp_prev['board'], opp_prev['current_puyo'])
    new_threat = analyze_immediate_threat(opp_new['board'], opp_new['current_puyo'])
    
    # 自分の行動が相手の脅威にどう影響したか
    # （通常は直接影響しないが、連鎖でおじゃまを送った場合等）
    threat_change = prev_threat - new_threat
    
    return threat_change * 0.02  # 小さな報酬

def analyze_immediate_threat(board: List[List[int]], current_puyo: Dict) -> float:
    """即座脅威レベルを分析"""
    if not current_puyo:
        return 0.0
    
    max_damage = 0
    colors = current_puyo['colors']
    
    # 全配置パターンで最大ダメージを計算
    for col in range(6):
        for rot in range(4):
            test_board = copy.deepcopy(board)
            if place_virtual_puyo(test_board, col, rot, colors):
                damage = simulate_chain_damage(test_board)
                max_damage = max(max_damage, damage)
    
    return max_damage / 100.0  # 正規化
```

#### 5. 連鎖実行・ダメージ報酬
```python
def _calculate_chain_execution_reward(self, new_state: Dict, player_id: str) -> float:
    """連鎖実行による報酬"""
    player = new_state['players'][player_id]
    
    if not player['is_chaining'] and player['chain_count'] == 0:
        return 0.0
    
    chain_count = player['chain_count']
    
    # 連鎖数に応じた指数的報酬
    if chain_count >= 2:
        base_reward = 0.1 * (1.5 ** (chain_count - 1))
        return min(2.0, base_reward)  # 上限設定
    
    return 0.0

def _calculate_damage_rewards(self, prev_state: Dict, new_state: Dict, player_id: str) -> Tuple[float, float]:
    """与ダメージ・被ダメージ報酬"""
    if len(new_state['players']) == 1:
        return 0.0, 0.0
    
    opp_id = 'B' if player_id == 'A' else 'A'
    
    # 与ダメージ（相手のおじゃま増加）
    prev_opp_ojama = prev_state['players'][opp_id]['ojama_pending']
    new_opp_ojama = new_state['players'][opp_id]['ojama_pending']
    damage_dealt = (new_opp_ojama - prev_opp_ojama) * 0.02
    
    # 被ダメージ（自分のおじゃま増加）
    prev_my_ojama = prev_state['players'][player_id]['ojama_pending']
    new_my_ojama = new_state['players'][player_id]['ojama_pending']
    damage_received = -(new_my_ojama - prev_my_ojama) * 0.02
    
    return damage_dealt, damage_received
```

## 自己対戦学習システム

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
        
        # ゲーム初期化
        game_engine = GameEngine(mode='versus', speed_multiplier=self.config.game_speed)
        experiences_a = []
        experiences_b = []
        
        prev_state = game_engine.get_state()
        
        while not game_engine.is_game_over():
            # 両エージェントの行動選択
            action_a = self.main_agent.get_action(prev_state, 'A')
            action_b = self.opponent_agent.get_action(prev_state, 'B')
            
            # 同時実行
            new_state = await game_engine.step_simultaneous(action_a, action_b)
            
            # 報酬計算
            reward_a = self._calculate_step_reward(prev_state, action_a, new_state, 'A')
            reward_b = self._calculate_step_reward(prev_state, action_b, new_state, 'B')
            
            # 経験保存
            experiences_a.append(Experience(
                state=self.main_agent.encode_state(prev_state, 'A'),
                action=action_a.action_id,
                reward=reward_a.total(),
                next_state=self.main_agent.encode_state(new_state, 'A'),
                done=new_state['game_over']
            ))
            
            experiences_b.append(Experience(
                state=self.opponent_agent.encode_state(prev_state, 'B'),
                action=action_b.action_id,
                reward=reward_b.total(),
                next_state=self.opponent_agent.encode_state(new_state, 'B'),
                done=new_state['game_over']
            ))
            
            prev_state = new_state
        
        # 経験をバッファに追加
        for exp in experiences_a:
            priority = self._calculate_priority(exp)
            self.experience_buffer.push(exp, priority)
        
        # 学習実行（対戦相手の経験は使わない - 自己進化のため）
        if len(self.experience_buffer) > self.config.batch_size:
            self._update_main_agent()
        
        self.episode_count += 1
        
        return {
            'episode': self.episode_count,
            'winner': new_state['winner'],
            'main_agent_reward': sum(exp.reward for exp in experiences_a),
            'game_length': len(experiences_a)
        }
    
    def _update_opponent(self):
        """相手エージェントを現在のメインエージェントで更新"""
        # モデルの重みをコピー
        self.opponent_agent.model.load_state_dict(self.main_agent.model.state_dict())
        
        # 少し異なる探索率で多様性確保
        self.opponent_agent.epsilon = max(0.05, self.main_agent.epsilon * 1.2)
    
    def _calculate_step_reward(self, prev_state: Dict, action: PlacementAction, 
                              new_state: Dict, player_id: str) -> RewardComponents:
        """1手の報酬計算"""
        calculator = DenseRewardCalculator()
        return calculator.calculate_reward(prev_state, action, new_state, player_id)
    
    def _update_main_agent(self):
        """メインエージェントの学習更新"""
        batch = self.experience_buffer.sample(self.config.batch_size)
        loss = self.main_agent.update_model(batch)
        
        # 優先度更新
        for i, exp in enumerate(batch):
            new_priority = self._calculate_priority(exp)
            self.experience_buffer.update_priority(i, new_priority)
        
        return loss
```

### 学習スケジュール
```python
@dataclass
class SelfPlayConfig:
    """自己対戦設定"""
    
    # 基本設定
    total_episodes: int = 100000
    opponent_update_interval: int = 1000    # 相手更新頻度
    game_speed: float = 50.0                # ゲーム速度倍率
    
    # 学習制御
    learning_starts: int = 1000             # 学習開始エピソード
    target_update_frequency: int = 2000     # ターゲットネット更新
    save_frequency: int = 5000              # モデル保存頻度
    evaluation_frequency: int = 10000       # 評価実行頻度
    
    # 探索制御
    epsilon_schedule: str = "linear"        # linear, exponential
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_episodes: int = 50000
    
    # 報酬設定
    reward_config: RewardConfig = field(default_factory=RewardConfig)

class CurriculumSelfPlay:
    """カリキュラム自己対戦学習"""
    
    def __init__(self):
        self.phase = "beginner"
        self.phase_episodes = 0
        
    def should_advance_phase(self, metrics: Dict) -> bool:
        """フェーズ進行判定"""
        if self.phase == "beginner" and metrics['win_rate'] > 0.55:
            return True
        elif self.phase == "intermediate" and metrics['average_chain_length'] > 3.0:
            return True
        return False
    
    def adjust_training_parameters(self, phase: str) -> SelfPlayConfig:
        """フェーズに応じた学習パラメータ調整"""
        config = SelfPlayConfig()
        
        if phase == "beginner":
            config.opponent_update_interval = 500   # 頻繁な相手更新
            config.epsilon_start = 0.8              # 多めの探索
            
        elif phase == "intermediate":
            config.opponent_update_interval = 1500  # 中程度の更新
            config.epsilon_start = 0.6
            
        elif phase == "advanced":
            config.opponent_update_interval = 3000  # 少ない更新頻度
            config.epsilon_start = 0.3              # 活用重視
            
        return config
```

この密な報酬＋自己対戦システムで効率的な学習が期待できます。次はネットワーク構造の詳細化を進めますか？