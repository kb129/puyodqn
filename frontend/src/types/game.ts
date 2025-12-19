/**
 * ゲーム関連の型定義
 */

// 色定義
export const Color = {
  EMPTY: 0,
  RED: 1,
  BLUE: 2,
  GREEN: 3,
  YELLOW: 4,
  OJAMA: 5
} as const;

export type Color = typeof Color[keyof typeof Color];

// 行動定義
export const Action = {
  MOVE_LEFT: 'move_left',
  MOVE_RIGHT: 'move_right',
  ROTATE_LEFT: 'rotate_left', 
  ROTATE_RIGHT: 'rotate_right',
  SOFT_DROP: 'soft_drop',
  NO_ACTION: 'no_action'
} as const;

export type Action = typeof Action[keyof typeof Action];

// プレイヤータイプ
export const PlayerType = {
  HUMAN: 'human',
  CPU_WEAK: 'cpu_weak',
  DQN: 'dqn'
} as const;

export type PlayerType = typeof PlayerType[keyof typeof PlayerType];

// アプリケーション状態
export const AppState = {
  MAIN_MENU: 'main_menu',
  PLAYER_SELECT: 'player_select',
  GAME_SINGLE: 'game_single', 
  GAME_VERSUS: 'game_versus',
  TRAINING_MENU: 'training_menu',      // 学習メニュー
  TRAINING_CONFIG: 'training_config',  // 学習設定
  TRAINING_ACTIVE: 'training_active',  // 学習実行中
  TRAINING_RESULTS: 'training_results', // 学習結果表示
  PAUSED: 'paused',
  GAME_OVER: 'game_over'
} as const;

export type AppState = typeof AppState[keyof typeof AppState];

// ぷよペア
export interface PuyoPair {
  colors: [Color, Color];  // [上, 下]
  x: number;              // x座標 (0-5)
  y: number;              // y座標 (0-12)
  rotation: number;       // 回転状態 (0-3)
}

// 盤面状態
export interface BoardState {
  grid: Color[][];        // 6x13配列
  width: number;          // = 6
  height: number;         // = 13
}

// プレイヤー状態
export interface PlayerState {
  id: 'A' | 'B';
  type: PlayerType;
  board: Color[][];
  fallingPuyo: PuyoPair | null;
  nextPuyos: PuyoPair[];  // [next, next2]
  score: number;
  isChaining: boolean;
  chainCount: number;
  ojamaPending: number;
  lastRotationTime?: { type: 'left' | 'right', timestamp: number };
}

// ゲーム状態
export interface GameState {
  mode: 'single' | 'versus';
  players: { [key: string]: PlayerState };
  gameOver: boolean;
  winner?: 'A' | 'B';
  turn: number;
  seed: number;
}

// おじゃまぷよ予告
export interface OjamaPreview {
  small: number;   // 小おじゃま (1個)
  large: number;   // 大おじゃま (6個)
  rock: number;    // 岩おじゃま (30個)
}

// ゲーム設定
export interface GameConfig {
  fallSpeed: number;      // 落下速度 (ms)
  chainAnimSpeed: number; // 連鎖アニメ速度 (ms)
  autoRepeat: boolean;    // 自動リピート
}

// 入力状態
export interface InputState {
  left: boolean;
  right: boolean;
  down: boolean;
  rotateLeft: boolean;
  rotateRight: boolean;
  pause: boolean;
}

// API関連型
export interface ApiMessage<T = any> {
  id: string;
  type: string;
  payload: T;
  timestamp: number;
}

export interface ActionRequest {
  game_state: any;
  player_id: string;
  cpu_type: string;
}

export interface ActionResponse {
  action: Action;
  thinking_time: number;
  debug_info?: any;
}

// 色からCSSクラス名へのマッピング
export const COLOR_CLASSES = {
  [Color.EMPTY]: 'puyo-empty',
  [Color.RED]: 'puyo-red', 
  [Color.BLUE]: 'puyo-blue',
  [Color.GREEN]: 'puyo-green',
  [Color.YELLOW]: 'puyo-yellow',
  [Color.OJAMA]: 'puyo-ojama'
} as const;

// 色から表示文字へのマッピング  
export const COLOR_CHARS = {
  [Color.EMPTY]: '　',
  [Color.RED]: 'R',
  [Color.BLUE]: 'B', 
  [Color.GREEN]: 'G',
  [Color.YELLOW]: 'Y',
  [Color.OJAMA]: '×'
} as const;

// 学習モード定義
export const TrainingMode = {
  SELF_PLAY: 'self_play',           // 自己対戦学習
  VS_CPU: 'vs_cpu',                 // 対CPU学習
  TOURNAMENT: 'tournament',         // AIトーナメント
  EVALUATION: 'evaluation'          // 性能評価
} as const;

export type TrainingMode = typeof TrainingMode[keyof typeof TrainingMode];

// 学習設定
export interface TrainingConfig {
  mode: TrainingMode;
  episodes: number;                 // エピソード数
  speedMultiplier: number;          // ゲーム速度倍率
  saveInterval: number;             // モデル保存間隔
  evalInterval: number;             // 評価実行間隔
  learningRate: number;             // 学習率
  epsilon: number;                  // 探索率
  batchSize: number;                // バッチサイズ
  memorySize: number;               // リプレイメモリサイズ
  targetUpdate: number;             // ターゲットネット更新頻度
}

// 学習ステータス
export interface TrainingStatus {
  isRunning: boolean;               // 学習実行中フラグ
  currentEpisode: number;           // 現在のエピソード
  totalEpisodes: number;            // 総エピソード数
  winRate: number;                  // 勝率
  averageScore: number;             // 平均スコア
  averageChain: number;             // 平均連鎖数
  learningProgress: number;         // 学習進捗(0-100%)
  estimatedTimeLeft: number;        // 推定残り時間(秒)
  lastModelSave: string;            // 最後のモデル保存時刻
}

// 学習結果
export interface TrainingResults {
  finalWinRate: number;             // 最終勝率
  bestScore: number;                // 最高スコア
  maxChain: number;                 // 最大連鎖数
  totalGames: number;               // 総ゲーム数
  trainingTime: number;             // 学習時間(秒)
  convergenceEpisode?: number;      // 収束エピソード
  modelPath: string;                // 保存されたモデルパス
}