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
  CPU_WEAK: 'cpu_weak'
} as const;

export type PlayerType = typeof PlayerType[keyof typeof PlayerType];

// アプリケーション状態
export const AppState = {
  MAIN_MENU: 'main_menu',
  PLAYER_SELECT: 'player_select',
  GAME_SINGLE: 'game_single', 
  GAME_VERSUS: 'game_versus',
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