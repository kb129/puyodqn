import { create } from "zustand";
import { AppState, PlayerType, Color } from "../types/game";
import type { GameState, PlayerState, PuyoPair, InputState } from "../types/game";

interface GameStore {
  appState: AppState;
  gameState: GameState | null;
  inputState: InputState;
  playerAType: PlayerType;
  playerBType: PlayerType;
  
  setAppState: (state: AppState) => void;
  setPlayerTypes: (playerA: PlayerType, playerB: PlayerType) => void;
  startSingleGame: () => void;
  startVersusGame: () => void;
  updateInputState: (input: Partial<InputState>) => void;
  applyAction: (playerId: "A" | "B", action: string) => void;
  updateFallingPuyo: (playerId: "A" | "B") => void;
  pauseGame: () => void;
  resumeGame: () => void;
  endGame: (winner?: "A" | "B") => void;
  resetGame: () => void;
}

const createInitialPlayer = (id: "A" | "B", type: PlayerType): PlayerState => ({
  id,
  type,
  board: Array(13).fill(null).map(() => Array(6).fill(Color.EMPTY)),
  fallingPuyo: {
    colors: [Color.RED, Color.GREEN], // [上のぷよ, 下のぷよ(軸)]
    x: 2,
    y: 1, // 初期位置: 上のぷよがy=1, 下のぷよがy=2
    rotation: 0
  },
  nextPuyos: [
    { colors: [Color.GREEN, Color.YELLOW], x: 2, y: 0, rotation: 0 },
    { colors: [Color.RED, Color.GREEN], x: 2, y: 0, rotation: 0 }
  ],
  score: 0,
  isChaining: false,
  chainCount: 0,
  ojamaPending: 0,
  lastRotationTime: undefined
});

const createInitialGameState = (mode: "single" | "versus", playerAType: PlayerType, playerBType: PlayerType): GameState => ({
  mode,
  players: {
    A: createInitialPlayer("A", playerAType),
    ...(mode === "versus" && { B: createInitialPlayer("B", playerBType) })
  },
  gameOver: false,
  turn: 0,
  seed: Math.floor(Math.random() * 1000000)
});

export const useGameStore = create<GameStore>((set, get) => ({
  appState: AppState.MAIN_MENU,
  gameState: null,
  inputState: {
    left: false,
    right: false,
    down: false,
    rotateLeft: false,
    rotateRight: false,
    pause: false
  },
  playerAType: PlayerType.HUMAN,
  playerBType: PlayerType.CPU_WEAK,

  setAppState: (state: AppState) => {
    set({ appState: state });
  },

  setPlayerTypes: (playerA: PlayerType, playerB: PlayerType) => {
    set({ playerAType: playerA, playerBType: playerB });
  },

  startSingleGame: () => {
    const { playerAType } = get();
    const gameState = createInitialGameState("single", playerAType, PlayerType.CPU_WEAK);
    set({ 
      appState: AppState.GAME_SINGLE,
      gameState
    });
  },

  startVersusGame: () => {
    const { playerAType, playerBType } = get();
    const gameState = createInitialGameState("versus", playerAType, playerBType);
    set({
      appState: AppState.GAME_VERSUS,
      gameState
    });
  },

  updateInputState: (input: Partial<InputState>) => {
    set(state => ({
      inputState: { ...state.inputState, ...input }
    }));
  },

  applyAction: (playerId: "A" | "B", action: string) => {
    const { gameState } = get();
    if (!gameState || gameState.gameOver) return;

    const player = gameState.players[playerId];
    if (!player || !player.fallingPuyo || player.isChaining) return;

    const puyo = player.fallingPuyo;
    let updated = false;

    switch (action) {
      case "move_left":
        if (puyo.x > 0 && canMove(player.board, puyo, -1, 0)) {
          puyo.x -= 1;
          updated = true;
        }
        break;
      case "move_right":
        if (puyo.x < 5 && canMove(player.board, puyo, 1, 0)) {
          puyo.x += 1;
          updated = true;
        }
        break;
      case "rotate_left":
        const newRotationL = (puyo.rotation - 1 + 4) % 4;
        const currentTime = Date.now();
        
        if (canRotateWithKick(player.board, puyo, newRotationL)) {
          const kickResult = tryRotationWithKick(player.board, puyo, newRotationL);
          if (kickResult) {
            puyo.rotation = newRotationL;
            puyo.x = kickResult.x;
            puyo.y = kickResult.y;
            updated = true;
          }
        } else {
          // 回転不可の場合、ダブル入力チェック
          if (player.lastRotationTime && 
              player.lastRotationTime.type === 'left' &&
              currentTime - player.lastRotationTime.timestamp <= 500) {
            // 500ms以内の同じ回転入力 - 上下入れ替え
            [puyo.colors[0], puyo.colors[1]] = [puyo.colors[1], puyo.colors[0]];
            updated = true;
          }
        }
        
        player.lastRotationTime = { type: 'left', timestamp: currentTime };
        break;
      case "rotate_right":
        const newRotationR = (puyo.rotation + 1) % 4;
        const currentTimeR = Date.now();
        
        if (canRotateWithKick(player.board, puyo, newRotationR)) {
          const kickResult = tryRotationWithKick(player.board, puyo, newRotationR);
          if (kickResult) {
            puyo.rotation = newRotationR;
            puyo.x = kickResult.x;
            puyo.y = kickResult.y;
            updated = true;
          }
        } else {
          // 回転不可の場合、ダブル入力チェック
          if (player.lastRotationTime && 
              player.lastRotationTime.type === 'right' &&
              currentTimeR - player.lastRotationTime.timestamp <= 500) {
            // 500ms以内の同じ回転入力 - 上下入れ替え
            [puyo.colors[0], puyo.colors[1]] = [puyo.colors[1], puyo.colors[0]];
            updated = true;
          }
        }
        
        player.lastRotationTime = { type: 'right', timestamp: currentTimeR };
        break;
      case "soft_drop":
        if (canMove(player.board, puyo, 0, 1)) {
          puyo.y += 1;
          updated = true;
        } else {
          landPuyo(gameState, playerId);
          updated = true;
        }
        break;
    }

    if (updated) {
      set({ gameState: { ...gameState } });
    }
  },

  updateFallingPuyo: (playerId: "A" | "B") => {
    const { gameState } = get();
    if (!gameState || gameState.gameOver) return;

    const player = gameState.players[playerId];
    if (!player || !player.fallingPuyo || player.isChaining) return;

    const puyo = player.fallingPuyo;
    
    if (canMove(player.board, puyo, 0, 1)) {
      puyo.y += 1;
      set({ gameState: { ...gameState } });
    } else {
      landPuyo(gameState, playerId);
      set({ gameState: { ...gameState } });
    }
  },



  pauseGame: () => {
    const { appState } = get();
    if (appState === AppState.GAME_SINGLE || appState === AppState.GAME_VERSUS) {
      set({ appState: AppState.PAUSED });
    }
  },

  resumeGame: () => {
    const { gameState } = get();
    if (!gameState) return;
    
    const targetState = gameState.mode === "single" 
      ? AppState.GAME_SINGLE 
      : AppState.GAME_VERSUS;
    set({ appState: targetState });
  },

  endGame: (winner?: "A" | "B") => {
    set(state => ({
      appState: AppState.GAME_OVER,
      gameState: state.gameState ? {
        ...state.gameState,
        gameOver: true,
        winner
      } : null
    }));
  },

  // 新しいぷよペアを生成
  generateNewPuyo: (playerId: "A" | "B") => {
    const { gameState } = get();
    if (!gameState) return;

    const player = gameState.players[playerId];
    if (!player) return;

    // Nextを更新
    const [nextPuyo, ...remainingNext] = player.nextPuyos;
    const colors = [Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW];
    const newNext: PuyoPair = {
      colors: [
        colors[Math.floor(Math.random() * colors.length)],
        colors[Math.floor(Math.random() * colors.length)]
      ],
      x: 2,
      y: 0,
      rotation: 0
    };

    // 新しいツモを設定
    player.fallingPuyo = {
      colors: nextPuyo.colors,
      x: 2,
      y: 1, // 初期位置
      rotation: 0
    };

    // Nextリストを更新
    player.nextPuyos = [...remainingNext, newNext];

    // ゲームオーバー判定
    if (isGameOver(player.board, player.fallingPuyo)) {
      gameState.gameOver = true;
      gameState.winner = playerId === "A" ? "B" : "A";
      set({ appState: AppState.GAME_OVER, gameState: { ...gameState } });
      return;
    }

    set({ gameState: { ...gameState } });
  },

  resetGame: () => {
    set({
      appState: AppState.MAIN_MENU,
      gameState: null,
      inputState: {
        left: false,
        right: false,
        down: false,
        rotateLeft: false,
        rotateRight: false,
        pause: false
      }
    });
  }
}));

function canMove(board: Color[][], puyo: PuyoPair, dx: number, dy: number): boolean {
  const positions = getPuyoPositions(puyo);
  
  for (const [px, py] of positions) {
    const newX = px + dx;
    const newY = py + dy;
    
    if (newX < 0 || newX >= 6 || newY < 0 || newY >= 13) {
      return false;
    }
    
    if (board[newY][newX] !== Color.EMPTY) {
      return false;
    }
  }
  
  return true;
}

// 回転判定（キック込み）
function canRotateWithKick(board: Color[][], puyo: PuyoPair, newRotation: number): boolean {
  return tryRotationWithKick(board, puyo, newRotation) !== null;
}

// 回転トライ（壁蹴り対応）
function tryRotationWithKick(board: Color[][], puyo: PuyoPair, newRotation: number): {x: number, y: number} | null {
  // キック試行順序: 元位置 → 左 → 右 → 上
  const kickOffsets = [
    [0, 0],   // 元位置
    [-1, 0],  // 左キック  
    [1, 0],   // 右キック
    [0, -1]   // 上キック
  ];
  
  for (const [kickX, kickY] of kickOffsets) {
    const testPuyo = { 
      ...puyo, 
      rotation: newRotation,
      x: puyo.x + kickX,
      y: puyo.y + kickY
    };
    
    const positions = getPuyoPositions(testPuyo);
    let canPlace = true;
    
    for (const [px, py] of positions) {
      if (px < 0 || px >= 6 || py < 0 || py >= 13) {
        canPlace = false;
        break;
      }
      
      if (board[py][px] !== Color.EMPTY) {
        canPlace = false;
        break;
      }
    }
    
    if (canPlace) {
      return { x: testPuyo.x, y: testPuyo.y };
    }
  }
  
  return null;
}

function getPuyoPositions(puyo: PuyoPair): [number, number][] {
  const positions: [number, number][] = [];
  
  // 軸ぷよ（下のぷよ）の位置計算
  // puyo.x, puyo.y は上のぷよの位置なので、軸ぷよは1つ下
  const axisX = puyo.x;
  const axisY = puyo.y + 1;
  
  // 上のぷよの相対位置を回転行列で計算（軸ぷよ基準）
  // rotation 0: 上 (0, -1) - 初期状態
  // rotation 1: 右 (1, 0)  - 右回転
  // rotation 2: 下 (0, 1)  - 180度回転  
  // rotation 3: 左 (-1, 0) - 左回転
  const rotationVectors = [
    [0, -1],  // rotation 0: 上のぷよが軸の上
    [1, 0],   // rotation 1: 上のぷよが軸の右
    [0, 1],   // rotation 2: 上のぷよが軸の下
    [-1, 0]   // rotation 3: 上のぷよが軸の左
  ];
  
  const [dx, dy] = rotationVectors[puyo.rotation];
  
  // positions[0] = 上のぷよ(colors[0]), positions[1] = 軸ぷよ(colors[1])
  positions.push([axisX + dx, axisY + dy]); // 上のぷよ
  positions.push([axisX, axisY]);           // 軸ぷよ（下）
  
  return positions;
}

// 連鎖処理システム（段階的実行）
function processChains(gameState: GameState, playerId: "A" | "B") {
  const player = gameState.players[playerId];
  if (!player) return;

  // 連鎖状態をリセット
  player.isChaining = false;
  player.chainCount = 0;
  
  // 完全重力適用
  applyFullGravity(player.board);
  
  // 1連鎖目をチェック・実行
  const chainGroups = findChainGroups(player.board);
  
  if (chainGroups.length > 0) {
    // 連鎖開始
    player.isChaining = true;
    player.chainCount = 1;
    
    // ぷよを消去してスコア計算
    let chainScore = 0;
    for (const group of chainGroups) {
      removePuyos(player.board, group);
      chainScore += calculateChainScore(group, 1);
    }
    
    player.score += chainScore;
    
    // おじゃまぷよ計算（対戦モード）
    if (gameState.mode === 'versus' && chainScore > 0) {
      const ojamaCount = Math.floor(chainScore / 70); // 70点で1個
      if (ojamaCount > 0) {
        const otherId = playerId === 'A' ? 'B' : 'A';
        const otherPlayer = gameState.players[otherId];
        if (otherPlayer) {
          otherPlayer.ojamaPending += ojamaCount;
        }
      }
    }
    
    // 重力処理を段階的に実行してから次の連鎖チェック
    applyGravityStepByStep(gameState, playerId);
  }
}

// 段階的重力処理
function applyGravityStepByStep(gameState: GameState, playerId: "A" | "B") {
  const player = gameState.players[playerId];
  if (!player || !player.isChaining) return;

  const moved = applyGravity(player.board);
  
  // 状態更新
  const { gameState: currentState } = useGameStore.getState();
  if (currentState) {
    useGameStore.setState({ gameState: { ...currentState } });
  }
  
  if (moved) {
    // まだ落下するぷよがある場合、0.1秒後に再実行
    window.setTimeout(() => {
      applyGravityStepByStep(gameState, playerId);
    }, 100);
  } else {
    // 重力処理完了、次の連鎖をチェック
    window.setTimeout(() => {
      continueChain(gameState, playerId);
    }, 200);
  }
}

// 連鎖継続処理
function continueChain(gameState: GameState, playerId: "A" | "B") {
  const player = gameState.players[playerId];
  if (!player || !player.isChaining) return;
  
  // 次の連鎖をチェック（重力は既に完了済み）
  const chainGroups = findChainGroups(player.board);
  
  if (chainGroups.length > 0) {
    // 連鎖継続
    player.chainCount++;
    
    // ぷよを消去してスコア計算
    let chainScore = 0;
    for (const group of chainGroups) {
      removePuyos(player.board, group);
      chainScore += calculateChainScore(group, player.chainCount);
    }
    
    player.score += chainScore;
    
    // おじゃまぷよ計算（対戦モード）
    if (gameState.mode === 'versus' && chainScore > 0) {
      const ojamaCount = Math.floor(chainScore / 70); // 70点で1個
      if (ojamaCount > 0) {
        const otherId = playerId === 'A' ? 'B' : 'A';
        const otherPlayer = gameState.players[otherId];
        if (otherPlayer) {
          otherPlayer.ojamaPending += ojamaCount;
        }
      }
    }
    
    // 状態更新
    const { gameState: currentState } = useGameStore.getState();
    if (currentState) {
      useGameStore.setState({ gameState: { ...currentState } });
    }
    
    // 重力処理を段階的に実行してから次の連鎖チェック
    applyGravityStepByStep(gameState, playerId);
  } else {
    // 連鎖終了 - 通常のゲームフローに戻る
    player.isChaining = false;
    console.log(`Chain ended for player ${playerId}, isChaining set to false`);
    
    // 対戦モードでは、おじゃまぷよは相手の次のぷよ着地時に降る
    // ここでは即座に降らせない
    
    // 状態更新して自動落下タイマー再開を促す
    const { gameState: currentState } = useGameStore.getState();
    if (currentState) {
      // 新しいオブジェクトとして状態を更新（リアクティブ更新のため）
      useGameStore.setState({ 
        gameState: { 
          ...currentState,
          players: {
            ...currentState.players,
            [playerId]: {
              ...currentState.players[playerId],
              isChaining: false
            }
          }
        } 
      });
    }
    
    // 連鎖表示を1秒後にリセット
    window.setTimeout(() => {
      if (player.chainCount > 0) {
        player.chainCount = 0;
        const { gameState: currentState } = useGameStore.getState();
        if (currentState) {
          useGameStore.setState({ gameState: { ...currentState } });
        }
      }
    }, 1000);
  }
}

// 重力適用
// 重力適用（ぷよを下に落下させる）- 段階的落下対応
function applyGravity(board: Color[][]): boolean {
  let moved = false;
  
  // 1段ずつ落下させる（アニメーション可能）
  for (let col = 0; col < 6; col++) {
    for (let row = 11; row >= 0; row--) { // 下から2段目から上へ
      if (board[row][col] !== Color.EMPTY && board[row + 1][col] === Color.EMPTY) {
        // 1段下に落下
        board[row + 1][col] = board[row][col];
        board[row][col] = Color.EMPTY;
        moved = true;
      }
    }
  }
  
  return moved; // 落下があったかどうかを返す
}

// 完全重力適用（すべて落下完了まで）
function applyFullGravity(board: Color[][]) {
  let moved = true;
  while (moved) {
    moved = applyGravity(board);
  }
}

// 連結群検出
function findChainGroups(board: Color[][]): Array<Array<[number, number]>> {
  const visited = new Set<string>();
  const chainGroups: Array<Array<[number, number]>> = [];
  
  for (let y = 0; y < 13; y++) {
    for (let x = 0; x < 6; x++) {
      const key = `${x},${y}`;
      if (!visited.has(key) && board[y][x] !== Color.EMPTY && board[y][x] !== Color.OJAMA) {
        const group = findConnectedGroup(board, x, y, board[y][x], visited);
        if (group.length >= 4) {
          chainGroups.push(group);
        }
      }
    }
  }
  
  return chainGroups;
}

// 連結群探索
function findConnectedGroup(
  board: Color[][], 
  startX: number, 
  startY: number, 
  color: Color, 
  visited: Set<string>
): Array<[number, number]> {
  const key = `${startX},${startY}`;
  
  if (visited.has(key) || 
      startX < 0 || startX >= 6 || 
      startY < 0 || startY >= 13 || 
      board[startY][startX] !== color) {
    return [];
  }
  
  visited.add(key);
  const group: Array<[number, number]> = [[startX, startY]];
  
  // 4方向に探索
  const directions = [[0, 1], [0, -1], [1, 0], [-1, 0]];
  for (const [dx, dy] of directions) {
    const connectedGroup = findConnectedGroup(board, startX + dx, startY + dy, color, visited);
    group.push(...connectedGroup);
  }
  
  return group;
}

// ぷよ消去
function removePuyos(board: Color[][], positions: Array<[number, number]>) {
  for (const [x, y] of positions) {
    board[y][x] = Color.EMPTY;
  }
}

// スコア計算（仕様書準拠）
function calculateChainScore(group: Array<[number, number]>, chainCount: number): number {
  const puyoCount = group.length;
  
  // 基本点 = 消した個数 × 10
  const baseScore = puyoCount * 10;
  
  // 連鎖ボーナステーブル（仕様書準拠）
  const chainBonusTable = [0, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512];
  const chainBonus = chainCount > 1 ? chainBonusTable[Math.min(chainCount - 1, chainBonusTable.length - 1)] || 512 : 0;
  
  // 連結ボーナステーブル（仕様書準拠）
  // 4個=0, 5個=2, 6個=3, 7個=4, 8個=5, 9個=6, 10個=7, 11個以上=10
  const countBonusTable = [0, 2, 3, 4, 5, 6, 7, 10];
  const countBonus = puyoCount >= 4 ? (countBonusTable[Math.min(puyoCount - 4, 7)] ?? 10) : 0;
  
  // 多色ボーナス（簡易版：1色なので0）
  const colorBonus = 0;
  
  // 仕様書の計算式：基本点 × (連鎖ボーナス + 多色ボーナス + 連結ボーナス)
  // ただし最小倍率は1（ボーナス合計が0でも基本点は保証）
  const totalBonus = Math.max(1, chainBonus + colorBonus + countBonus);
  
  // 最終スコア
  return baseScore * totalBonus;
}



// おじゃまぷよ落下処理（最大5段まで）
function dropOjamaPuyos(board: Color[][], count: number) {
  // 最大5段分（30個）まで制限
  const maxDrop = 5 * 6; // 5段 × 6列 = 30個
  const actualDrop = Math.min(count, maxDrop);
  
  let remaining = actualDrop;
  
  // 下から順に配置するように変更
  for (let y = 12; y >= 1 && remaining > 0; y--) {
    for (let x = 0; x < 6 && remaining > 0; x++) {
      if (board[y][x] === Color.EMPTY) {
        board[y][x] = Color.OJAMA;
        remaining--;
      }
    }
  }
  
  // 重力適用
  applyFullGravity(board);
  
  return actualDrop;
}

function landPuyo(gameState: GameState, playerId: "A" | "B") {
  const player = gameState.players[playerId];
  if (!player || !player.fallingPuyo) return;

  // まず相手のおじゃまぷよがあれば降らせる（着地直後に降る、5段分まで）
  if (gameState.mode === 'versus') {
    if (player.ojamaPending > 0) {
      const dropped = dropOjamaPuyos(player.board, player.ojamaPending);
      player.ojamaPending = Math.max(0, player.ojamaPending - dropped);
      
      // おじゃま落下後のゲームオーバー判定（簡易チェック）
      if (player.board[0][2] !== Color.EMPTY || player.board[0][3] !== Color.EMPTY) {
        gameState.gameOver = true;
        gameState.winner = playerId === "A" ? "B" : "A";
        return;
      }
    }
  }

  const positions = getPuyoPositions(player.fallingPuyo);
  const separatedPuyos: Array<{x: number, y: number, color: Color}> = [];
  
  // 各ぷよの着地可能性をチェック（ちぎれ処理）
  for (let i = 0; i < positions.length; i++) {
    const [x, y] = positions[i];
    // positions[0] = 上のぷよ(colors[0]), positions[1] = 軸ぷよ(colors[1])
    const color = player.fallingPuyo.colors[i];
    
    if (y >= 0 && y < 13 && x >= 0 && x < 6) {
      // この位置で着地できるかチェック
      if (y + 1 >= 13 || player.board[y + 1][x] !== Color.EMPTY) {
        // 着地可能 - その場に配置
        player.board[y][x] = color;
      } else {
        // ちぎれて独立して落下
        separatedPuyos.push({x, y, color});
      }
    }
  }
  
  // ちぎれたぷよの落下処理
  for (const separatedPuyo of separatedPuyos) {
    let fallY = separatedPuyo.y;
    
    // 落下可能な最下段まで落とす
    while (fallY + 1 < 13 && player.board[fallY + 1][separatedPuyo.x] === Color.EMPTY) {
      fallY++;
    }
    
    player.board[fallY][separatedPuyo.x] = separatedPuyo.color;
  }

  // 連鎖判定処理
  processChains(gameState, playerId);

  // 新しいぷよペア生成
  player.fallingPuyo = player.nextPuyos.shift() || null;
  
  if (player.nextPuyos.length < 2) {
    const colors: [Color, Color] = [
      Math.floor(Math.random() * 4) + 1 as Color,
      Math.floor(Math.random() * 4) + 1 as Color
    ];
    player.nextPuyos.push({
      colors,
      x: 2,
      y: 0,
      rotation: 0
    });
  }

  // ゲームオーバーチェック（新しいぷよが配置できないか）
  if (player.fallingPuyo) {
    const newPuyoPositions = getPuyoPositions(player.fallingPuyo);
    let canPlace = true;
    
    for (const [x, y] of newPuyoPositions) {
      if (y >= 0 && y < 13 && x >= 0 && x < 6) {
        if (player.board[y][x] !== Color.EMPTY) {
          canPlace = false;
          break;
        }
      }
    }
    
    if (!canPlace) {
      gameState.gameOver = true;
      if (gameState.mode === "versus") {
        gameState.winner = playerId === "A" ? "B" : "A";
      }
      return;
    }
  }
}

// ゲームオーバー判定
function isGameOver(board: Color[][], fallingPuyo: PuyoPair): boolean {
  // 13段目（隠し段）の中央2列に既にぷよがあるかチェック
  if (board[0][2] !== Color.EMPTY || board[0][3] !== Color.EMPTY) {
    return true;
  }

  // 新しいぷよが初期位置に置けるかチェック
  const positions = getPuyoPositions(fallingPuyo);
  for (const [x, y] of positions) {
    if (y >= 0 && y < 13 && x >= 0 && x < 6) {
      if (board[y][x] !== Color.EMPTY) {
        return true;
      }
    }
  }

  return false;
}
