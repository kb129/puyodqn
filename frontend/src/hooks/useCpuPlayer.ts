/**
 * CPU プレイヤー制御フック
 */

import { useEffect, useRef } from 'react';
import { useGameStore } from '../store/gameStore';
import { PlayerType } from '../types/game';
import { apiClient } from '../api/client';

const CPU_THINK_DELAY = 500; // CPUの思考時間（ミリ秒）

export const useCpuPlayer = (enabled: boolean = true) => {
  const { gameState, applyAction } = useGameStore();
  const thinkingTimeoutRef = useRef<number | null>(null);

  useEffect(() => {
    if (!enabled || !gameState || gameState.gameOver) {
      if (thinkingTimeoutRef.current) {
        clearTimeout(thinkingTimeoutRef.current);
        thinkingTimeoutRef.current = null;
      }
      return;
    }

    // CPUプレイヤーの行動処理
    const processCpuPlayer = async (player: any, playerId: 'A' | 'B') => {
      if (player && 
          (player.type === PlayerType.CPU_WEAK || player.type === PlayerType.DQN) && 
          player.fallingPuyo && 
          !player.isChaining) {
        
        // CPUの思考タイマー
        thinkingTimeoutRef.current = window.setTimeout(async () => {
          try {
            const response = await apiClient.getCpuAction({
              game_state: {
                players: {
                  [playerId]: {
                    board: player.board,
                    current_puyo: {
                      colors: player.fallingPuyo!.colors,
                      x: player.fallingPuyo!.x,
                      y: player.fallingPuyo!.y,
                      rotation: player.fallingPuyo!.rotation
                    }
                  }
                }
              },
              player_id: playerId,
              cpu_type: player.type === PlayerType.DQN ? "dqn" : "weak"
            });

            // CPU行動を適用
            if (response.action && response.action !== 'no_action') {
              applyAction(playerId, response.action);
            }
          } catch (error) {
            console.error(`CPU action failed for player ${playerId}:`, error);
            // フォールバック: ランダム行動
            const actions = ['move_left', 'move_right', 'rotate_right', 'soft_drop'];
            const randomAction = actions[Math.floor(Math.random() * actions.length)];
            applyAction(playerId, randomAction);
          }
        }, CPU_THINK_DELAY);
      }
    };

    // プレイヤーA・Bをチェック
    const playerA = gameState.players.A;
    const playerB = gameState.players.B;
    
    // いずれかのプレイヤーがCPUの場合に処理
    if (playerA && (playerA.type === PlayerType.CPU_WEAK || playerA.type === PlayerType.DQN)) {
      processCpuPlayer(playerA, 'A');
    } else if (playerB && (playerB.type === PlayerType.CPU_WEAK || playerB.type === PlayerType.DQN)) {
      processCpuPlayer(playerB, 'B');
    }

    return () => {
      if (thinkingTimeoutRef.current) {
        clearTimeout(thinkingTimeoutRef.current);
      }
    };
  }, [enabled, gameState, applyAction]);

  // クリーンアップ
  useEffect(() => {
    return () => {
      if (thinkingTimeoutRef.current) {
        clearTimeout(thinkingTimeoutRef.current);
      }
    };
  }, []);
};