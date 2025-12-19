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

    // プレイヤーBがCPUかつ連鎖中でない場合に行動
    const playerB = gameState.players.B;
    if (playerB && 
        playerB.type === PlayerType.CPU_WEAK && 
        playerB.fallingPuyo && 
        !playerB.isChaining) {
      
      // CPUの思考タイマー
      thinkingTimeoutRef.current = window.setTimeout(async () => {
        try {
          const response = await apiClient.getCpuAction({
            game_state: {
              players: {
                B: {
                  board: playerB.board,
                  current_puyo: {
                    colors: playerB.fallingPuyo!.colors,
                    x: playerB.fallingPuyo!.x,
                    y: playerB.fallingPuyo!.y,
                    rotation: playerB.fallingPuyo!.rotation
                  }
                }
              }
            },
            player_id: "B",
            cpu_type: "weak"
          });

          // CPU行動を適用
          if (response.action && response.action !== 'no_action') {
            applyAction('B', response.action);
          }
        } catch (error) {
          console.error('CPU action failed:', error);
          // フォールバック: ランダム行動
          const actions = ['move_left', 'move_right', 'rotate_right', 'soft_drop'];
          const randomAction = actions[Math.floor(Math.random() * actions.length)];
          applyAction('B', randomAction);
        }
      }, CPU_THINK_DELAY);
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