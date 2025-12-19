/**
 * 連鎖タイミング制御フック
 */

import { useEffect } from 'react';
import { useGameStore } from '../store/gameStore';

export const useChainTimer = () => {
  const { gameState } = useGameStore();

  useEffect(() => {
    if (!gameState) return;

    // プレイヤーの連鎖状態変化を監視
    const playerA = gameState.players.A;
    const playerB = gameState.players.B;

    // 連鎖中は新しいぷよの落下を停止
    if (playerA?.isChaining || playerB?.isChaining) {
      console.log(`連鎖実行中: A=${playerA?.chainCount || 0}連鎖, B=${playerB?.chainCount || 0}連鎖`);
    }

  }, [gameState?.players.A?.isChaining, gameState?.players.A?.chainCount,
      gameState?.players.B?.isChaining, gameState?.players.B?.chainCount]);
};