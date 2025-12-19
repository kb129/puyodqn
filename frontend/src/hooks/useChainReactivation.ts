/**
 * 連鎖終了後のゲーム再開フック
 */

import { useEffect } from 'react';
import { useGameStore } from '../store/gameStore';

export const useChainReactivation = () => {
  const { gameState } = useGameStore();

  useEffect(() => {
    if (!gameState) return;

    const playerA = gameState.players.A;
    const playerB = gameState.players.B;

    // 連鎖終了を検知してログ出力
    if (playerA && !playerA.isChaining && playerA.chainCount === 0) {
      console.log('Player A: 連鎖終了、通常ゲーム再開');
    }
    
    if (playerB && !playerB.isChaining && playerB.chainCount === 0) {
      console.log('Player B: 連鎖終了、通常ゲーム再開');
    }

  }, [
    gameState?.players.A?.isChaining,
    gameState?.players.B?.isChaining,
    gameState?.players.A?.chainCount,
    gameState?.players.B?.chainCount
  ]);
};