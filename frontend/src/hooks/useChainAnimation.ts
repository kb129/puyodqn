/**
 * 連鎖アニメーションフック
 */

import { useEffect, useState } from 'react';
import { useGameStore } from '../store/gameStore';

export const useChainAnimation = () => {
  const { gameState } = useGameStore();
  const [animatingChains, setAnimatingChains] = useState<{[key: string]: boolean}>({});

  useEffect(() => {
    if (!gameState) return;

    const newAnimating: {[key: string]: boolean} = {};
    
    // プレイヤーAの連鎖チェック
    if (gameState.players.A?.isChaining) {
      newAnimating['A'] = true;
    }
    
    // プレイヤーBの連鎖チェック
    if (gameState.players.B?.isChaining) {
      newAnimating['B'] = true;
    }
    
    setAnimatingChains(newAnimating);
  }, [gameState]);

  return animatingChains;
};