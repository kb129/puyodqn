import { useEffect, useRef } from "react";
import { useGameStore } from "../store/gameStore";

const FALL_INTERVAL = 800; // 0.8秒ごとに強制落下

export const useGameLoop = (enabled: boolean = true) => {
  const { gameState, updateFallingPuyo } = useGameStore();
  const gameLoopRef = useRef<number | null>(null);

  // シンプルなゲームループ
  const startGameLoop = () => {
    if (gameLoopRef.current) {
      clearInterval(gameLoopRef.current);
    }
    
    gameLoopRef.current = window.setInterval(() => {
      const currentGameState = useGameStore.getState().gameState;
      if (!currentGameState || currentGameState.gameOver) return;
      
      // プレイヤーAの自動落下
      if (currentGameState.players.A && !currentGameState.players.A.isChaining) {
        updateFallingPuyo("A");
      }
      
      // プレイヤーBの自動落下（対戦時）
      if (currentGameState.players.B && !currentGameState.players.B.isChaining) {
        updateFallingPuyo("B");
      }
    }, FALL_INTERVAL);
  };

  useEffect(() => {
    if (!enabled || !gameState || gameState.gameOver) {
      if (gameLoopRef.current) {
        clearInterval(gameLoopRef.current);
        gameLoopRef.current = null;
      }
      return;
    }

    startGameLoop();

    return () => {
      if (gameLoopRef.current) {
        clearInterval(gameLoopRef.current);
        gameLoopRef.current = null;
      }
    };
  }, [enabled, gameState?.gameOver]);

  return {};
};
