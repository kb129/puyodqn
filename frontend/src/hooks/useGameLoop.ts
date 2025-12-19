import { useEffect, useRef } from "react";
import { useGameStore } from "../store/gameStore";

const FALL_INTERVAL = 800; // 1秒ごとに強制落下

export const useGameLoop = (enabled: boolean = true) => {
  const { gameState, updateFallingPuyo } = useGameStore();
  const playerATimerRef = useRef<number | null>(null);
  const playerBTimerRef = useRef<number | null>(null);

  // プレイヤー別のタイマー管理
  const startPlayerTimer = (playerId: "A" | "B") => {
    const timerRef = playerId === "A" ? playerATimerRef : playerBTimerRef;
    
    if (timerRef.current) {
      clearTimeout(timerRef.current);
    }
    
    timerRef.current = window.setTimeout(() => {
      const currentGameState = useGameStore.getState().gameState;
      const player = currentGameState?.players[playerId];
      
      if (currentGameState && player) {
        if (!player.isChaining) {
          // 連鎖中でない場合は自動落下実行
          updateFallingPuyo(playerId);
          // 次のタイマーを開始
          startPlayerTimer(playerId);
        } else {
          // 連鎖中の場合は少し待ってから再チェック
          const recheckTimer = window.setTimeout(() => {
            const recheckGameState = useGameStore.getState().gameState;
            const recheckPlayer = recheckGameState?.players[playerId];
            if (recheckGameState && recheckPlayer && !recheckPlayer.isChaining) {
              startPlayerTimer(playerId);
            } else if (recheckGameState && recheckPlayer && recheckPlayer.isChaining) {
              // まだ連鎖中なら再度チェック
              startPlayerTimer(playerId);
            }
          }, 200);
          
          // 新しいタイマーIDを保存
          if (playerId === "A") {
            playerATimerRef.current = recheckTimer;
          } else {
            playerBTimerRef.current = recheckTimer;
          }
        }
      }
    }, FALL_INTERVAL);
  };

  useEffect(() => {
    if (!enabled || !gameState || gameState.gameOver) {
      // タイマークリア
      if (playerATimerRef.current) {
        clearTimeout(playerATimerRef.current);
        playerATimerRef.current = null;
      }
      if (playerBTimerRef.current) {
        clearTimeout(playerBTimerRef.current);
        playerBTimerRef.current = null;
      }
      return;
    }

    // プレイヤーAのタイマー開始・再開
    if (gameState.players.A) {
      if (!gameState.players.A.isChaining && !playerATimerRef.current) {
        console.log("Starting timer for player A");
        startPlayerTimer("A");
      } else if (!gameState.players.A.isChaining && playerATimerRef.current) {
        // 連鎖が終了した場合、タイマーを再開
        console.log("Restarting timer for player A after chain");
        startPlayerTimer("A");
      }
    }
    
    // プレイヤーBのタイマー開始・再開（対戦時）
    if (gameState.players.B) {
      if (!gameState.players.B.isChaining && !playerBTimerRef.current) {
        console.log("Starting timer for player B");
        startPlayerTimer("B");
      } else if (!gameState.players.B.isChaining && playerBTimerRef.current) {
        // 連鎖が終了した場合、タイマーを再開
        console.log("Restarting timer for player B after chain");
        startPlayerTimer("B");
      }
    }

  }, [enabled, gameState?.players.A?.isChaining, gameState?.players.B?.isChaining, gameState, updateFallingPuyo]);

  // コンポーネントアンマウント時のクリーンアップ
  useEffect(() => {
    return () => {
      if (playerATimerRef.current) {
        clearTimeout(playerATimerRef.current);
      }
      if (playerBTimerRef.current) {
        clearTimeout(playerBTimerRef.current);
      }
    };
  }, []);

  // 手動でタイマーリセット（着地時などに呼ばれる）
  const resetPlayerTimer = (playerId: "A" | "B") => {
    startPlayerTimer(playerId);
  };

  return { resetPlayerTimer };
};
