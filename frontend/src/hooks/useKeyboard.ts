import { useEffect, useCallback } from "react";
import { useGameStore } from "../store/gameStore";

export const useKeyboard = (enabled: boolean = true) => {
  const { applyAction, pauseGame } = useGameStore();

  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    if (!enabled) return;

    event.preventDefault();
    
    switch (event.code) {
      case "ArrowLeft":
      case "KeyA":
        applyAction("A", "move_left");
        break;
      
      case "ArrowRight":
      case "KeyD":
        applyAction("A", "move_right");
        break;
      
      case "ArrowDown":
      case "Space":
      case "KeyS":
        applyAction("A", "soft_drop");
        break;
      
      case "KeyL":
        applyAction("A", "rotate_right");
        break;
      
      case "KeyK":
        applyAction("A", "rotate_left");
        break;
      
      case "Escape":
      case "KeyP":
        pauseGame();
        break;
    }
  }, [enabled, applyAction, pauseGame]);

  useEffect(() => {
    if (enabled) {
      window.addEventListener("keydown", handleKeyDown);
      return () => {
        window.removeEventListener("keydown", handleKeyDown);
      };
    }
  }, [enabled, handleKeyDown]);
};
