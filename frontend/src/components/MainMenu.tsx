/**
 * メインメニューコンポーネント
 */

import React from 'react';
import { useGameStore } from '../store/gameStore';
import { AppState } from '../types/game';
import './MainMenu.css';

export const MainMenu: React.FC = () => {
  const { setAppState, startSingleGame } = useGameStore();

  const handleSinglePlay = () => {
    startSingleGame();
  };

  const handleVersusMode = () => {
    setAppState(AppState.PLAYER_SELECT);
  };

  const handleTrainingMode = () => {
    setAppState(AppState.TRAINING_MENU);
  };

  const handleSettings = () => {
    // 未実装
    alert('設定画面は未実装です');
  };

  return (
    <div className="main-menu">
      <div className="menu-container">
        <h1 className="game-title">PUYO PUYO</h1>
        
        <div className="menu-buttons">
          <button 
            className="menu-button"
            onClick={handleSinglePlay}
          >
            シングルプレイ
          </button>
          
          <button 
            className="menu-button"
            onClick={handleVersusMode}
          >
            対戦モード
          </button>

          <button 
            className="menu-button ai-training"
            onClick={handleTrainingMode}
          >
            AI学習モード
          </button>
          
          <button 
            className="menu-button"
            onClick={handleSettings}
          >
            設定
          </button>
        </div>
      </div>
    </div>
  );
};