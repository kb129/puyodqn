/**
 * プレイヤー選択画面コンポーネント
 */

import React from 'react';
import { useGameStore } from '../store/gameStore';
import { AppState, PlayerType } from '../types/game';
import './PlayerSelect.css';

export const PlayerSelect: React.FC = () => {
  const { 
    playerAType, 
    playerBType, 
    setPlayerTypes, 
    setAppState, 
    startVersusGame 
  } = useGameStore();

  const handlePlayerAChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const newType = event.target.value as PlayerType;
    setPlayerTypes(newType, playerBType);
  };

  const handlePlayerBChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const newType = event.target.value as PlayerType;
    setPlayerTypes(playerAType, newType);
  };

  const handleStartGame = () => {
    startVersusGame();
  };

  const handleBack = () => {
    setAppState(AppState.MAIN_MENU);
  };

  return (
    <div className="player-select">
      <div className="select-container">
        <h1 className="select-title">PLAYER SELECT</h1>
        
        <div className="player-config">
          <div className="player-row">
            <label htmlFor="player-a">プレイヤーA:</label>
            <select 
              id="player-a"
              value={playerAType}
              onChange={handlePlayerAChange}
              className="player-dropdown"
            >
              <option value={PlayerType.HUMAN}>人間プレイヤー</option>
              <option value={PlayerType.CPU_WEAK}>無能CPU</option>
            </select>
          </div>

          <div className="vs-label">VS</div>

          <div className="player-row">
            <label htmlFor="player-b">プレイヤーB:</label>
            <select 
              id="player-b"
              value={playerBType}
              onChange={handlePlayerBChange}
              className="player-dropdown"
            >
              <option value={PlayerType.HUMAN}>人間プレイヤー</option>
              <option value={PlayerType.CPU_WEAK}>無能CPU</option>
            </select>
          </div>
        </div>

        <div className="select-buttons">
          <button 
            className="select-button start-button"
            onClick={handleStartGame}
          >
            ゲーム開始
          </button>
          
          <button 
            className="select-button back-button"
            onClick={handleBack}
          >
            戻る
          </button>
        </div>
      </div>
    </div>
  );
};