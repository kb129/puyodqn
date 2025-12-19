/**
 * AI学習メニューコンポーネント
 */

import React from 'react';
import { useGameStore } from '../store/gameStore';
import { AppState, TrainingMode } from '../types/game';
import './TrainingMenu.css';

export const TrainingMenu: React.FC = () => {
  const { setAppState, setTrainingMode } = useGameStore();

  const handleSelfPlayTraining = () => {
    setTrainingMode(TrainingMode.SELF_PLAY);
    setAppState(AppState.TRAINING_CONFIG);
  };

  const handleCpuTraining = () => {
    setTrainingMode(TrainingMode.VS_CPU);
    setAppState(AppState.TRAINING_CONFIG);
  };

  const handleTournament = () => {
    setTrainingMode(TrainingMode.TOURNAMENT);
    setAppState(AppState.TRAINING_CONFIG);
  };

  const handleEvaluation = () => {
    setTrainingMode(TrainingMode.EVALUATION);
    setAppState(AppState.TRAINING_CONFIG);
  };

  const handleBack = () => {
    setAppState(AppState.MAIN_MENU);
  };

  return (
    <div className="training-menu">
      <div className="menu-container">
        <h1 className="menu-title">AI学習モード</h1>
        <p className="menu-description">
          DQNエージェントの学習・評価を行います
        </p>
        
        <div className="training-options">
          <div className="training-card">
            <h3>自己対戦学習</h3>
            <p>同じモデル同士を対戦させて学習</p>
            <button 
              className="training-button primary"
              onClick={handleSelfPlayTraining}
            >
              開始
            </button>
          </div>

          <div className="training-card">
            <h3>対CPU学習</h3>
            <p>無能CPUと対戦して基礎学習</p>
            <button 
              className="training-button secondary"
              onClick={handleCpuTraining}
            >
              開始
            </button>
          </div>

          <div className="training-card">
            <h3>AIトーナメント</h3>
            <p>複数AIモデルの総当り戦</p>
            <button 
              className="training-button tournament"
              onClick={handleTournament}
            >
              開始
            </button>
          </div>

          <div className="training-card">
            <h3>性能評価</h3>
            <p>学習済みモデルの性能測定</p>
            <button 
              className="training-button evaluation"
              onClick={handleEvaluation}
            >
              開始
            </button>
          </div>
        </div>

        <button 
          className="back-button"
          onClick={handleBack}
        >
          メインメニューに戻る
        </button>
      </div>
    </div>
  );
};