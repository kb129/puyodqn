/**
 * 学習実行中画面コンポーネント
 */

import React, { useEffect, useState } from 'react';
import { useGameStore } from '../store/gameStore';
import { AppState } from '../types/game';
import './TrainingActive.css';

export const TrainingActive: React.FC = () => {
  const { trainingStatus, stopTraining, setAppState } = useGameStore();
  const [elapsedTime, setElapsedTime] = useState(0);

  // 経過時間カウンター
  useEffect(() => {
    const timer = setInterval(() => {
      setElapsedTime(prev => prev + 1);
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  const handleStopTraining = () => {
    if (window.confirm('学習を停止しますか？進捗は保存されます。')) {
      stopTraining();
      setAppState(AppState.TRAINING_RESULTS);
    }
  };

  const handlePauseTraining = () => {
    // 一時停止機能（将来実装）
    alert('一時停止機能は未実装です');
  };

  const formatTime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const formatEstimatedTime = (seconds: number): string => {
    if (seconds < 60) return `${seconds}秒`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}分`;
    return `${Math.floor(seconds / 3600)}時間${Math.floor((seconds % 3600) / 60)}分`;
  };

  if (!trainingStatus) {
    return (
      <div className="training-active">
        <div className="loading">学習を開始しています...</div>
      </div>
    );
  }

  return (
    <div className="training-active">
      <div className="training-container">
        <h1 className="training-title">AI学習実行中</h1>
        
        {/* 進捗バー */}
        <div className="progress-section">
          <div className="progress-header">
            <span>進捗: {trainingStatus.currentEpisode} / {trainingStatus.totalEpisodes}</span>
            <span>{trainingStatus.learningProgress.toFixed(1)}%</span>
          </div>
          <div className="progress-bar">
            <div 
              className="progress-fill"
              style={{ width: `${trainingStatus.learningProgress}%` }}
            />
          </div>
        </div>

        {/* 統計情報 */}
        <div className="stats-grid">
          <div className="stat-card">
            <h3>現在のエピソード</h3>
            <div className="stat-value">{trainingStatus.currentEpisode.toLocaleString()}</div>
          </div>

          <div className="stat-card">
            <h3>勝率</h3>
            <div className="stat-value">{(trainingStatus.winRate * 100).toFixed(1)}%</div>
          </div>

          <div className="stat-card">
            <h3>平均スコア</h3>
            <div className="stat-value">{trainingStatus.averageScore.toLocaleString()}</div>
          </div>

          <div className="stat-card">
            <h3>平均連鎖数</h3>
            <div className="stat-value">{trainingStatus.averageChain.toFixed(1)}</div>
          </div>
        </div>

        {/* 時間情報 */}
        <div className="time-info">
          <div className="time-item">
            <span>経過時間:</span>
            <span className="time-value">{formatTime(elapsedTime)}</span>
          </div>
          <div className="time-item">
            <span>推定残り時間:</span>
            <span className="time-value">{formatEstimatedTime(trainingStatus.estimatedTimeLeft)}</span>
          </div>
          <div className="time-item">
            <span>最終モデル保存:</span>
            <span className="time-value">{trainingStatus.lastModelSave || '未保存'}</span>
          </div>
        </div>

        {/* リアルタイムログ */}
        <div className="log-section">
          <h3>学習ログ</h3>
          <div className="log-container">
            <div className="log-entry">Episode {trainingStatus.currentEpisode}: Win rate improved to {(trainingStatus.winRate * 100).toFixed(1)}%</div>
            <div className="log-entry">Model saved at episode {Math.floor(trainingStatus.currentEpisode / 1000) * 1000}</div>
            <div className="log-entry">Average score: {trainingStatus.averageScore.toFixed(0)}</div>
            {/* 実際の実装では WebSocket などでリアルタイムログを受信 */}
          </div>
        </div>

        {/* 制御ボタン */}
        <div className="training-controls">
          <button 
            className="control-button pause"
            onClick={handlePauseTraining}
            disabled
          >
            一時停止
          </button>
          
          <button 
            className="control-button stop"
            onClick={handleStopTraining}
          >
            学習停止
          </button>
        </div>

        {/* 学習曲線グラフ（将来実装） */}
        <div className="chart-placeholder">
          <h3>学習曲線</h3>
          <div className="chart-mock">
            📈 リアルタイムグラフ表示予定
            <br />
            (勝率・スコア・連鎖数の推移)
          </div>
        </div>
      </div>
    </div>
  );
};