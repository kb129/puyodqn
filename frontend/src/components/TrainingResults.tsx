/**
 * 学習結果表示画面コンポーネント
 */

import React from 'react';
import { useGameStore } from '../store/gameStore';
import { AppState } from '../types/game';
import './TrainingResults.css';

export const TrainingResults: React.FC = () => {
  const { trainingResults, setAppState } = useGameStore();

  const handleNewTraining = () => {
    setAppState(AppState.TRAINING_MENU);
  };

  const handleMainMenu = () => {
    setAppState(AppState.MAIN_MENU);
  };

  const handleExportResults = () => {
    if (!trainingResults) return;
    
    const data = JSON.stringify(trainingResults, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `training-results-${new Date().toISOString().slice(0, 19)}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const formatTime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (hours > 0) {
      return `${hours}時間${minutes}分`;
    }
    return `${minutes}分${seconds % 60}秒`;
  };

  if (!trainingResults) {
    return (
      <div className="training-results">
        <div className="no-results">学習結果がありません</div>
      </div>
    );
  }

  const getPerformanceGrade = (winRate: number): { grade: string, color: string } => {
    if (winRate >= 0.9) return { grade: 'S', color: '#gold' };
    if (winRate >= 0.8) return { grade: 'A', color: '#silver' };
    if (winRate >= 0.7) return { grade: 'B', color: '#cd7f32' };
    if (winRate >= 0.6) return { grade: 'C', color: '#666' };
    return { grade: 'D', color: '#999' };
  };

  const performance = getPerformanceGrade(trainingResults.finalWinRate);

  return (
    <div className="training-results">
      <div className="results-container">
        <h1 className="results-title">学習完了</h1>
        
        {/* 総合評価 */}
        <div className="overall-grade">
          <div className={`grade-badge ${performance.color}`}>
            {performance.grade}
          </div>
          <div className="grade-description">
            <h2>最終勝率: {(trainingResults.finalWinRate * 100).toFixed(1)}%</h2>
            <p>学習が正常に完了しました</p>
          </div>
        </div>

        {/* 詳細統計 */}
        <div className="results-stats">
          <div className="stats-section">
            <h3>パフォーマンス統計</h3>
            <div className="stats-grid">
              <div className="stat-item">
                <span className="stat-label">最終勝率</span>
                <span className="stat-value">{(trainingResults.finalWinRate * 100).toFixed(1)}%</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">最高スコア</span>
                <span className="stat-value">{trainingResults.bestScore.toLocaleString()}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">最大連鎖数</span>
                <span className="stat-value">{trainingResults.maxChain}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">総ゲーム数</span>
                <span className="stat-value">{trainingResults.totalGames.toLocaleString()}</span>
              </div>
            </div>
          </div>

          <div className="stats-section">
            <h3>学習統計</h3>
            <div className="stats-grid">
              <div className="stat-item">
                <span className="stat-label">学習時間</span>
                <span className="stat-value">{formatTime(trainingResults.trainingTime)}</span>
              </div>
              {trainingResults.convergenceEpisode && (
                <div className="stat-item">
                  <span className="stat-label">収束エピソード</span>
                  <span className="stat-value">{trainingResults.convergenceEpisode.toLocaleString()}</span>
                </div>
              )}
              <div className="stat-item">
                <span className="stat-label">モデルパス</span>
                <span className="stat-value file-path">{trainingResults.modelPath}</span>
              </div>
            </div>
          </div>
        </div>

        {/* 学習曲線（プレースホルダー） */}
        <div className="results-charts">
          <h3>学習曲線</h3>
          <div className="chart-container">
            <div className="chart-placeholder">
              📊 勝率・スコア・連鎖数の推移グラフ
              <br />
              (Chart.js等で実装予定)
            </div>
          </div>
        </div>

        {/* 学習過程のハイライト */}
        <div className="learning-highlights">
          <h3>学習のハイライト</h3>
          <div className="highlight-timeline">
            <div className="highlight-item">
              <span className="highlight-episode">Episode 0</span>
              <span className="highlight-event">学習開始</span>
              <span className="highlight-metric">勝率: 0%</span>
            </div>
            {trainingResults.convergenceEpisode && (
              <div className="highlight-item">
                <span className="highlight-episode">Episode {trainingResults.convergenceEpisode}</span>
                <span className="highlight-event">性能収束</span>
                <span className="highlight-metric">安定した学習を達成</span>
              </div>
            )}
            <div className="highlight-item">
              <span className="highlight-episode">Episode {trainingResults.totalGames}</span>
              <span className="highlight-event">学習完了</span>
              <span className="highlight-metric">勝率: {(trainingResults.finalWinRate * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>

        {/* アクションボタン */}
        <div className="results-actions">
          <button 
            className="action-button export"
            onClick={handleExportResults}
          >
            結果をエクスポート
          </button>
          
          <button 
            className="action-button new-training"
            onClick={handleNewTraining}
          >
            新しい学習を開始
          </button>
          
          <button 
            className="action-button main-menu"
            onClick={handleMainMenu}
          >
            メインメニュー
          </button>
        </div>
      </div>
    </div>
  );
};