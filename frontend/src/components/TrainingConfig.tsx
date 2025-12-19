/**
 * 学習設定画面コンポーネント
 */

import React, { useState } from 'react';
import { useGameStore } from '../store/gameStore';
import { AppState, TrainingMode } from '../types/game';
import type { TrainingConfig as TrainingConfigType } from '../types/game';
import './TrainingConfig.css';

export const TrainingConfig: React.FC = () => {
  const { trainingMode, setAppState, startTraining } = useGameStore();
  
  const [config, setConfig] = useState<TrainingConfigType>({
    mode: trainingMode || TrainingMode.SELF_PLAY,
    episodes: 10000,
    speedMultiplier: 50.0,
    saveInterval: 1000,
    evalInterval: 2000,
    learningRate: 0.0001,
    epsilon: 0.1,
    batchSize: 32,
    memorySize: 100000,
    targetUpdate: 2000
  });

  const handleConfigChange = (field: keyof TrainingConfigType, value: number) => {
    setConfig(prev => ({ ...prev, [field]: value }));
  };

  const handleStartTraining = () => {
    startTraining(config);
    setAppState(AppState.TRAINING_ACTIVE);
  };

  const handleBack = () => {
    setAppState(AppState.TRAINING_MENU);
  };

  const getModeDescription = () => {
    switch (config.mode) {
      case TrainingMode.SELF_PLAY:
        return '同一モデルのコピー同士を対戦させて学習します。最も効率的な学習方法です。';
      case TrainingMode.VS_CPU:
        return '無能CPUと対戦して基礎的な戦術を学習します。初期学習に適しています。';
      case TrainingMode.TOURNAMENT:
        return '複数のAIモデルを総当り戦で比較評価します。';
      case TrainingMode.EVALUATION:
        return '学習済みモデルの性能を詳細に測定・分析します。';
      default:
        return '';
    }
  };

  return (
    <div className="training-config">
      <div className="config-container">
        <h1 className="config-title">学習設定</h1>
        
        <div className="mode-info">
          <h3>モード: {config.mode}</h3>
          <p className="mode-description">{getModeDescription()}</p>
        </div>

        <div className="config-sections">
          {/* 基本設定 */}
          <div className="config-section">
            <h3>基本設定</h3>
            <div className="config-grid">
              <div className="config-item">
                <label>エピソード数</label>
                <input
                  type="number"
                  value={config.episodes}
                  min="100"
                  max="1000000"
                  step="100"
                  onChange={(e) => handleConfigChange('episodes', parseInt(e.target.value))}
                />
                <span className="config-hint">学習するゲーム数</span>
              </div>

              <div className="config-item">
                <label>ゲーム速度倍率</label>
                <input
                  type="number"
                  value={config.speedMultiplier}
                  min="1"
                  max="1000"
                  step="1"
                  onChange={(e) => handleConfigChange('speedMultiplier', parseFloat(e.target.value))}
                />
                <span className="config-hint">通常の何倍速で実行</span>
              </div>
            </div>
          </div>

          {/* ハイパーパラメータ */}
          <div className="config-section">
            <h3>ハイパーパラメータ</h3>
            <div className="config-grid">
              <div className="config-item">
                <label>学習率</label>
                <input
                  type="number"
                  value={config.learningRate}
                  min="0.00001"
                  max="0.01"
                  step="0.00001"
                  onChange={(e) => handleConfigChange('learningRate', parseFloat(e.target.value))}
                />
              </div>

              <div className="config-item">
                <label>探索率 (ε)</label>
                <input
                  type="number"
                  value={config.epsilon}
                  min="0.01"
                  max="1.0"
                  step="0.01"
                  onChange={(e) => handleConfigChange('epsilon', parseFloat(e.target.value))}
                />
              </div>

              <div className="config-item">
                <label>バッチサイズ</label>
                <input
                  type="number"
                  value={config.batchSize}
                  min="8"
                  max="128"
                  step="8"
                  onChange={(e) => handleConfigChange('batchSize', parseInt(e.target.value))}
                />
              </div>

              <div className="config-item">
                <label>メモリサイズ</label>
                <input
                  type="number"
                  value={config.memorySize}
                  min="10000"
                  max="1000000"
                  step="10000"
                  onChange={(e) => handleConfigChange('memorySize', parseInt(e.target.value))}
                />
              </div>
            </div>
          </div>

          {/* 保存・評価設定 */}
          <div className="config-section">
            <h3>保存・評価設定</h3>
            <div className="config-grid">
              <div className="config-item">
                <label>モデル保存間隔</label>
                <input
                  type="number"
                  value={config.saveInterval}
                  min="100"
                  max="10000"
                  step="100"
                  onChange={(e) => handleConfigChange('saveInterval', parseInt(e.target.value))}
                />
                <span className="config-hint">エピソード単位</span>
              </div>

              <div className="config-item">
                <label>評価実行間隔</label>
                <input
                  type="number"
                  value={config.evalInterval}
                  min="100"
                  max="10000"
                  step="100"
                  onChange={(e) => handleConfigChange('evalInterval', parseInt(e.target.value))}
                />
                <span className="config-hint">エピソード単位</span>
              </div>

              <div className="config-item">
                <label>ターゲットネット更新頻度</label>
                <input
                  type="number"
                  value={config.targetUpdate}
                  min="100"
                  max="10000"
                  step="100"
                  onChange={(e) => handleConfigChange('targetUpdate', parseInt(e.target.value))}
                />
                <span className="config-hint">ステップ単位</span>
              </div>
            </div>
          </div>
        </div>

        <div className="config-actions">
          <button 
            className="start-training-button"
            onClick={handleStartTraining}
          >
            学習開始
          </button>
          
          <button 
            className="back-button"
            onClick={handleBack}
          >
            戻る
          </button>
        </div>
      </div>
    </div>
  );
};