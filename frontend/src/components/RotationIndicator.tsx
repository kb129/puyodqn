/**
 * 回転表示インジケーター（デバッグ用）
 */

import React from 'react';
import type { PuyoPair } from '../types/game';
import { COLOR_CHARS } from '../types/game';

interface RotationIndicatorProps {
  puyo: PuyoPair;
}

export const RotationIndicator: React.FC<RotationIndicatorProps> = ({ puyo }) => {
  const rotationNames = ['上', '右', '下', '左'];
  
  return (
    <div style={{
      position: 'absolute',
      top: '10px',
      right: '10px',
      background: 'rgba(0,0,0,0.7)',
      color: 'white',
      padding: '8px',
      borderRadius: '4px',
      fontSize: '0.8rem'
    }}>
      <div>回転: {puyo.rotation} ({rotationNames[puyo.rotation]})</div>
      <div>上: {COLOR_CHARS[puyo.colors[0]]} | 下(軸): {COLOR_CHARS[puyo.colors[1]]}</div>
      <div style={{marginTop: '4px', fontSize: '0.7rem', color: '#ccc'}}>
        K: 左回転 | L: 右回転
      </div>
    </div>
  );
};