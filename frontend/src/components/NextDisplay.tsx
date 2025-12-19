/**
 * ネクスト表示コンポーネント
 */

import React from 'react';
import { COLOR_CHARS } from '../types/game';
import type { PuyoPair } from '../types/game';
import './NextDisplay.css';

interface NextDisplayProps {
  nextPuyos: PuyoPair[];
  playerId: 'A' | 'B';
}

export const NextDisplay: React.FC<NextDisplayProps> = ({ nextPuyos, playerId }) => {
  const renderPuyoPair = (puyo: PuyoPair, label: string) => (
    <div className="next-item" key={label}>
      <div className="next-label">{label}</div>
      <div className="next-puyo-pair">
        <div className={`puyo puyo-${puyo.colors[0]}`}>
          {COLOR_CHARS[puyo.colors[0]]}
        </div>
        <div className={`puyo puyo-${puyo.colors[1]}`}>
          {COLOR_CHARS[puyo.colors[1]]}
        </div>
      </div>
    </div>
  );

  return (
    <div className={`next-display player-${playerId.toLowerCase()}`}>
      <div className="next-group">
        {nextPuyos.length > 0 && renderPuyoPair(nextPuyos[0], 'NEXT')}
        {nextPuyos.length > 1 && renderPuyoPair(nextPuyos[1], 'NEXT2')}
      </div>
    </div>
  );
};