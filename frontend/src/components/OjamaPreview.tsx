/**
 * おじゃまぷよ予告表示コンポーネント
 */

import React from 'react';
import './OjamaPreview.css';

interface OjamaPreviewProps {
  pending: number;
  playerId: 'A' | 'B';
}

export const OjamaPreview: React.FC<OjamaPreviewProps> = ({ pending, playerId }) => {
  // おじゃまぷよの種類を計算（大きいものから左詰め）
  const calculateOjamaDisplay = (count: number): string => {
    if (count <= 0) return '';
    
    let result = '';
    let remaining = count;
    
    // 岩おじゃま（30個分）
    const rocks = Math.floor(remaining / 30);
    for (let i = 0; i < rocks; i++) {
      result += '◆';
    }
    remaining %= 30;
    
    // 大おじゃま（6個分）
    const larges = Math.floor(remaining / 6);
    for (let i = 0; i < larges; i++) {
      result += '●';
    }
    remaining %= 6;
    
    // 小おじゃま（1個分）
    for (let i = 0; i < remaining; i++) {
      result += '○';
    }
    
    return result;
  };

  const displayText = calculateOjamaDisplay(pending);

  return (
    <div className={`ojama-preview player-${playerId.toLowerCase()}`}>
      <span className="ojama-label">おじゃま予告:</span>
      <span className="ojama-display">
        {displayText || '　'}
      </span>
      {pending > 0 && (
        <span className="ojama-count">({pending}個)</span>
      )}
    </div>
  );
};