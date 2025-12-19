/**
 * 連鎖エフェクト表示コンポーネント
 */

import React from 'react';

interface ChainEffectProps {
  chainCount: number;
  isVisible: boolean;
}

export const ChainEffect: React.FC<ChainEffectProps> = ({ chainCount, isVisible }) => {
  if (!isVisible || chainCount === 0) return null;

  const getChainMessage = (count: number): string => {
    if (count >= 10) return `${count}連鎖！すごい！`;
    if (count >= 5) return `${count}連鎖！`;
    if (count >= 3) return `${count}連鎖！`;
    return `${count}連鎖`;
  };

  const getChainColor = (count: number): string => {
    if (count >= 10) return '#e91e63'; // ピンク
    if (count >= 7) return '#ff5722';  // 深いオレンジ
    if (count >= 5) return '#ff9800';  // オレンジ
    if (count >= 3) return '#ffc107';  // 黄色
    return '#4caf50'; // 緑
  };

  return (
    <div 
      className="chain-effect-overlay"
      style={{
        position: 'absolute',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        fontSize: `${Math.min(2 + chainCount * 0.2, 4)}rem`,
        fontWeight: 'bold',
        color: getChainColor(chainCount),
        textShadow: `
          2px 2px 4px rgba(0, 0, 0, 0.8),
          0 0 10px ${getChainColor(chainCount)}
        `,
        animation: 'chain-effect-appear 0.5s ease-out',
        zIndex: 100,
        pointerEvents: 'none'
      }}
    >
      {getChainMessage(chainCount)}
    </div>
  );
};

// CSS-in-JS スタイル
const chainEffectStyles = `
@keyframes chain-effect-appear {
  0% { 
    transform: translate(-50%, -50%) scale(0.5); 
    opacity: 0;
  }
  50% { 
    transform: translate(-50%, -50%) scale(1.2); 
    opacity: 1;
  }
  100% { 
    transform: translate(-50%, -50%) scale(1); 
    opacity: 1;
  }
}
`;

// スタイルを動的に追加
if (typeof document !== 'undefined') {
  const styleSheet = document.createElement('style');
  styleSheet.textContent = chainEffectStyles;
  document.head.appendChild(styleSheet);
}