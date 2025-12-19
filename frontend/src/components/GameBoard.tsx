import React from "react";
import { Color, COLOR_CHARS } from "../types/game";
import type { PuyoPair } from "../types/game";
import "./GameBoard.css";

interface GameBoardProps {
  board: Color[][];
  fallingPuyo?: PuyoPair | null;
  playerId: "A" | "B";
}

export const GameBoard: React.FC<GameBoardProps> = ({ 
  board, 
  fallingPuyo,
  playerId 
}) => {
  // 落下中ぷよの位置を計算（下のぷよを軸とした回転）
  const getFallingPuyoPositions = (): { [key: string]: Color } => {
    const positions: { [key: string]: Color } = {};
    
    if (!fallingPuyo) return positions;
    
    const { x, y, rotation, colors } = fallingPuyo;
    
    // 軸ぷよ（下のぷよ）の位置
    const axisX = x;
    const axisY = y + 1;
    positions[`${axisX},${axisY}`] = colors[1]; // 軸ぷよ（下）
    
    // 上のぷよの位置（軸ぷよ基準で回転）
    const rotationVectors = [
      [0, -1],  // rotation 0: 上のぷよが軸の上
      [1, 0],   // rotation 1: 上のぷよが軸の右  
      [0, 1],   // rotation 2: 上のぷよが軸の下
      [-1, 0]   // rotation 3: 上のぷよが軸の左
    ];
    
    const [dx, dy] = rotationVectors[rotation];
    positions[`${axisX + dx},${axisY + dy}`] = colors[0]; // 上のぷよ
    
    return positions;
  };

  const fallingPositions = getFallingPuyoPositions();

  // セルの色を決定（落下中ぷよが優先）
  const getCellColor = (x: number, boardY: number): Color => {
    const key = `${x},${boardY}`;
    return fallingPositions[key] ?? board[boardY]?.[x] ?? Color.EMPTY;
  };

  // CSSクラス名を取得
  const getCellClass = (color: Color, isFalling: boolean = false): string => {
    const baseClass = "board-cell";
    const colorNames = ["empty", "red", "blue", "green", "yellow", "ojama"];
    const colorClass = `puyo-${colorNames[color] || "empty"}`;
    const fallingClass = isFalling ? "falling" : "";
    
    return [baseClass, colorClass, fallingClass].filter(Boolean).join(" ");
  };

  return (
    <div className={`game-board player-${playerId.toLowerCase()}`}>
      <div className="board-grid">
        {/* 表示範囲は1-12行（隠し段0は非表示） */}
        {board.slice(1, 13).map((row, displayY) =>
          row.map((_, x) => {
            const boardY = displayY + 1; // 実際の盤面座標
            const color = getCellColor(x, boardY);
            const key = `${x},${boardY}`;
            const isFalling = key in fallingPositions;
            
            return (
              <div
                key={key}
                className={getCellClass(color, isFalling)}
                data-x={x}
                data-y={boardY}
              >
                {COLOR_CHARS[color]}
              </div>
            );
          })
        )}
      </div>
      
      {/* ゲームオーバーライン表示 */}
      <div className="gameover-line" />
    </div>
  );
};
