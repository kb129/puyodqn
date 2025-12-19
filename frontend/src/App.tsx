import { useGameStore } from './store/gameStore';
import { AppState } from './types/game';
import { MainMenu } from './components/MainMenu';
import { PlayerSelect } from './components/PlayerSelect';
import { TrainingMenu } from './components/TrainingMenu';
import { TrainingConfig } from './components/TrainingConfig';
import { TrainingActive } from './components/TrainingActive';
import { TrainingResults } from './components/TrainingResults';
import { GameBoard } from './components/GameBoard';
import { NextDisplay } from './components/NextDisplay';
import { OjamaPreview } from './components/OjamaPreview';
import { RotationIndicator } from './components/RotationIndicator';
import { ChainEffect } from './components/ChainEffect';
import { useKeyboard } from './hooks/useKeyboard';
import { useGameLoop } from './hooks/useGameLoop';
import { useChainAnimation } from './hooks/useChainAnimation';
import { useCpuPlayer } from './hooks/useCpuPlayer';
import { useChainTimer } from './hooks/useChainTimer';
import { useChainReactivation } from './hooks/useChainReactivation';
import './App.css';

function App() {
  console.log('App component loaded');
  const { appState, gameState } = useGameStore();
  console.log('App state:', appState);

  // ゲーム中のみキーボード入力とゲームループを有効化
  const isInGame = appState === AppState.GAME_SINGLE || appState === AppState.GAME_VERSUS;
  const chainAnimations = useChainAnimation();
  useKeyboard(isInGame);
  useGameLoop(isInGame);
  useCpuPlayer(isInGame);
  useChainTimer();
  useChainReactivation();

  const renderCurrentScreen = () => {
    switch (appState) {
      case AppState.MAIN_MENU:
        return <MainMenu />;
      
      case AppState.PLAYER_SELECT:
        return <PlayerSelect />;
      
      case AppState.TRAINING_MENU:
        return <TrainingMenu />;
      
      case AppState.TRAINING_CONFIG:
        return <TrainingConfig />;
      
      case AppState.TRAINING_ACTIVE:
        return <TrainingActive />;
      
      case AppState.TRAINING_RESULTS:
        return <TrainingResults />;
      
      case AppState.GAME_SINGLE:
        if (!gameState) return <MainMenu />;
        return (
          <div className="game-screen single-game">
            <div className="game-header">
              <h2>PUYO PUYO</h2>
              <div className="score">
                SCORE: {gameState.players.A.score.toLocaleString()}
              </div>
            </div>
            <div className="game-content">
              <GameBoard 
                board={gameState.players.A.board}
                fallingPuyo={gameState.players.A.fallingPuyo}
                playerId="A"
              />
              <NextDisplay 
                nextPuyos={gameState.players.A.nextPuyos}
                playerId="A"
              />
            </div>
            <div className="game-instructions">
              <p>🎮 操作: ←→/A,D(移動) ↓/Space(落下) K,L(回転) | ⏰ 0.8秒で自動落下</p>
            </div>
            <button className="pause-button">PAUSE</button>
          </div>
        );
      
      case AppState.GAME_VERSUS:
        if (!gameState) return <MainMenu />;
        return (
          <div className="game-screen versus-game">
            <div className="game-header">
              <h2>PUYO PUYO VS</h2>
              <div className="scores">
                <span>1P: {gameState.players.A.score.toLocaleString()}</span>
                <span>2P: {gameState.players.B?.score.toLocaleString()}</span>
              </div>
            </div>
            <div className="game-content versus-content">
              {/* プレイヤーA */}
              <div className="player-area">
                <OjamaPreview 
                  pending={gameState.players.A.ojamaPending}
                  playerId="A"
                />
                <div className="board-and-next">
                  <div className={chainAnimations['A'] ? 'board-chaining' : ''} style={{position: 'relative'}}>
                    <GameBoard 
                      board={gameState.players.A.board}
                      fallingPuyo={gameState.players.A.fallingPuyo}
                      playerId="A"
                    />
                    {gameState.players.A.fallingPuyo && (
                      <RotationIndicator puyo={gameState.players.A.fallingPuyo} />
                    )}
                    <ChainEffect 
                      chainCount={gameState.players.A.chainCount}
                      isVisible={gameState.players.A.isChaining}
                    />
                  </div>
                  <NextDisplay 
                    nextPuyos={gameState.players.A.nextPuyos}
                    playerId="A"
                  />
                </div>
              </div>

              {/* プレイヤーB */}
              {gameState.players.B && (
                <div className="player-area">
                  <OjamaPreview 
                    pending={gameState.players.B.ojamaPending}
                    playerId="B"
                  />
                  <div className="board-and-next">
                    <div className={chainAnimations['B'] ? 'board-chaining' : ''}>
                      <GameBoard 
                        board={gameState.players.B.board}
                        fallingPuyo={gameState.players.B.fallingPuyo}
                        playerId="B"
                      />
                    </div>
                    <NextDisplay 
                      nextPuyos={gameState.players.B.nextPuyos}
                      playerId="B"
                    />
                  </div>
                </div>
              )}
            </div>
            <div className="game-instructions">
              <p>🎮 操作: ←→/A,D(移動) ↓/Space(落下) K,L(回転) | ⏰ 0.8秒で自動落下</p>
            </div>
            <button className="pause-button">PAUSE</button>
          </div>
        );
      
      case AppState.PAUSED:
        return (
          <div className="pause-screen">
            <div className="pause-menu">
              <h2>PAUSED</h2>
              <button className="menu-button" onClick={() => useGameStore.getState().resumeGame()}>
                ゲームを続ける
              </button>
              <button className="menu-button" onClick={() => useGameStore.getState().setAppState(AppState.MAIN_MENU)}>
                メインメニューに戻る
              </button>
            </div>
          </div>
        );
      
      case AppState.GAME_OVER:
        return (
          <div className="gameover-screen">
            <div className="gameover-menu">
              <h2>GAME OVER</h2>
              {gameState?.winner && (
                <p>Winner: Player {gameState.winner}</p>
              )}
              <button className="menu-button" onClick={() => useGameStore.getState().resetGame()}>
                リトライ
              </button>
              <button className="menu-button" onClick={() => useGameStore.getState().setAppState(AppState.MAIN_MENU)}>
                メインメニューに戻る
              </button>
            </div>
          </div>
        );
      
      default:
        return <MainMenu />;
    }
  };

  return (
    <div className="app">
      {renderCurrentScreen()}
    </div>
  );
}

export default App;
