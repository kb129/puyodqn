/**
 * API通信クライアント
 */

import type { ActionRequest, ActionResponse, ApiMessage } from '../types/game';

const API_BASE_URL = 'http://localhost:8000';

class ApiClient {
  async getCpuAction(request: ActionRequest): Promise<ActionResponse> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/cpu/action`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('CPU action request failed:', error);
      throw error;
    }
  }

  async validateGameState(gameState: any): Promise<any> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/game/validate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(gameState),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Game state validation failed:', error);
      throw error;
    }
  }

  async checkHealth(): Promise<any> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/health`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  }
}

// WebSocket クライアント（将来の拡張用）
class WebSocketClient {
  private ws: WebSocket | null = null;
  private messageCallbacks = new Map<string, (response: ApiMessage) => void>();

  connect(playerId: string): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(`ws://localhost:8000/ws/game/${playerId}`);
        
        this.ws.onopen = () => {
          console.log(`WebSocket connected for player ${playerId}`);
          resolve();
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          reject(error);
        };

        this.ws.onmessage = (event) => {
          try {
            const message: ApiMessage = JSON.parse(event.data);
            const callback = this.messageCallbacks.get(message.id);
            
            if (callback) {
              callback(message);
              this.messageCallbacks.delete(message.id);
            }
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
          }
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  async sendMessage<T>(message: ApiMessage<T>): Promise<ApiMessage> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket is not connected');
    }

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.messageCallbacks.delete(message.id);
        reject(new Error('Request timeout'));
      }, 5000);

      this.messageCallbacks.set(message.id, (response) => {
        clearTimeout(timeout);
        resolve(response);
      });

      this.ws!.send(JSON.stringify(message));
    });
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.messageCallbacks.clear();
  }
}

// シングルトンインスタンス
export const apiClient = new ApiClient();
export const wsClient = new WebSocketClient();