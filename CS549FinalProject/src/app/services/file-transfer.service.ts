// file-transfer.service.ts

import { Injectable } from '@angular/core';
import { Socket } from 'ngx-socket-io';

@Injectable({
  providedIn: 'root'
})
export class WebSocketService {

  constructor(private socket: Socket) {}

  requestDelayedResponse(): void {
    this.socket.emit('delayed_request');
  }

  onDelayedResponse(callback: (data: any) => void): void {
    this.socket.fromEvent('delayed_response').subscribe(callback);
  }
}