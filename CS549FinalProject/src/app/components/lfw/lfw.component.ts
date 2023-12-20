import { Component, ElementRef, OnInit, ViewChild } from '@angular/core';
import { HttpClient, HttpErrorResponse, HttpHeaders } from '@angular/common/http';
import { catchError, throwError } from 'rxjs';
import { MatDialog } from '@angular/material/dialog';
import { WebSocketService } from 'src/app/services/file-transfer.service';

@Component({
  selector: 'app-lfw',
  templateUrl: './lfw.component.html',
  styleUrls: ['./lfw.component.css']
})
export class LfwComponent implements OnInit{

  /* for displaying the sample images */
  LFWSampleImages: any[] = [];
  currPredictedLabel: string = "No image selected yet";
  isLFWSampleImagesLoading: boolean = false;


  constructor(private http: HttpClient, private dialog: MatDialog, private webSocketService: WebSocketService) {}

  ngOnInit(): void {
    this.loadSampleImages();
  }

  loadSampleImages(): void {

    if (!this.isLFWSampleImagesLoading) {
      this.isLFWSampleImagesLoading = true;
      const startIndex = this.LFWSampleImages.length;
      this.http.get<any>('https://127.0.0.1:5000/get-sample-image-faces?startIndex=' + 0)
        .subscribe((response: any) => {
          let parsed_response = []
          for (let i = 0; i < response.images.length; i++) {
            parsed_response.push([response.images[i], response.image_paths[i]])
          }
          this.LFWSampleImages = this.LFWSampleImages.concat(parsed_response);
          console.log('Recieved LFW Images with size' + this.LFWSampleImages.length)
          this.isLFWSampleImagesLoading = false;
        });
    }

  }

  getPredictedLabel(i: string): void {
    this.http.get<any>('https://127.0.0.1:5000/get-predicted-label?startIndex=' + i)
        .subscribe((response: any) => {
          this.currPredictedLabel = response;
          console.log('Recieved predicted label' + this.currPredictedLabel)
        });
  }

  onStartProcessing() {
    
  }
}
