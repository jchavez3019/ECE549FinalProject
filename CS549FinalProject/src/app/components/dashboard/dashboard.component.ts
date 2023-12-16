import { Component, ElementRef, OnInit, ViewChild } from '@angular/core';
import { HttpClient, HttpErrorResponse, HttpHeaders } from '@angular/common/http';
import { catchError, throwError } from 'rxjs';
import { ModalComponent } from './modal/modal.component';
import { MatDialog } from '@angular/material/dialog';
import { WebSocketService } from 'src/app/services/file-transfer.service';

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.css']
})
export class DashboardComponent implements OnInit {

  selectedFiles: FileList | undefined;

  onDragOverTrainingFlag: boolean = false;
  onDragOverTestingFlag: boolean = false;
  uploadedTrainingData: boolean = false;
  numTrainingDataImagesUploaded: number = 0;
  uploadedTestingData: boolean = false;
  numTestingDataImagesUploaded: number = 0;
  firstImgUploaded: boolean = false;
  successfulUpload: boolean = false;
  lastError: HttpErrorResponse | null = null;
  standBy: boolean = false;

  /* training data set */
  trainingDataFiles: { file: File; dataURL: string }[] = [];

  /* testing data set */
  testingDataFiles: { file: File; dataURL: string }[] = [];

  /* for displaying the uploaded images */
  trainingImages: any[] = []; // Array to hold images
  isTrainingImagesLoading: boolean = false;
  

  testingImages: any[] = []; // Array to hold images
  LFWSampleImages: any[] = [];
  isTestingImagesLoading: boolean = false;
  isLFWSampleImagesLoading: boolean = false;

  batchSize: number = 20;

  showTrainingImages = true;

  constructor(private http: HttpClient, private dialog: MatDialog, private webSocketService: WebSocketService) {}

  ngOnInit(): void {
    this.loadSampleImages();
  }

  /* gets a handle on DOM elements for uploading files */
  @ViewChild('trainingsetInput') trainingsetInput: any;
  @ViewChild('testingsetInput') testingsetInput: any;

  /* functions called for clicking on elements */
  openTrainingsetInput() {
    this.trainingsetInput.nativeElement.click();
  }
  openTestingsetInput() {
    this.testingsetInput.nativeElement.click();
  }

  onDragOver(event: Event, el: any): void {
    if (el.id == 'trainingsetDrop') {
      this.onDragOverTrainingFlag = true;
    }
    else if (el.id == 'testingsetDrop') {
      this.onDragOverTestingFlag = true;
    }
    event.preventDefault();
  }

  onDragLeave(event: Event, el: any): void {
    if (el.id == 'trainingsetDrop') {
      this.onDragOverTrainingFlag = false;
    }
    else if (el.id == 'testingsetDrop') {
      this.onDragOverTestingFlag = false;
    }
    event.preventDefault();
  }

  onDrop(event: DragEvent, el: any): void {

    event.preventDefault();
    if (el.id == 'trainingsetDrop') {
      this.onDragOverTrainingFlag = false;
    }
    else if (el.id == 'testingsetDrop') {
      this.onDragOverTestingFlag = false;
    }

    /* ensure the event has files that can be received */
    if (event != null && event.dataTransfer != null) {
      this.handleFiles(event.dataTransfer.files, el);
    }
      
  }

  handleFiles(files: FileList | null, el: any): void {
    /* DEBUG: temporarily here to test web socket service */
    const fn_handler = (data: any) => {
      console.log(`Received delayed response from Flask server with message: ${data.message}`);
    }

    this.webSocketService.onDelayedResponse(fn_handler);
    this.webSocketService.requestDelayedResponse();

    if (files == null) {
      console.error("handleFiles did not received type FileList");
      return;
    }

    /* check which container the images were dropped into */
    let isTrainingData: Boolean;
    if (el.id == "trainingsetDrop") {
      isTrainingData = true;
    }
    else {
      isTrainingData = false;
    }

    this.standBy = true;

    // this.uploadedFiles = [];

    let uploadedFiles: { file: File; dataURL: string }[] = [];

    // const file: File = files.item(0)!;
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      const file: File = files.item(i)!;

      if (!file.type.startsWith('image/')) {
        console.warn(`File with name ${file.name} was skipped because it is not of type image`);
        continue;
      }

      const reader = new FileReader();
      reader.onload = (event) => {
        const dataURL = event.target?.result as string;
        uploadedFiles.push({ file, dataURL });
      };
      reader.readAsDataURL(file);
      formData.append('images', file);
    }

    /* save the files that were uploaded */
    if (isTrainingData) {
      this.numTrainingDataImagesUploaded = files.length;
      this.trainingDataFiles = uploadedFiles;
      formData.append('isTrainingData', 'true');
      this.uploadedTrainingData = true;
    }
    else {
      this.numTestingDataImagesUploaded = files.length;
      this.testingDataFiles = uploadedFiles;
      formData.append('isTrainingData', 'false');
      this.uploadedTestingData = true;
    }

    this.http.post('https://127.0.0.1:5000/upload', formData).pipe(
      catchError((error: HttpErrorResponse) => {
        this.lastError = error;
        this.successfulUpload = false;

        /* the user has uploaded an image for the first item */
        if (!this.firstImgUploaded) {
          this.firstImgUploaded = true;
        }

        if (error.status === 0) {
          // A client-side or network error occurred. Handle it accordingly.
          console.error('An error occurred:', error.error, error.status);
        } else {
          // The backend returned an unsuccessful response code.
          // The response body may contain clues as to what went wrong.
          console.error(`Backend returned code ${error.status}, body was: `, error.error);
        }

        this.standBy = false;

        return throwError(() => new Error('Something bad happened; please try again later.'));
      })
    ).subscribe((resp: any) => {
      console.log("Successfully sent data to backend with response: ", resp);
      this.lastError = null;
      this.successfulUpload = true;

      /* the user has uploaded an image for the first item */
      if (!this.firstImgUploaded) {
        this.firstImgUploaded = true;
      }

      this.standBy = false;

      /* retrive the images to display */
      if (isTrainingData) {
        this.loadMoreTrainingImages();
        console.log(`Training data images: ${this.numTrainingDataImagesUploaded}`)
      }
      else {
        this.loadMoreTestingImages();
      }

    });
    
  }

  loadMoreTrainingImages(): void {

    if (!this.uploadedTrainingData)
      return;

    if (!this.isTrainingImagesLoading) {
      this.isTrainingImagesLoading = true;
      const startIndex = this.trainingImages.length;
      this.http.get<any>('https://127.0.0.1:5000/get-training-images?startIndex=' + startIndex)
        .subscribe((response: any) => {
          this.trainingImages = this.trainingImages.concat(response);
          this.isTrainingImagesLoading = false;
        });
    }
  }

  loadMoreTestingImages(): void {

    if (!this.uploadedTestingData)
      return;

    if (!this.isTestingImagesLoading) {
      this.isTestingImagesLoading = true;
      const startIndex = this.testingImages.length;
      this.http.get<any>('https://127.0.0.1:5000/get-testing-images?startIndex=' + startIndex)
        .subscribe((response: any) => {
          this.testingImages = this.testingImages.concat(response);
          this.isTestingImagesLoading = false;
        });
    }
  }

  loadSampleImages(): void {

    if (!this.isLFWSampleImagesLoading) {
      this.isLFWSampleImagesLoading = true;
      const startIndex = this.LFWSampleImages.length;
      this.http.get<any>('https://127.0.0.1:5000/get-sample-image-faces?startIndex=' + 0)
        .subscribe((response: any) => {
          this.LFWSampleImages = this.LFWSampleImages.concat(response);
          console.log('Recieved LFW Images with size' + this.LFWSampleImages.length)
          this.isLFWSampleImagesLoading = false;
        });
    }

  }

  toggleImageContainer(el: any) {
    if (el.id == "toggleTrainingImages") {
      if (this.showTrainingImages == true || !this.uploadedTrainingData) {
        return;
      }
      this.showTrainingImages = true;
    }
    else if (el.id == "toggleTestingImages") {
      if (this.showTrainingImages == false || !this.uploadedTestingData) {
        return;
      }
      this.showTrainingImages = false;
    }
  }

  openImageModal(base64Img: any, i: number, isTraining: boolean): void {
    // Simulate base64 image data (replace this with your actual base64 image data)
    // const base64Image = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...'; // Replace with your base64 image

    const dialogRef = this.dialog.open(ModalComponent, {
      width: '80%', // Adjust the width as needed
      height: '80%', // Adjust the height as needed
      data: { imageBase64: base64Img, idx: i , isTraining: isTraining} // Pass the base64 image data to the modal
    });
  }
  onStartProcessing() {
    
  }

}

// [ngClass]="[(!showTrainingImages) ? 'imageToggleDisabledEffect' : null]"