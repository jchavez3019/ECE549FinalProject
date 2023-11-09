import { Component, ElementRef, OnInit, ViewChild } from '@angular/core';
import { HttpClient, HttpErrorResponse, HttpHeaders } from '@angular/common/http';
import { catchError, throwError } from 'rxjs';

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
  uploadedTestingData: boolean = false;
  firstImgUploaded: boolean = false;
  successfulUpload: boolean = false;
  lastError: HttpErrorResponse | null = null;
  standBy: boolean = false;

  /* training data set */
  trainingDataFiles: { file: File; dataURL: string }[] = [];

  /* testing data set */
  testingDataFiles: { file: File; dataURL: string }[] = [];

  constructor(private http: HttpClient) {}

  ngOnInit(): void {}

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
      this.trainingDataFiles = uploadedFiles;
      formData.append('isTrainingData', 'true');
      this.uploadedTrainingData = true;
    }
    else {
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

    });
    
  }

  onStartProcessing() {
    
  }

}
