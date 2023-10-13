import { Component, OnInit } from '@angular/core';
import { HttpClient, HttpErrorResponse, HttpHeaders } from '@angular/common/http';
import { catchError, throwError } from 'rxjs';

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.css']
})
export class DashboardComponent implements OnInit {

  selectedFiles: FileList | undefined;

  onDragOverFlag: boolean = false;
  firstImgUploaded: boolean = false;
  successfulUpload: boolean = false;
  lastError: HttpErrorResponse | null = null;
  standBy: boolean = false;

  constructor(private http: HttpClient) {}

  ngOnInit(): void {}

  uploadedFiles: { file: File; dataURL: string }[] = [];

  onDragOver(event: Event): void {
    this.onDragOverFlag = true;
    event.preventDefault();
  }

  onDragLeave(event: Event): void {
    this.onDragOverFlag = false;
    event.preventDefault();
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    this.onDragOverFlag = false;

    /* ensure the event has files that can be received */
    if (event != null && event.dataTransfer != null) {
      this.handleFiles(event.dataTransfer.files);
    }
      
  }

  handleFiles(files: FileList | null): void {
    if (files == null) {
      console.error("handleFiles did not received type FileList");
      return;
    }

    if (files.length > 1) {
      console.error("Please only give one image");
      return;
    }

    this.standBy = true;

    this.uploadedFiles = [];
    const file: File = files.item(0)!;
    const reader = new FileReader();
    reader.onload = (event) => {
      const dataURL = event.target?.result as string;
      this.uploadedFiles.push({ file, dataURL });
    };
    reader.readAsDataURL(file);

    const formData = new FormData();
    formData.append('file', file);

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

}
