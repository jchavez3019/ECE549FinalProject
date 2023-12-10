import { Component, Inject } from '@angular/core';
import { MAT_DIALOG_DATA, MatDialogRef } from '@angular/material/dialog';
import { HttpClient, HttpErrorResponse, HttpHeaders } from '@angular/common/http';
import { catchError, throwError } from 'rxjs';

@Component({
  selector: 'app-modal',
  templateUrl: 'modal.component.html',
  styleUrls: ['modal.component.css']
})
export class ModalComponent {

  faceImages: any[]= [];

  constructor(
    public dialogRef: MatDialogRef<ModalComponent>,
    @Inject(MAT_DIALOG_DATA) public data: any,
    private http: HttpClient
  ) {

    if (data.isTraining) {
      this.http.get<any>('https://127.0.0.1:5000/get-training-image-faces?startIndex=' + data.idx)
        .subscribe((response: any) => {
          this.faceImages = this.faceImages.concat(response);
          console.log(`There are ${this.faceImages.length} face images`);
        });
    }
    else {
      this.http.get<any>('https://127.0.0.1:5000/get-testing-image-faces?startIndex=' + data.idx)
        .subscribe((response: any) => {
          this.faceImages = this.faceImages.concat(response);
          console.log(`There are ${this.faceImages.length} face images`);
        });
    }

    
  }

  close(): void {
    this.dialogRef.close();
  }
}
