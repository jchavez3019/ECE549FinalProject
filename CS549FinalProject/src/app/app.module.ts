import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';
import { RouterModule } from '@angular/router';
import { SocketIoModule, SocketIoConfig } from 'ngx-socket-io';

import { AppComponent } from './app.component';
import { DashboardComponent } from './components/dashboard/dashboard.component';
import { NavBarComponent } from './components/nav-bar/nav-bar.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';

/* routes */
import { appRoutes } from './routes';

/* Material imports */
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatIconModule } from '@angular/material/icon';
import { MatCardModule } from '@angular/material/card';
import { DragDropModule } from '@angular/cdk/drag-drop';
import { MatButtonModule } from '@angular/material/button';
import { MatDialogModule } from '@angular/material/dialog';
import { MatInputModule } from '@angular/material/input';
import { MatExpansionModule } from '@angular/material/expansion';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatSnackBar } from '@angular/material/snack-bar';
import { MatListModule } from '@angular/material/list';
import { DescriptionComponent } from './components/description/description.component';
import { AboutUsComponent } from './components/about-us/about-us.component';
import { MatGridListModule } from '@angular/material/grid-list';

import { ScrollingModule } from '@angular/cdk/scrolling';
import { ModalComponent } from './components/dashboard/modal/modal.component';
import { LfwComponent } from './components/lfw/lfw.component';

const config: SocketIoConfig = { url: 'https://127.0.0.1:5000/', options: {} };

@NgModule({
  declarations: [
    AppComponent,
    DashboardComponent,
    NavBarComponent,
    DescriptionComponent,
    AboutUsComponent,
    ModalComponent,
    LfwComponent
  ],
  imports: [
    BrowserModule,
    SocketIoModule.forRoot(config),
    RouterModule.forRoot(appRoutes),
    BrowserAnimationsModule,
    HttpClientModule,
    MatToolbarModule,
    MatIconModule,
    MatCardModule,
    DragDropModule,
    MatButtonModule,
    MatDialogModule,
    MatInputModule,
    MatExpansionModule,
    MatTooltipModule,
    MatListModule,
    FormsModule,
    ReactiveFormsModule, 
    ScrollingModule,
    MatGridListModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
