import { Component } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-nav-bar',
  templateUrl: './nav-bar.component.html',
  styleUrls: ['./nav-bar.component.css']
})
export class NavBarComponent {
  constructor(private router: Router) {}

  /* navigates to the other sections based on the selection */
  navigateTo(el: any) {
    switch(el.id) {
      case 'navToDashboard':
        this.router.navigate(['dashboard']);
        break;
      case 'navToDescription':
        this.router.navigate(['description']);
        break;
      case 'navToAboutUs':
        this.router.navigate(['about-us']);
        break;
    }
  }
}
