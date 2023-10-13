import { Routes } from '@angular/router';

/* generated components */
import { DashboardComponent } from './components/dashboard/dashboard.component';
import { DescriptionComponent } from './components/description/description.component';
import { AboutUsComponent } from './components/about-us/about-us.component';

/* 
    The 4th entry is a wildcard where any url that 
    doesn't match other entries just redirects to the login page 
*/
export const appRoutes: Routes = [
    { path: 'dashboard', component: DashboardComponent},
    { path: 'description', component: DescriptionComponent}, 
    { path: 'about-us', component: AboutUsComponent},
    { path: '', redirectTo: '/dashboard', pathMatch: 'full' }
];