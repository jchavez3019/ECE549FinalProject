import { ComponentFixture, TestBed } from '@angular/core/testing';

import { LfwComponent } from './lfw.component';

describe('LfwComponent', () => {
  let component: LfwComponent;
  let fixture: ComponentFixture<LfwComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [LfwComponent]
    });
    fixture = TestBed.createComponent(LfwComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
