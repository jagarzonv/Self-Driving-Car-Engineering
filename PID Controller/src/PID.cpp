#include "PID.h"
#include <cmath>
#include <limits>
#include <iostream>

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {

	this->Kp = 0;
	this->Ki = 0;
	this->Kd = 0;
	this->p_error=0;
	this->d_error=0;
	this->i_error=0;
	this->prev_cte=0;
	this->sum_cte=0;
}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {

    this->Kp = Kp;
    this->Ki = Ki;
    this->Kd = Kd;
    sum_cte = 0;
    prev_cte = 0;

}

void PID::UpdateError(double cte) {

    sum_cte +=  cte;
    p_error = - Kp * cte;
    i_error = - Ki * sum_cte;
    d_error = - Kd * (cte - prev_cte);
    prev_cte = cte;

}

double PID::TotalError() {

double tot_err=p_error + i_error + d_error;

return tot_err;

}
