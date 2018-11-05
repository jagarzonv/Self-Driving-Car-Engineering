#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
	x_ = F_ * x_;
	MatrixXd Ft = F_.transpose();
	P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::UpdateCommon(const VectorXd &y){


	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
	VectorXd z_pred = H_ * x_;
	VectorXd y = z - z_pred;
    UpdateCommon(y);

}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */
	double rho = sqrt(x_(0)*x_(0) + x_(1)*x_(1));
	double phi =0;
	 if (fabs(x_(0)) > 0.001) {
	    phi = atan2(x_(1) , x_(0));
	 }
	double rhodot =0;
	 if (fabs(rho) > 0.001) {
	    rhodot = (x_(0)*x_(2) + x_(1)*x_(3)) / rho;
	 }
	VectorXd z_pred = VectorXd(3);
	z_pred << rho, phi, rhodot;
	VectorXd y = z - z_pred;
	int n=1;
	//n=abs(trunc(y[1]/M_PI));
	if (y[1]>M_PI){
		y[1]=y[1]-n*(2*M_PI);
	}else if (y[1]<-M_PI){
		y[1]=y[1]+n*(2*M_PI);
	}
	UpdateCommon(y);

}
