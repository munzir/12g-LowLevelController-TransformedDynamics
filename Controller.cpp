/*
 * Copyright (c) 2014-2016, Humanoid Lab, Georgia Tech Research Corporation
 * Copyright (c) 2014-2017, Graphics Lab, Georgia Tech Research Corporation
 * Copyright (c) 2016-2017, Personal Robotics Lab, Carnegie Mellon University
 * All rights reserved.
 *
 * This file is provided under the following "BSD-style" License:
 *   Redistribution and use in source and binary forms, with or
 *   without modification, are permitted provided that the following
 *   conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 *   CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *   MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *   USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 *   AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *   POSSIBILITY OF SUCH DAMAGE.
 */

#include "Controller.hpp"
#include <nlopt.hpp>
#include <string>

//==========================================================================
Controller::Controller(dart::dynamics::SkeletonPtr _robot,
                       dart::dynamics::BodyNode* _LeftendEffector,
                       dart::dynamics::BodyNode* _RightendEffector)
  : mRobot(_robot),
    mLeftEndEffector(_LeftendEffector),
    mRightEndEffector(_RightendEffector)
   {
  assert(_robot != nullptr);
  assert(_LeftendEffector != nullptr);
  assert(_RightendEffector != nullptr);

  int dof = mRobot->getNumDofs();
  std::cout << "[controller] DoF: " << dof << std::endl;

  mForces.setZero(19);
  mKp.setZero();
  mKv.setZero();

  for (int i = 0; i < 3; ++i) {
    mKp(i, i) = 750.0;
    mKv(i, i) = 250.0;
  }

  mSteps = 0;

  mLWheel = mRobot->getBodyNode("LWheel");
  mRWheel = mRobot->getBodyNode("RWheel");
  
  qInit = mRobot->getPositions();

  Eigen::Vector3d bodyCOM = ( \
    mRobot->getMass()*mRobot->getCOM() - mLWheel->getMass()*mLWheel->getCOM() - mRWheel->getMass()*mLWheel->getCOM()) \
    /(mRobot->getMass() - mLWheel->getMass() - mRWheel->getMass());
  zCOMInit = bodyCOM(2) - qInit(5);
  // Remove position limits
  for(int i = 6; i < dof-1; ++i)
    _robot->getJoint(i)->setPositionLimitEnforced(false);
  std::cout << "Position Limit Enforced set to false" << std::endl;

  // Set joint damping
  for(int i = 6; i < dof-1; ++i)
    _robot->getJoint(i)->setDampingCoefficient(0, 0.5);
  std::cout << "Damping coefficients set" << std::endl;

  dqFilt = new filter(25, 100);
}

//=========================================================================
Controller::~Controller() {}
//=========================================================================
struct OptParams {
  Eigen::MatrixXd P;
  Eigen::VectorXd b;
};

//=========================================================================
void printMatrix(Eigen::MatrixXd A){
  for(int i=0; i<A.rows(); i++){
    for(int j=0; j<A.cols(); j++){
      std::cout << A(i,j) << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}
//========================================================================
void constraintFunc(unsigned m, double *result, unsigned n, const double* x, double* grad, void* f_data) {

  OptParams* constParams = reinterpret_cast<OptParams *>(f_data);
  //std::cout << "done reading optParams " << std::endl;
  
  if (grad != NULL) {
    for(int i=0; i<m; i++) {
      for(int j=0; j<n; j++){
        grad[i*n+j] = constParams->P(i, j);
      }
    }
  }
  // std::cout << "done with gradient" << std::endl;

  Eigen::Matrix<double, 30, 1> X;
  for(size_t i=0; i<n; i++) X(i) = x[i];
  //std::cout << "done reading x" << std::endl;
  
  Eigen::VectorXd mResult;
  mResult = constParams->P*X - constParams->b;
  for(size_t i=0; i<m; i++) {
    result[i] = mResult(i);
  }
  // std::cout << "done calculating the result"
}

//========================================================================
double optFunc(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data) {
  OptParams* optParams = reinterpret_cast<OptParams *>(my_func_data);
  //std::cout << "done reading optParams " << std::endl;
  Eigen::Matrix<double, 30, 1> X(x.data());
  //std::cout << "done reading x" << std::endl;
  
  if (!grad.empty()) {
    Eigen::Matrix<double, 30, 1> mGrad = optParams->P.transpose()*(optParams->P*X - optParams->b);
    //std::cout << "done calculating gradient" << std::endl;
    Eigen::VectorXd::Map(&grad[0], mGrad.size()) = mGrad;
    //std::cout << "done changing gradient cast" << std::endl;
  }
  //std::cout << "about to return something" << std::endl;
  return (0.5 * pow((optParams->P*X - optParams->b).norm(), 2));
}

//=========================================================================
void Controller::update(const Eigen::Vector3d& _LefttargetPosition,const Eigen::Vector3d& _RighttargetPosition) {

  using namespace dart;
  using namespace std;  
  const int dof = (const int)mRobot->getNumDofs();
  const int nConstraints = 5;
  Eigen::VectorXd q = mRobot->getPositions();
  Eigen::VectorXd dqUnFilt    = mRobot->getVelocities();                // n x 1
  dqFilt->AddSample(dqUnFilt);
  Eigen::VectorXd dq = dqFilt->average;
  // cout << "The size of q = " << q.rows() << "*" << q.cols() << endl;
  double wEER = 0.01, wEEL = 0.01, wSpeedReg = 0.001, wReg = 0.001, wPose = 0.01, wxdotReg = 0.01, wpsidotReg = 0.01; 
  Eigen::DiagonalMatrix<double, 3> wBal(10.0, 0.0, 1.0); //(-5 0 0)
  double KpxCOM = 15.0, KvxCOM = 2.0;
  double KvSpeedReg = 10; // Speed Reg
  double KpPose = 10.0, KvPose = 20.0;
  double kdxdotReg = 20;
  double kpsidotReg = 250;
  Eigen::Matrix<double, 4, 4> baseTf = mRobot->getBodyNode(0)->getTransform().matrix();
  Eigen::Vector3d xyz0 = q.segment(3,3); // position of frame 0 in the world frame represented in the world frame
  Eigen::Vector3d dxyz0 = baseTf.matrix().block<3,3>(0,0)*dq.segment(3,3); // velocity of frame 0 in the world frame represented in the world frame


  // increase the step counter
  mSteps++;
  //cout << mSteps << endl;

  // Rotation Transform of Frame 0
  double psi =  atan2(baseTf(0,0), -baseTf(1,0));
  Eigen::Transform<double, 3, Eigen::Affine> Tf0 = Eigen::Transform<double, 3, Eigen::Affine>::Identity();
  Tf0.rotate(Eigen::AngleAxisd(psi, Eigen::Vector3d::UnitZ()));
  Eigen::Matrix<double, 3, 3> Rot0 = Tf0.matrix().block<3, 3>(0, 0).transpose();
  if(mSteps==1){
  cout << "Correct Rot0:" << endl;
  for(int i=0; i<3; i++) { for(int j=0; j<3; j++) { cout << Rot0(i,j) << ", "; } cout << endl;  }
    cout << "Our Rot0:" << endl;
    cout << cos(psi) << ", " << sin(psi) << ", " << "0" << endl;
    cout << -sin(psi) << ", " << cos(psi) << ", " << "0" << endl;
    cout << "0, 0, 1" << endl;
  }

  // Derivative of Rot0
  double dpsi = 0;//(baseTf.block<3,3>(0,0) * dq.head(3))(2);
  Eigen::Matrix<double, 3, 3> dRot0;
  dRot0 << (-sin(psi)*dpsi), (cos(psi)*dpsi), 0,
           (-cos(psi)*dpsi), (-sin(psi)*dpsi), 0,
           0, 0, 0;
  
  // xEEref
  Eigen::VectorXd xEErefL = xyz0 + Rot0.transpose()*_LefttargetPosition;
  Eigen::VectorXd xEErefR = xyz0 + Rot0.transpose()*_RighttargetPosition;
  if(mSteps == 1) { 
    cout << "xEErefL: " << xEErefL(0) << ", " << xEErefL(1) << ", " << xEErefL(2) << endl;
    cout << "xEErefR: " << xEErefR(0) << ", " << xEErefR(1) << ", " << xEErefR(2) << endl;
   }
  
  // ********************************* Left arm
  // Zero Columns
  Eigen::Vector3d zeroCol(0.0, 0.0, 0.0);
  Eigen::Matrix<double, 3, 7> zero7Col;
  zero7Col << zeroCol, zeroCol, zeroCol, zeroCol, zeroCol, zeroCol, zeroCol;
  

  // x, dx, ddxref
  Eigen::Vector3d xEEL = mLeftEndEffector->getTransform().translation();
  Eigen::Vector3d dxEEL = mLeftEndEffector->getLinearVelocity();
  Eigen::Vector3d ddxEELref = -mKp*(xEEL - xEErefL) - mKv*dxEEL;

  // Jacobian
  math::LinearJacobian JEEL_small = mLeftEndEffector->getLinearJacobian(); 
  Eigen::Matrix<double, 3, 25> JEEL;
  JEEL << JEEL_small.block<3,6>(0,0), zeroCol, zeroCol, JEEL_small.block<3,2>(0,6), zeroCol, JEEL_small.block<3,7>(0,8), zero7Col;
  
  // Jacobian Derivative
  math::LinearJacobian dJEEL_small = mLeftEndEffector->getLinearJacobianDeriv(); 
  Eigen::Matrix<double, 3, 25> dJEEL;
  dJEEL << dJEEL_small.block<3,6>(0,0), zeroCol, zeroCol, dJEEL_small.block<3,2>(0,6), zeroCol, dJEEL_small.block<3,7>(0,8), zero7Col;
  
  // P and b
  Eigen::Matrix<double, 3, 30> PEEL;
  PEEL << wEEL*JEEL, zeroCol, zeroCol, zeroCol, zeroCol, zeroCol;
  Eigen::VectorXd bEEL = -wEEL*(dJEEL*dq - ddxEELref);
  
  //*********************************** Right Arm 
  // x, dx, ddxref
  Eigen::Vector3d xEER = mRightEndEffector->getTransform().translation();
  Eigen::Vector3d dxEER = mRightEndEffector->getLinearVelocity();
  Eigen::Vector3d ddxEERref = -mKp*(xEER - xEErefR) - mKv*dxEER;

  // Jacobian
  math::LinearJacobian JEER_small = mRightEndEffector->getLinearJacobian(); 
  Eigen::Matrix<double, 3, 25> JEER;
  JEER << JEER_small.block<3,6>(0,0), zeroCol, zeroCol, JEER_small.block<3,2>(0,6), zeroCol, zero7Col, JEER_small.block<3,7>(0,8);
  
  // Jacobian Derivative
  math::LinearJacobian dJEER_small = mRightEndEffector->getLinearJacobianDeriv(); 
  Eigen::Matrix<double, 3, 25> dJEER;
  dJEER << dJEER_small.block<3,6>(0,0), zeroCol, zeroCol, dJEER_small.block<3,2>(0,6), zeroCol, zero7Col, dJEER_small.block<3,7>(0,8);
  
  // P and b
  Eigen::Matrix<double, 3, 30> PEER;
  PEER << wEER*JEER, zeroCol, zeroCol, zeroCol, zeroCol, zeroCol;
  Eigen::VectorXd bEER = -wEER*(dJEER*dq - ddxEERref);

  
  //*********************************** Balance
  // Excluding wheels from COM Calculation
  Eigen::Vector3d bodyCOM = mRobot->getCOM();
  Eigen::Vector3d bodyCOMLinearVelocity = mRobot->getCOMLinearVelocity();
  
  // x, dx, ddxref
  Eigen::Vector3d COM = Rot0*(mRobot->getCOM()-xyz0);
  Eigen::Vector3d dCOM = Rot0*(mRobot->getCOMLinearVelocity()-dxyz0);;
  Eigen::Vector3d zCOMInitVec; zCOMInitVec << 0, 0, zCOMInit;
  Eigen::Vector3d ddCOMref = -KpxCOM*(COM-zCOMInitVec) - KvxCOM*dCOM;
  
  // Jacobian
  Eigen::MatrixXd JCOM = Rot0*mRobot->getCOMLinearJacobian(); 
  
  // Jacobian Derivative
  Eigen::MatrixXd dJCOM = Rot0*mRobot->getCOMLinearJacobianDeriv(); 

  // P and b 
  Eigen::Matrix<double, 3, 30> PBal;
  PBal << wBal*JCOM, zeroCol, zeroCol, zeroCol, zeroCol, zeroCol;
  for(int i=0; i<30; i++) PBal(1,i) = 0;
  Eigen::Matrix<double, 3, 1> bBal;
  bBal << (wBal*(-dJCOM*dq + ddCOMref));
  
  // ***************************** Pose
  Eigen::MatrixXd wMatPose = Eigen::MatrixXd::Identity(30, 30);
  wMatPose(0,0) = 10*wPose; // Base Link Pitch
  for(int i=1; i<6; i++) wMatPose(i, i) = 0; // Base Link other speeds + wheels
  for(int i=6; i<8; i++) wMatPose(i, i) = 0; // Base Link other speeds + wheels
  for(int i=8; i<10; i++) wMatPose(i, i) = 10*wPose; // Waist + Torso
  for(int i=10; i<25; i++) wMatPose(i, i) = wPose/1.0; // Other Upper Body Joints
  for(int i=25; i<30; i++) wMatPose(i, i) = 0.0; // Lambdas
  Eigen::MatrixXd PPose;
  PPose = wMatPose;
  Eigen::Matrix<double, 30, 1> bPose; 
  bPose << (-KpPose*(q - qInit) - KvPose*dq), 0, 0, 0, 0, 0;
  bPose = wMatPose*bPose;
  
  
  // ***************************** Speed Regulator
  Eigen::MatrixXd wMatSpeedReg = Eigen::MatrixXd::Identity(30, 30);
  wMatSpeedReg(0,0) = 100*wSpeedReg; // Base Link Pitch
  for(int i=1; i<8; i++) wMatSpeedReg(i, i) = wSpeedReg;//wSpeedReg; // Base Link other speeds + wheels
  for(int i=8; i<10; i++) wMatSpeedReg(i, i) = 10*wSpeedReg; // Waist + Torso
  for(int i=10; i<25; i++) wMatSpeedReg(i, i) = wSpeedReg/1.0; // Other Upper Body Joints
  for(int i=25; i<30; i++) wMatSpeedReg(i, i) = 0.0; // Lambdas
  Eigen::MatrixXd PSpeedReg;
  PSpeedReg = wMatSpeedReg;
  Eigen::Matrix<double, 30, 1> bSpeedReg; 
  bSpeedReg << -KvSpeedReg*dq, 0, 0, 0, 0, 0;
  bSpeedReg = wMatSpeedReg*bSpeedReg;
  
  // ***************************** Regulator
  Eigen::MatrixXd wMatReg = Eigen::MatrixXd::Identity(30, 30);
  wMatReg(0,0) = 0; // Base Link Pitch
  for(int i=1; i<8; i++) wMatReg(i, i) = wReg; // Base Link other speeds + wheels
  for(int i=8; i<10; i++) wMatReg(i, i) = wReg; // Waist + Torso
  for(int i=10; i<25; i++) wMatReg(i, i) = 10*wReg; // Other Upper Body Joints
  for(int i=25; i<30; i++) wMatReg(i, i) = 0.0; // Lambdas
  Eigen::MatrixXd PReg;
  PReg = wMatReg;
  Eigen::Matrix<double, 30, 1> bReg = Eigen::VectorXd::Zero(30); 

  // **************************** Constraint Jacobian
  // Constraints:
  //  0. dZ0 = 0                                               
  //                                                              => dq_orig(4)*cos(qBody1) + dq_orig(5)*sin(qBody1) = 0
  //  1. da3 + R/L*(dthL - dthR) = 0                           
  //                                                              => dq_orig(1)*cos(qBody1) + dq_orig(2)*sin(qBody1) + R/L*(dq_orig(6) - dq_orig(7)) = 0 
  //  2. da1*cos(psii) + da2*sin(psii) = 0                     
  //                                                              => dq_orig(1)*sin(qBody1) - dq_orig(2)*cos(qBody1) = 0
  //  3. dX0*sin(psii) - dY0*cos(psii) = 0                     
  //                                                              => dq_orig(3) = 0
  //  4. dX0*cos(psii) + dY0*sin(psii) - R/2*(dthL + dthR) = 0 
  //                                                              => dq_orig(4)*sin(qBody1) - dq_orig(5)*cos(qBody1) - R/2*(dq_orig(6) + dq_orig(7) - 2*dq_orig(0)) = 0
  double R = 0.265, L = 0.68;
  double qBody1, dqBody1; 
  qBody1 = atan2(baseTf(0,1)*cos(psi) + baseTf(1,1)*sin(psi), baseTf(2,1));
  dqBody1 = -dq(0);
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(5, 25);
  J(0,4) = cos(qBody1); J(0,5) = sin(qBody1);
  J(1,1) = cos(qBody1); J(1,2) = sin(qBody1); J(1,6) = R/L; J(1,7) = -R/L;
  J(2,1) = sin(qBody1); J(2,2) = -cos(qBody1); 
  J(3,3) = 1;
  J(4,0) = R; J(4,4) = sin(qBody1); J(4,5) = -cos(qBody1); J(4,6) = -R/2; J(4,7) = -R/2; 


//*************************xdot Regulator
  Eigen::Matrix<double, 1, 19> zero19Rows;
  Eigen::Matrix<double, 1, 24> zero24Rows;
  zero19Rows = Eigen::MatrixXd::Zero(1, 19);
  zero24Rows = Eigen::MatrixXd::Zero(1, 24);
  Eigen::Matrix<double, 1, 30> PxdotReg;
  Eigen::Matrix<double, 1, 25> dJxdotReg;
  PxdotReg << 0,0,0,0, wxdotReg*sin(qBody1), wxdotReg*-cos(qBody1), zero24Rows;
  // cout << "PxdotReg = " << PxdotReg<<endl;
  dJxdotReg << 0,0,0,0, cos(qBody1)*dqBody1, sin(qBody1)*dqBody1, zero19Rows;
  // cout << "dJxdotReg = " << dJxdotReg<<endl;
  double xdotref = 0;
  
  Eigen::Matrix<double, 1, 1> bxdotReg;
  bxdotReg << -wxdotReg*(kdxdotReg*dJxdotReg*dq - xdotref);


  //********************************Psi Regulator
  Eigen::Matrix<double, 1, 22> zero22Rows;
  Eigen::Matrix<double, 1, 27> zero27Rows;
  zero22Rows = Eigen::MatrixXd::Zero(1, 22);
  zero27Rows = Eigen::MatrixXd::Zero(1, 27);
  Eigen::Matrix<double, 1, 30> PpsidotReg;
  Eigen::Matrix<double, 1, 25> dJpsidotReg;
  PpsidotReg << 0,wxdotReg*cos(qBody1), wxdotReg*sin(qBody1), zero27Rows;
  // cout << "PxdotReg = " << PxdotReg<<endl;
  dJpsidotReg << 0,-sin(qBody1)*dqBody1, cos(qBody1)*dqBody1, zero22Rows;
  // cout << "dJxdotReg = " << dJxdotReg<<endl;
  double psidotref = 0;
  
  Eigen::Matrix<double, 1, 1> bpsidotReg;
  bxdotReg << -wpsidotReg*(kpsidotReg*dJxdotReg*dq - psidotref);


  // ***************************** Inertia and Coriolis Matrices
  Eigen::MatrixXd M = mRobot->getMassMatrix();
  Eigen::VectorXd h = mRobot->getCoriolisAndGravityForces();
  // cout << "Size of M = " << M.rows() << "*" << M.cols() << endl;
  // cout << "Size of h = " << h.rows() << "*" << h.cols() << endl;

//*******************************Torque Limits
  Eigen::Matrix<double, 19, 1> T_ub;
  Eigen::Matrix<double, 19, 1> T_lb;
  T_ub << 60, 60, 740, 370, 10, 370, 370, 175, 175, 40, 40, 9.5, 370, 370, 175, 175, 40, 40, 9.5;
  T_lb << -T_ub; 


  // ***************************** QP
  OptParams optParams;
  if(mSteps == 1) {
    cout << "PEER: " << PEER.rows() << " x " << PEER.cols() << endl;
    cout << "PEEL: " << PEEL.rows() << " x " << PEEL.cols() << endl;
    cout << "PBal: " << PBal.rows() << " x " << PBal.cols() << endl;
    cout << "PPose: " << PPose.rows() << " x " << PPose.cols() << endl;
    cout << "PSpeedReg: " << PSpeedReg.rows() << " x " << PSpeedReg.cols() << endl;
    cout << "PReg: " << PReg.rows() << " x " << PReg.cols() << endl;
    cout << "PxdotReg: " << PxdotReg.rows() << " x " << PxdotReg.cols() << endl;
    cout << "bEER: " << bEER.rows() << " x " << bEER.cols() << endl;
    cout << "bEEL: " << bEEL.rows() << " x " << bEEL.cols() << endl;
    cout << "bBal: " << bBal.rows() << " x " << bBal.cols() << endl;
    cout << "bPose: " << bPose.rows() << " x " << bPose.cols() << endl;
    cout << "bSpeedReg: " << bSpeedReg.rows() << " x " << bSpeedReg.cols() << endl;
    cout << "bReg: " << bReg.rows() << " x " << bReg.cols() << endl;
    cout << "bxdotReg: " << bxdotReg.rows() << " x " << bxdotReg.cols() << endl;
  }

  Eigen::MatrixXd P(PEER.rows() + PEEL.rows() + PBal.rows() + PPose.rows() + PSpeedReg.rows() + PReg.rows() + PxdotReg.rows() + PpsidotReg.rows() , PEER.cols() );
  P << PEER,
       PEEL,
       PBal,
       PPose,
       PSpeedReg,
       PReg,
       PxdotReg,
       PpsidotReg;

  Eigen::VectorXd b(bEER.rows() + bEEL.rows() + bBal.rows() + bPose.rows() + bSpeedReg.rows() + bReg.rows() + bxdotReg.rows() + bpsidotReg.rows(), bEER.cols() );
  b << bEER,
       bEEL,
       bBal,
       bPose,
       bSpeedReg,
       bReg,
       bxdotReg,
       bpsidotReg;

  optParams.P = P;
  optParams.b = b;
  

  OptParams constraintParams[2];
  Eigen::Matrix<double, 6, 30> P_;
  Eigen::Matrix<double, 6, 1> b_;
  P_ << M.block<6,25>(0, 0), (-J.block<5, 6>(0, 0).transpose());
  b_ << -h.head(6);
  constraintParams[0].P = P_;
  constraintParams[0].b = b_;
  constraintParams[1].P = -P_;
  constraintParams[1].b = -b_;


  OptParams inequalityconstraintParams[2];
  Eigen::Matrix<double, 19, 30> P1_;
  Eigen::Matrix<double, 19, 30> P2_;
  Eigen::Matrix<double, 19, 1> b1_;
  Eigen::Matrix<double, 19, 1> b2_;
  P1_ << M.block<19, 25>(6,0), -J.block<5, 19>(0,6).transpose();
  P2_ << -M.block<19, 25>(6,0), J.block<5, 19>(0,6).transpose();
  b1_ << -h.tail(19) + T_ub;
  b2_ << h.tail(19) - T_lb;

  const vector<double> constraintTol(6, 1e-3);
  const vector<double> lb(30, -10);
  const vector<double> ub(30, 10);

  const vector<double> inequalityconstraintTol(19, 1e-3);
  inequalityconstraintParams[0].P = P1_;
  inequalityconstraintParams[0].b = b1_;
  inequalityconstraintParams[1].P = P2_;
  inequalityconstraintParams[1].b = b2_;

  //nlopt::opt opt(nlopt::LN_COBYLA, 30);
  nlopt::opt opt(nlopt::LD_SLSQP, 30);
  double minf;
  opt.set_min_objective(optFunc, &optParams);
  opt.add_inequality_mconstraint(constraintFunc, &inequalityconstraintParams[0], inequalityconstraintTol);
  opt.add_inequality_mconstraint(constraintFunc, &inequalityconstraintParams[1], inequalityconstraintTol);
  opt.add_equality_mconstraint(constraintFunc, &constraintParams[1], constraintTol);
  //opt.set_lower_bounds(lb);
  //opt.set_upper_bounds(ub);
  opt.set_xtol_rel(1e-3);
  int maxtimeSet = 0;
  //opt.set_maxtime(0.01); int maxtimeSet = 1;
  vector<double> ddq_lambda_vec(30);
  Eigen::VectorXd::Map(&ddq_lambda_vec[0], ddq_lambda.size()) = ddq_lambda;
  opt.optimize(ddq_lambda_vec, minf);
  Eigen::Matrix<double, 30, 1> ddq_lambda(ddq_lambda_vec.data());
  if(mSteps < 0) {
    cout << "ddq_lambda: " << endl; for(int i=0; i<30; i++) {cout << ddq_lambda(i) << ", ";} cout << endl;
    cout << "ddq_lambda_vec: " << endl; for(int i=0; i<30; i++) {cout << ddq_lambda_vec[i] << ", ";} cout << endl;
  }

  // Torques
  mForces << (M.block<19, 25>(6,0)*ddq_lambda.head(25) + h.tail(19) - (J.block<5, 19>(0,6).transpose())*ddq_lambda.tail(5));
  if(mSteps%(maxtimeSet==1?30:30) == 0) {
    cout << "mForces: " << mForces(0);
    for(int i=1; i<19; i++){ 
      cout << ", " << mForces(i); 
    }
    cout << endl;
    // print wheel rows of M
    cout << "M6: "; for(int i=0; i<25; i++) { cout << M(6, i) << ", "; } cout << endl;
    cout << "M7: "; for(int i=0; i<25; i++) { cout << M(7, i) << ", "; } cout << endl;
    // print ddq
    cout << "ddq: "; for(int i=0; i<25; i++) { cout << ddq_lambda(i) << ", "; } cout << endl;
    // print M*ddq for wheel rows
    cout << "M6*ddq: "<< (M.block<1,25>(6,0)*ddq_lambda.head(25)) << endl;
    cout << "M7*ddq: "<< (M.block<1,25>(7,0)*ddq_lambda.head(25)) << endl;
    // print h for wheel rows
    cout << "h6: " << h(6) << endl;
    cout << "h7: " << h(7) << endl;
    // print wheel rows of J'
    cout << "J6: "; for(int i=0; i<5; i++) { cout << J(i, 6) << ", "; } cout << endl;
    cout << "J7: "; for(int i=0; i<5; i++) { cout << J(i, 7) << ", "; } cout << endl;
    // print lambdas
    cout << "lambda: "; for(int i=0; i<5; i++) { cout << ddq_lambda(25+i) << ", "; } cout << endl;
    // print J'*lambda for wheel rows
    cout << "J6*lambda: " << (J.block<5,1>(0,6).transpose()*ddq_lambda.tail(5)) << endl;
    cout << "J7*lambda: " << (J.block<5,1>(0,7).transpose()*ddq_lambda.tail(5)) << endl;
    // Print the objective function components 
    cout << "EEL loss: " << pow((PEEL*ddq_lambda-bEEL).norm(), 2) << endl;
    cout << "EER loss: " << pow((PEER*ddq_lambda-bEER).norm(), 2) << endl;
    cout << "Bal loss: " << pow((PBal*ddq_lambda-bBal).norm(), 2) << endl;
    cout << "Pose loss: " << pow((PPose*ddq_lambda-bPose).norm(), 2) << endl;
    cout << "Speed Reg loss: " << pow((PSpeedReg*ddq_lambda-bSpeedReg).norm(), 2) << endl;
    cout << "Reg loss: " << pow((PReg*ddq_lambda-bReg).norm(), 2) << endl;
    cout << "xdot loss: " << pow((PxdotReg*ddq_lambda-bxdotReg).norm(), 2) << endl;
    cout << "psidot loss: " << pow((PpsidotReg*ddq_lambda-bpsidotReg).norm(), 2) << endl;
    cout << "Equality: "; for(int i=0; i<6; i++) {cout << (P_*ddq_lambda-b_)(i) << ", ";} cout << endl << endl << endl;
  }
  const vector<size_t > index{6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
  mRobot->setForces(index, mForces);
}

//=========================================================================
dart::dynamics::SkeletonPtr Controller::getRobot() const {
  return mRobot;
}

//=========================================================================
dart::dynamics::BodyNode* Controller::getEndEffector(const std::string &s) const {
  if (s.compare("left")) {  return mLeftEndEffector; }
  else if (s.compare("right")) { return mRightEndEffector; }
}

//=========================================================================
void Controller::keyboard(unsigned char /*_key*/, int /*_x*/, int /*_y*/) {
}
