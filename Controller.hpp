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

#ifndef EXAMPLES_OPERATIONALSPACECONTROL_CONTROLLER_HPP_
#define EXAMPLES_OPERATIONALSPACECONTROL_CONTROLLER_HPP_

#include <Eigen/Eigen>
#include <string>
#include <dart/dart.hpp>
#include <boost/circular_buffer.hpp>

class filter {
  public:
    filter(const int dim, const int n)
    {
      samples.set_capacity(n);
      total = Eigen::VectorXd::Zero(dim,1);
    }
    void AddSample(Eigen::VectorXd v)
    {
      if(samples.full()) 
      {
        total -= samples.front();
      }
      samples.push_back(v);
      total += v;
      average = total/samples.size();
    }
  
    boost::circular_buffer<Eigen::VectorXd> samples;
    Eigen::VectorXd total;
    Eigen::VectorXd average;
    
};

/// \brief Operational space controller for 6-dof manipulator
class Controller {
public:
  /// \brief Constructor
  Controller( dart::dynamics::SkeletonPtr _robot,
              dart::dynamics::BodyNode* _LeftendEffector,
              dart::dynamics::BodyNode* _RightendEffector);

  /// \brief Destructor
  virtual ~Controller();

  /// \brief
  void update(const Eigen::Vector3d& _LefttargetPosition,const Eigen::Vector3d& _RighttargetPosition);

  /// \brief Get robot
  dart::dynamics::SkeletonPtr getRobot() const;

  /// \brief Get end effector of the robot
  dart::dynamics::BodyNode* getEndEffector(const std::string &s) const;

  /// \brief Keyboard control
  virtual void keyboard(unsigned char _key, int _x, int _y);

//private:
  /// \brief Robot
  dart::dynamics::SkeletonPtr mRobot;

  /// \brief Left End-effector of the robot
  dart::dynamics::BodyNode* mLeftEndEffector;

  /// \brief Right End-effector of the robot
  dart::dynamics::BodyNode* mRightEndEffector;

  dart::dynamics::BodyNode* mLWheel;
  dart::dynamics::BodyNode* mRWheel;

  /// \brief Control forces
  Eigen::Matrix<double, 19, 1> mForces;

  /// \brief Proportional gain for the virtual spring forces at the end effector
  Eigen::Matrix3d mKp;

  /// \brief Derivative gain for the virtual spring forces at the end effector
  Eigen::Matrix3d mKv;

  size_t mSteps;

  Eigen::Matrix<double, 30, 1> ddq_lambda;

  double zCOMInit;

  Eigen::Matrix<double, 25, 1> qInit;

  filter *dqFilt;
};

#endif  // EXAMPLES_OPERATIONALSPACECONTROL_CONTROLLER_HPP_
