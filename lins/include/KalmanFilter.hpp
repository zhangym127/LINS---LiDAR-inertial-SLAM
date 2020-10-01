// This file is part of LINS.
//
// Copyright (C) 2020 Chao Qin <cscharlesqin@gmail.com>,
// Robotics and Multiperception Lab (RAM-LAB <https://ram-lab.com>),
// The Hong Kong University of Science and Technology
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.

#ifndef INCLUDE_KALMANFILTER_HPP_
#define INCLUDE_KALMANFILTER_HPP_

#include <math_utils.h>
#include <parameters.h>

#include <iostream>
#include <map>

using namespace std;
using namespace math_utils;
using namespace parameter;

namespace filter {

// GlobalState Class contains state variables including position, velocity,
// attitude, acceleration bias, gyroscope bias, and gravity
class GlobalState {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static constexpr unsigned int DIM_OF_STATE_ = 18;
  static constexpr unsigned int DIM_OF_NOISE_ = 12;
  static constexpr unsigned int pos_ = 0;
  static constexpr unsigned int vel_ = 3;
  static constexpr unsigned int att_ = 6;
  static constexpr unsigned int acc_ = 9;
  static constexpr unsigned int gyr_ = 12;
  static constexpr unsigned int gra_ = 15;

  GlobalState() { setIdentity(); }

  GlobalState(const V3D& rn, const V3D& vn, const Q4D& qbn, const V3D& ba,
              const V3D& bw) {
    setIdentity();
    rn_ = rn;
    vn_ = vn;
    qbn_ = qbn;
    ba_ = ba;
    bw_ = bw;
  }

  ~GlobalState() {}

  void setIdentity() {
    rn_.setZero();
    vn_.setZero();
    qbn_.setIdentity();
    ba_.setZero();
    bw_.setZero();
    gn_ << 0.0, 0.0, -G0;
  }

  /* 完成ESKF的⊕操作 */
  // boxPlus operator
  void boxPlus(const Eigen::Matrix<double, DIM_OF_STATE_, 1>& xk,
               GlobalState& stateOut) {
    stateOut.rn_ = rn_ + xk.template segment<3>(pos_);
    stateOut.vn_ = vn_ + xk.template segment<3>(vel_);
    stateOut.ba_ = ba_ + xk.template segment<3>(acc_);
    stateOut.bw_ = bw_ + xk.template segment<3>(gyr_);
    Q4D dq = axis2Quat(xk.template segment<3>(att_));
    stateOut.qbn_ = (qbn_ * dq).normalized();

    stateOut.gn_ = gn_ + xk.template segment<3>(gra_);
  }

  // boxMinus operator
  void boxMinus(const GlobalState& stateIn,
                Eigen::Matrix<double, DIM_OF_STATE_, 1>& xk) {
    xk.template segment<3>(pos_) = rn_ - stateIn.rn_;
    xk.template segment<3>(vel_) = vn_ - stateIn.vn_;
    xk.template segment<3>(acc_) = ba_ - stateIn.ba_;
    xk.template segment<3>(gyr_) = bw_ - stateIn.bw_;
    V3D da = Quat2axis(stateIn.qbn_.inverse() * qbn_);
    xk.template segment<3>(att_) = da;

    xk.template segment<3>(gra_) = gn_ - stateIn.gn_;
  }

  GlobalState& operator=(const GlobalState& other) {
    if (this == &other) return *this;

    this->rn_ = other.rn_;
    this->vn_ = other.vn_;
    this->qbn_ = other.qbn_;
    this->ba_ = other.ba_;
    this->bw_ = other.bw_;
    this->gn_ = other.gn_;

    return *this;
  }

  // !@State ESKF的标称状态变量X
  V3D rn_;   // position in n-frame 当前位置
  V3D vn_;   // velocity in n-frame 当前速度
  Q4D qbn_;  // rotation from b-frame to n-frame 从起点到当前的旋转
  V3D ba_;   // acceleartion bias 加速度偏差
  V3D bw_;   // gyroscope bias 角速度偏差
  V3D gn_;   // gravity 重力加速度
};

class StatePredictor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  StatePredictor() { reset(); }

  ~StatePredictor() {}

  /** ESKF的预测过程 */
  bool predict(double dt, const V3D& acc, const V3D& gyr,
               bool update_jacobian_ = true) {
    if (!isInitialized()) return false;

    if (!flag_init_imu_) {
      flag_init_imu_ = true;
      acc_last = acc;
      gyr_last = gyr;
    }

    // Average acceleration and angular rate
    GlobalState state_tmp = state_;
	/* 将前一状态的acc转到世界坐标系，考虑偏差和重力加速度在内 */
    V3D un_acc_0 = state_tmp.qbn_ * (acc_last - state_tmp.ba_) + state_tmp.gn_;
	/* 对前一状态的gyr和当前状态的gyr求均值，考虑了偏差在内 */
    V3D un_gyr = 0.5 * (gyr_last + gyr) - state_tmp.bw_;
	/* 将gyr的均值乘以dt转成四元数形式的旋转增量 */
    Q4D dq = axis2Quat(un_gyr * dt);
	/* 将旋转增量叠加到已有的旋转状态变量，获得当前状态对应的旋转变量 */
    state_tmp.qbn_ = (state_tmp.qbn_ * dq).normalized();
	/* 将当前状态的acc转到世界坐标系，考虑偏差和重力加速度在内 */
    V3D un_acc_1 = state_tmp.qbn_ * (acc - state_tmp.ba_) + state_tmp.gn_;
	/* 对转成世界坐标系的前一状态acc和当前状态acc求均值 */
    V3D un_acc = 0.5 * (un_acc_0 + un_acc_1);

    // State integral
	/* 更新标称状态的位置和速度S=s0+vt+att/2 */
    state_tmp.rn_ = state_tmp.rn_ + dt * state_tmp.vn_ + 0.5 * dt * dt * un_acc;
    state_tmp.vn_ = state_tmp.vn_ + dt * un_acc;

	/* 构造ESKF的雅可比矩阵Fx，即状态转移矩阵，或基本矩阵，尺度是18×18 */
    if (update_jacobian_) {
	  /* 首先构造系统动力学矩阵A
	   * FIXME: 与ESKF文献唯一有区别的地方是A[6,6]项，多了一个负号，为什么？ */
      MXD Ft =
          MXD::Zero(GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_);
	  /* A[0,3] = I */
      Ft.block<3, 3>(GlobalState::pos_, GlobalState::vel_) = M3D::Identity();
	  /* A[3,6] = -q.toMatrix * acc  */
      Ft.block<3, 3>(GlobalState::vel_, GlobalState::att_) =
          -state_tmp.qbn_.toRotationMatrix() * skew(acc - state_tmp.ba_);
	  /* A[3,9] = -q.toMatrix        */
      Ft.block<3, 3>(GlobalState::vel_, GlobalState::acc_) =
          -state_tmp.qbn_.toRotationMatrix();
	  /* A[3,15] = I */
      Ft.block<3, 3>(GlobalState::vel_, GlobalState::gra_) = M3D::Identity();
	  /* A[6,6] = gyr - bias */
      Ft.block<3, 3>(GlobalState::att_, GlobalState::att_) =
          - skew(gyr - state_tmp.bw_);
	  /* A[6,12] = -I */
      Ft.block<3, 3>(GlobalState::att_, GlobalState::gyr_) = -M3D::Identity();

	  /* 构造ESKF的扰动矩阵Fi，尺度是18×12 */
      MXD Gt =
          MXD::Zero(GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_NOISE_);
      Gt.block<3, 3>(GlobalState::vel_, 0) = -state_tmp.qbn_.toRotationMatrix();
      Gt.block<3, 3>(GlobalState::att_, 3) = -M3D::Identity();
      Gt.block<3, 3>(GlobalState::acc_, 6) = M3D::Identity();
      Gt.block<3, 3>(GlobalState::gyr_, 9) = M3D::Identity();
      Gt = Gt * dt;

	  /* 构造状态转移矩阵，采用矩阵幂级数展开的方式，取级数的前三部分，获得更高的近似精度
	   * F = I + A*dt + A*A*dt*dt/2 */
      const MXD I =
          MXD::Identity(GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_);
      F_ = I + Ft * dt + 0.5 * Ft * Ft * dt * dt;

	  // 下面的步骤被省略
	  // δx = F * δx
      // jacobian_ = F * jacobian_;
	  /* P = F * P * F.transpose() + Q */
      covariance_ =
          F_ * covariance_ * F_.transpose() + Gt * noise_ * Gt.transpose();
      covariance_ = 0.5 * (covariance_ + covariance_.transpose()).eval();
    }

    state_ = state_tmp;
    time_ += dt;
    acc_last = acc;
    gyr_last = gyr;
    return true;
  }

  static void calculateRPfromIMU(const V3D& acc, double& roll, double& pitch) {
    pitch = -sign(acc.z()) * asin(acc.x() / G0);
    roll = sign(acc.z()) * asin(acc.y() / G0);
  }

  void set(const GlobalState& state) { state_ = state; }

  void update(const GlobalState& state,
              const Eigen::Matrix<double, GlobalState::DIM_OF_STATE_,
                                  GlobalState::DIM_OF_STATE_>& covariance) {
    state_ = state;
    covariance_ = covariance;
  }

  void initialization(double time, const V3D& rn, const V3D& vn, const Q4D& qbn,
                      const V3D& ba, const V3D& bw) {
    state_ = GlobalState(rn, vn, qbn, ba, bw);
    time_ = time;
    flag_init_state_ = true;

    initializeCovariance();
  }

  void initialization(double time, const V3D& rn, const V3D& vn, const Q4D& qbn,
                      const V3D& ba, const V3D& bw, const V3D& acc,
                      const V3D& gyr) {
    state_ = GlobalState(rn, vn, qbn, ba, bw);
    time_ = time;
    acc_last = acc;
    gyr_last = gyr;
    flag_init_imu_ = true;
    flag_init_state_ = true;

    initializeCovariance();
  }

  void initialization(double time, const V3D& rn, const V3D& vn, const V3D& ba,
                      const V3D& bw, double roll = 0.0, double pitch = 0.0,
                      double yaw = 0.0) {
    state_ = GlobalState(rn, vn, rpy2Quat(V3D(roll, pitch, yaw)), ba, bw);
    time_ = time;
    flag_init_state_ = true;

    initializeCovariance();
  }

  void initialization(double time, const V3D& rn, const V3D& vn, const V3D& ba,
                      const V3D& bw, const V3D& acc, const V3D& gyr,
                      double roll = 0.0, double pitch = 0.0, double yaw = 0.0) {
    state_ = GlobalState(rn, vn, rpy2Quat(V3D(roll, pitch, yaw)), ba, bw);
    time_ = time;
    acc_last = acc;
    gyr_last = gyr;
    flag_init_imu_ = true;
    flag_init_state_ = true;

    initializeCovariance();
  }

  void initializeCovariance(int type = 0) {
    double covX = pow(INIT_POS_STD(0), 2);
    double covY = pow(INIT_POS_STD(1), 2);
    double covZ = pow(INIT_POS_STD(2), 2);
    double covVx = pow(INIT_VEL_STD(0), 2);
    double covVy = pow(INIT_VEL_STD(1), 2);
    double covVz = pow(INIT_VEL_STD(2), 2);
    double covRoll = pow(deg2rad(INIT_ATT_STD(0)), 2);
    double covPitch = pow(deg2rad(INIT_ATT_STD(1)), 2);
    double covYaw = pow(deg2rad(INIT_ATT_STD(2)), 2);

    V3D covPos = INIT_POS_STD.array().square();
    V3D covVel = INIT_VEL_STD.array().square();
    V3D covAcc = INIT_ACC_STD.array().square();
    V3D covGyr = INIT_GYR_STD.array().square();

    double peba = pow(ACC_N * ug, 2);
    double pebg = pow(GYR_N * dph, 2);
    double pweba = pow(ACC_W * ugpsHz, 2);
    double pwebg = pow(GYR_W * dpsh, 2);
    V3D gra_cov(0.01, 0.01, 0.01);

    if (type == 0) {
      // Initialize using offline parameters
      covariance_.setZero();
      covariance_.block<3, 3>(GlobalState::pos_, GlobalState::pos_) =
          covPos.asDiagonal();  // pos
      covariance_.block<3, 3>(GlobalState::vel_, GlobalState::vel_) =
          covVel.asDiagonal();  // vel
      covariance_.block<3, 3>(GlobalState::att_, GlobalState::att_) =
          V3D(covRoll, covPitch, covYaw).asDiagonal();  // att
      covariance_.block<3, 3>(GlobalState::acc_, GlobalState::acc_) =
          covAcc.asDiagonal();  // ba
      covariance_.block<3, 3>(GlobalState::gyr_, GlobalState::gyr_) =
          covGyr.asDiagonal();  // bg
      covariance_.block<3, 3>(GlobalState::gra_, GlobalState::gra_) =
          gra_cov.asDiagonal();  // gravity
    } else if (type == 1) {
      // Inheritage previous covariance
      M3D vel_cov =
          covariance_.block<3, 3>(GlobalState::vel_, GlobalState::vel_);
      M3D acc_cov =
          covariance_.block<3, 3>(GlobalState::acc_, GlobalState::acc_);
      M3D gyr_cov =
          covariance_.block<3, 3>(GlobalState::gyr_, GlobalState::gyr_);
      M3D gra_cov =
          covariance_.block<3, 3>(GlobalState::gra_, GlobalState::gra_);

      covariance_.setZero();
      covariance_.block<3, 3>(GlobalState::pos_, GlobalState::pos_) =
          covPos.asDiagonal();  // pos
      covariance_.block<3, 3>(GlobalState::vel_, GlobalState::vel_) =
          vel_cov;  // vel
      covariance_.block<3, 3>(GlobalState::att_, GlobalState::att_) =
          V3D(covRoll, covPitch, covYaw).asDiagonal();  // att
      covariance_.block<3, 3>(GlobalState::acc_, GlobalState::acc_) = acc_cov;
      covariance_.block<3, 3>(GlobalState::gyr_, GlobalState::gyr_) = gyr_cov;
      covariance_.block<3, 3>(GlobalState::gra_, GlobalState::gra_) = gra_cov;
    }

    noise_.setZero();
    noise_.block<3, 3>(0, 0) = V3D(peba, peba, peba).asDiagonal();
    noise_.block<3, 3>(3, 3) = V3D(pebg, pebg, pebg).asDiagonal();
    noise_.block<3, 3>(6, 6) = V3D(pweba, pweba, pweba).asDiagonal();
    noise_.block<3, 3>(9, 9) = V3D(pwebg, pwebg, pwebg).asDiagonal();
  }

  void reset(int type = 0) {
    if (type == 0) {
      state_.rn_.setZero();
      state_.vn_ = state_.qbn_.inverse() * state_.vn_;
      state_.qbn_.setIdentity();
      initializeCovariance();
    } else if (type == 1) {
      V3D covPos = INIT_POS_STD.array().square();
      double covRoll = pow(deg2rad(INIT_ATT_STD(0)), 2);
      double covPitch = pow(deg2rad(INIT_ATT_STD(1)), 2);
      double covYaw = pow(deg2rad(INIT_ATT_STD(2)), 2);

      M3D vel_cov =
          covariance_.block<3, 3>(GlobalState::vel_, GlobalState::vel_);
      M3D acc_cov =
          covariance_.block<3, 3>(GlobalState::acc_, GlobalState::acc_);
      M3D gyr_cov =
          covariance_.block<3, 3>(GlobalState::gyr_, GlobalState::gyr_);
      M3D gra_cov =
          covariance_.block<3, 3>(GlobalState::gra_, GlobalState::gra_);

      covariance_.setZero();
      covariance_.block<3, 3>(GlobalState::pos_, GlobalState::pos_) =
          covPos.asDiagonal();  // pos
      covariance_.block<3, 3>(GlobalState::vel_, GlobalState::vel_) =
          state_.qbn_.inverse() * vel_cov * state_.qbn_;  // vel
      covariance_.block<3, 3>(GlobalState::att_, GlobalState::att_) =
          V3D(covRoll, covPitch, covYaw).asDiagonal();  // att
      covariance_.block<3, 3>(GlobalState::acc_, GlobalState::acc_) = acc_cov;
      covariance_.block<3, 3>(GlobalState::gyr_, GlobalState::gyr_) = gyr_cov;
      covariance_.block<3, 3>(GlobalState::gra_, GlobalState::gra_) =
          state_.qbn_.inverse() * gra_cov * state_.qbn_;

      state_.rn_.setZero();
      state_.vn_ = state_.qbn_.inverse() * state_.vn_;
      state_.qbn_.setIdentity();
      state_.gn_ = state_.qbn_.inverse() * state_.gn_;
      state_.gn_ = state_.gn_ * 9.81 / state_.gn_.norm();
      // initializeCovariance(1);
    }
  }

  void reset(V3D vn, V3D ba, V3D bw) {
    state_.setIdentity();
    state_.vn_ = vn;
    state_.ba_ = ba;
    state_.bw_ = bw;
    initializeCovariance();
  }

  inline bool isInitialized() { return flag_init_state_; }

  GlobalState state_;
  double time_;
  Eigen::Matrix<double, GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_>
      F_;
  Eigen::Matrix<double, GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_>
      jacobian_, covariance_;
  Eigen::Matrix<double, GlobalState::DIM_OF_NOISE_, GlobalState::DIM_OF_NOISE_>
      noise_;

  V3D acc_last;  // last acceleration measurement
  V3D gyr_last;  // last gyroscope measurement

  bool flag_init_state_;
  bool flag_init_imu_;
};

};  // namespace filter

#endif  // INCLUDE_KALMANFILTER_HPP_
