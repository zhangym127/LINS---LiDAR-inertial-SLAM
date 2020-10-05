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

#ifndef INCLUDE_STATEESTIMATOR_HPP_
#define INCLUDE_STATEESTIMATOR_HPP_

#include <integrationBase.h>
#include <math_utils.h>
#include <parameters.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/Imu.h>
#include <tf/transform_broadcaster.h>
#include <tic_toc.h>

#include <KalmanFilter.hpp>
#include <algorithm>
#include <boost/shared_ptr.hpp>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <sensor_utils.hpp>
#include <vector>

#include "cloud_msgs/cloud_info.h"
#include "integrationBase.h"

using namespace Eigen;
using namespace std;
using namespace math_utils;
using namespace sensor_utils;
using namespace parameter;
using namespace filter;

namespace fusion {

const int LINE_NUM_ = 16;
const int SCAN_NUM_ = 1800;

struct Smooth {
  Smooth() {
    value = 0.0;
    ind = 0;
  }
  double value;
  size_t ind;
};

struct byValue {
  bool operator()(Smooth const& left, Smooth const& right) {
    return left.value < right.value;
  }
};

// Scan Class stores all kinds of information of a point cloud, including
// the whole point cloud, its smoothness, timestamp, and features.
class Scan {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Scan() : id_(scan_counter_++) {
    distPointCloud_.reset(new pcl::PointCloud<PointType>());
    undistPointCloud_.reset(new pcl::PointCloud<PointType>());
    outlierPointCloud_.reset(new pcl::PointCloud<PointType>());
    cornerPointsSharp_.reset(new pcl::PointCloud<PointType>());
    cornerPointsLessSharp_.reset(new pcl::PointCloud<PointType>());
    surfPointsFlat_.reset(new pcl::PointCloud<PointType>());
    surfPointsLessFlat_.reset(new pcl::PointCloud<PointType>());
    cloudInfo_.reset(new cloud_msgs::cloud_info());

    cornerPointsLessSharpYZX_.reset(new pcl::PointCloud<PointType>());
    surfPointsLessFlatYZX_.reset(new pcl::PointCloud<PointType>());
    outlierPointCloudYZX_.reset(new pcl::PointCloud<PointType>());

    cloudCurvature_.resize(LINE_NUM * SCAN_NUM);
    cloudSmoothness_.resize(LINE_NUM * SCAN_NUM);

    reset();
  }

  ~Scan() {
    distPointCloud_.reset();
    undistPointCloud_.reset();
    outlierPointCloud_.reset();
    cornerPointsSharp_.reset();
    cornerPointsLessSharp_.reset();
    surfPointsFlat_.reset();
    surfPointsLessFlat_.reset();
    cloudInfo_.reset();

    cornerPointsLessSharpYZX_.reset();
    surfPointsLessFlatYZX_.reset();
    outlierPointCloudYZX_.reset();
  }

  void reset() {
    time_ = 0.0;

    distPointCloud_->clear();
    undistPointCloud_->clear();
    outlierPointCloud_->clear();
    cornerPointsSharp_->clear();
    cornerPointsLessSharp_->clear();
    surfPointsFlat_->clear();
    surfPointsLessFlat_->clear();

    cornerPointsLessSharpYZX_->clear();
    surfPointsLessFlatYZX_->clear();
    outlierPointCloudYZX_->clear();

    cloudCurvature_.assign(LINE_NUM * SCAN_NUM, 0.0);
    cloudSmoothness_.assign(LINE_NUM * SCAN_NUM, Smooth());
  }

  void setPointCloud(double time,
                     pcl::PointCloud<PointType>::Ptr distPointCloud,
                     cloud_msgs::cloud_info cloudInfo,
                     pcl::PointCloud<PointType>::Ptr outlierPointCloud) {
    distPointCloud_ = distPointCloud;
    *(cloudInfo_) = cloudInfo;
    outlierPointCloud_ = outlierPointCloud;
    time_ = time;
  }

 public:
  // !@ScanInfo
  static int scan_counter_;
  int id_;
  double time_;

  // !@PointCloud
  pcl::PointCloud<PointType>::Ptr distPointCloud_;
  pcl::PointCloud<PointType>::Ptr undistPointCloud_;
  pcl::PointCloud<PointType>::Ptr outlierPointCloud_;
  cloud_msgs::cloud_info::Ptr cloudInfo_;

  // !@PclFeatures
  std::vector<double> cloudCurvature_;
  std::vector<Smooth> cloudSmoothness_;

  int cloudNeighborPicked_[LINE_NUM_ * SCAN_NUM_];
  int cloudLabel_[LINE_NUM_ * SCAN_NUM_];

  pcl::PointCloud<PointType>::Ptr cornerPointsSharp_;
  pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp_;
  pcl::PointCloud<PointType>::Ptr surfPointsFlat_;
  pcl::PointCloud<PointType>::Ptr surfPointsLessFlat_;

  pcl::PointCloud<PointType>::Ptr cornerPointsLessSharpYZX_;
  pcl::PointCloud<PointType>::Ptr surfPointsLessFlatYZX_;
  pcl::PointCloud<PointType>::Ptr outlierPointCloudYZX_;
};
typedef shared_ptr<Scan> ScanPtr;  // Define a pointer class for Scan class

// StateEstimator Class implement a iterative-ESKF, including state propagation
// and update.
class StateEstimator {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  enum FusionStatus {
    STATUS_INIT = 0,
    STATUS_FIRST_SCAN = 1,
    STATUS_SECOND_SCAN = 2,
    STATUS_RUNNING = 3,
    STATUS_RESET = 4,
  };

  StateEstimator() {
    filter_ = new StatePredictor();

    // Initialize KD tree and downsize filter
    downSizeFilter_.setLeafSize(0.2, 0.2, 0.2);
    kdtreeCorner_.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeSurf_.reset(new pcl::KdTreeFLANN<PointType>());
    scan_new_.reset(new Scan());
    scan_last_.reset(new Scan());

    keypoints_.reset(new pcl::PointCloud<PointType>());
    jacobians_.reset(new pcl::PointCloud<PointType>());
    keypointCorns_.reset(new pcl::PointCloud<PointType>());
    keypointSurfs_.reset(new pcl::PointCloud<PointType>());
    jacobianCoffCorns.reset(new pcl::PointCloud<PointType>());
    jacobianCoffSurfs.reset(new pcl::PointCloud<PointType>());

    surfPointsLessFlatScan.reset(new pcl::PointCloud<PointType>());
    surfPointsLessFlatScanDS.reset(new pcl::PointCloud<PointType>());

    pointSelCornerInd.resize(LINE_NUM * SCAN_NUM);
    pointSearchCornerInd1.resize(LINE_NUM * SCAN_NUM);
    pointSearchCornerInd2.resize(LINE_NUM * SCAN_NUM);
    pointSelSurfInd.resize(LINE_NUM * SCAN_NUM);
    pointSearchSurfInd1.resize(LINE_NUM * SCAN_NUM);
    pointSearchSurfInd2.resize(LINE_NUM * SCAN_NUM);
    pointSearchSurfInd3.resize(LINE_NUM * SCAN_NUM);

    globalState_.setIdentity();
    globalStateYZX_.setIdentity();

    // Rotation matrics between XYZ convention and YZX-convention
    R_yzx_to_xyz << 0., 0., 1., 1., 0., 0., 0., 1., 0.;
    R_xyz_to_yzx = R_yzx_to_xyz.transpose();
    Q_yzx_to_xyz = R_yzx_to_xyz;
    Q_xyz_to_yzx = R_xyz_to_yzx;

    // gravity_feedback << 0, 0, -G0;

    status_ = STATUS_INIT;
  }

  ~StateEstimator() {
    delete filter_;
    delete preintegration_;
  }

  inline const double getTime() const { return filter_->time_; }
  inline bool isInitialized() const { return status_ != STATUS_INIT; }

  /********Relative Variables*********/
  V3D pos_;
  V3D vel_;
  M3D quad_;
  V3D acc_0_;
  V3D gyr_0_;
  /***********************************/
  void processImu(double dt, const V3D& acc, const V3D& gyr) {
    switch (status_) {
      case STATUS_INIT:
        break;
      case STATUS_FIRST_SCAN:
        preintegration_->push_back(dt, acc, gyr);
        filter_->time_ += dt;
        acc_0_ = acc;
        gyr_0_ = gyr;
        break;
      case STATUS_RUNNING:
        filter_->predict(dt, acc, gyr, true);
        break;
      default:
        break;
    }

    // For no use here. Just propagte IMU measurements for testing
    V3D un_acc_0_ = quad_ * (acc_0_ - INIT_BA) + filter_->state_.gn_;
    V3D un_gyr = 0.5 * (gyr_0_ + gyr) - INIT_BW;
    quad_ *= math_utils::deltaQ(un_gyr * dt).toRotationMatrix();
    V3D un_acc_1 = quad_ * (acc - INIT_BA) + filter_->state_.gn_;
    V3D un_acc = 0.5 * (un_acc_0_ + un_acc_1);
    pos_ += dt * vel_ + 0.5 * dt * dt * un_acc;
    vel_ += dt * un_acc;

    acc_0_ = acc;
    gyr_0_ = gyr;
  }

  /********Relative Variables*********/
  double duration_fea_ = 0;
  double duration_opt_ = 0;
  double num_of_edge_ = 0;
  double num_of_surf_ = 0;
  int lidar_counter_ = 0;
  /***********************************/
  void processPCL(double time, const Imu& imu,
                  pcl::PointCloud<PointType>::Ptr distortedPointCloud,
                  cloud_msgs::cloud_info cloudInfo,
                  pcl::PointCloud<PointType>::Ptr outlierPointCloud) {
    TicToc ts_fea;  // Calculate the time used in feature extraction
    scan_new_->setPointCloud(time, distortedPointCloud, cloudInfo,
                             outlierPointCloud);
	/* 将点云从雷达坐标系转到车辆坐标系，并获得每个点的相对时间戳，保存在
     * intensity字段的小数部分中。 */
    undistortPcl(scan_new_);
	/* 计算点云的曲率 */
    calculateSmoothness(scan_new_);
	/* 对点云扫描线上的凹陷区域和离点进行标识，排除在提取特征点的范围之外 */
    markOccludedPoints(scan_new_);
	/* 提取Sharp、LessSharp、Flat、LessFlat四种特征点云 */
    extractFeatures(scan_new_);
    imu_last_ = imu;
    double time_fea = ts_fea.toc();

    TicToc ts_opt;  // Calculate the time used in state estimation
    switch (status_) {
      case STATUS_INIT:
        if (processFirstScan()) status_ = STATUS_FIRST_SCAN;
        break;
      case STATUS_FIRST_SCAN:
        if (processSecondScan())
          status_ = STATUS_RUNNING;
        else
          status_ = STATUS_INIT;
        break;
      case STATUS_RUNNING:
        if (!processScan()) status_ = STATUS_RUNNING;
        break;
    }
    double time_opt = ts_opt.toc();

    // if (VERBOSE) {
    //   duration_fea_ =
    //       (duration_fea_ * lidar_counter_ + time_fea) / (lidar_counter_ + 1);
    //   duration_opt_ =
    //       (duration_opt_ * lidar_counter_ + time_opt) / (lidar_counter_ + 1);
    //   num_of_edge_ = (num_of_edge_ * lidar_counter_ +
    //                   scan_last_->cornerPointsLessSharpYZX_->points.size()) /
    //                  (lidar_counter_ + 1);
    //   num_of_surf_ = (num_of_surf_ * lidar_counter_ +
    //                   scan_last_->surfPointsLessFlatYZX_->points.size()) /
    //                  (lidar_counter_ + 1);
    //   lidar_counter_++;

    //   // cout << "Feature Extraction: time: " << duration_fea_ << endl;
    //   // cout << "Feature Extraction: corners: " << num_of_edge_
    //   //      << ", surfs: " << num_of_surf_ << endl;
    //   // cout << "State Estimation: time: " << duration_opt_ << endl;
    // }
  }

  // Initialize the Kalman filter and KD tree
  bool processFirstScan() {
    if (scan_new_->cornerPointsLessSharp_->points.size() < 10 ||
        scan_new_->surfPointsLessFlat_->points.size() < 100) {
      ROS_WARN("Wait for more features for initialization...");
      scan_new_.reset(new Scan());
      return false;
    }

    // Initialize the Kalman filter
    Fk_.resize(GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_);
    Gk_.resize(GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_NOISE_);
    Pk_.resize(GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_);
    Qk_.resize(GlobalState::DIM_OF_NOISE_, GlobalState::DIM_OF_NOISE_);
    IKH_.resize(GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_);

    Fk_.setIdentity();
    Gk_.setZero();
    Pk_.setZero();
    Qk_.setZero();

    // Set the relative transform to identity
    linState_.setIdentity();

    // Initialize IMU preintegration variable
    preintegration_ = new integration::IntegrationBase(
        imu_last_.acc, imu_last_.gyr, INIT_BA, INIT_BW);

    // Initialize position, velocity, acceleration bias, gyroscope bias by zeros
    filter_->initialization(scan_new_->time_, V3D(0, 0, 0), V3D(0, 0, 0),
                            V3D(0, 0, 0), V3D(0, 0, 0), imu_last_.acc,
                            imu_last_.gyr);

    kdtreeCorner_->setInputCloud(scan_new_->cornerPointsLessSharp_);
    kdtreeSurf_->setInputCloud(scan_new_->surfPointsLessFlat_);

    pos_.setZero();
    vel_.setZero();
    quad_.setIdentity();

    // Slide the point cloud from scan_new_ to scan_last_
    scan_last_.swap(scan_new_);
    scan_new_.reset(new Scan());

    return true;
  }

  // Calculate initial velocity and IMU biases using two consecutive frames and
  // IMU preintegration results
  bool processSecondScan() {
    if (scan_new_->cornerPointsLessSharp_->points.size() < 10 ||
        scan_new_->surfPointsLessFlat_->points.size() < 100) {
      ROS_WARN("Wait for more features for initialization...");
      scan_new_.reset(new Scan());
      return false;
    }

    // Calculate relative transform, linState_, using ICP method
    V3D pl;
    Q4D ql;
    V3D v0, v1, ba0, bw0;
    ba0.setZero();
    ql = preintegration_->delta_q;
    pl = preintegration_->delta_p +
         0.5 * linState_.gn_ * preintegration_->sum_dt *
             preintegration_->sum_dt -
         0.5 * ba0 * preintegration_->sum_dt * preintegration_->sum_dt;
    estimateTransform(scan_last_, scan_new_, pl, ql);

    // Calculate initial state using relative transform calculated by point
    // clouds and that by IMU preintegration
    estimateInitialState(pl, ql, v0, v1, ba0, bw0);

    // Initialize the Kalman filter by estimated values
    V3D r1 = pl;
    filter_->initialization(scan_new_->time_, r1, v1, ba0, bw0, imu_last_.acc,
                            imu_last_.gyr);

    double roll_init, pitch_init, yaw_init = deg2rad(0.0);
    // Calculate rough roll and pitch angles using IMU measurements
    calculateRPfromGravity(imu_last_.acc - ba0, roll_init, pitch_init);

    // Initialize the global state, e.g., position, velocity, and orientation
    // represented in the original frame (the first-scan-frame)
    globalState_ = GlobalState(
        r1, v1, rpy2Quat(V3D(roll_init, pitch_init, yaw_init)), ba0, bw0);

    // Use relative transorm linState_ to undistort point cloud (under the
    // constant-speed assumption)
    updatePointCloud();

    scan_last_.swap(scan_new_);
    scan_new_.reset(new Scan());

    return true;
  }

  void correctRollPitch(const double& roll, const double& pitch) {
    V3D rpy = math_utils::Q2rpy(globalState_.qbn_);
    Q4D quad = math_utils::rpy2Quat(V3D(roll, pitch, rpy[2]));
    globalState_.qbn_ = quad;
  }

  void correctOrientation(const Q4D& quad) { globalState_.qbn_ = quad; }

  bool processScan() {
    if (scan_new_->cornerPointsLessSharp_->points.size() <= 5 ||
        scan_new_->surfPointsLessFlat_->points.size() <= 10) {
      ROS_WARN("Insufficient features...State estimation fails.");
      return false;
    }

	/** 进行IESKF的Update过程，对当前(k+1)帧与上一(k)帧之间的位姿变化量进行校正 */
    // Update states
    performIESKF();
	/** 将校正后的帧间位姿变化量叠加到位姿的全局状态变量中 */
    // Update global transform by estimated relative transform
    integrateTransformation();
	/** 将ESKF复位 */
    filter_->reset(1);

	/** 
	 * 根据重力加速度的方向直接计算出roll和pitch的值，并更新globalState_中的roll
	 * 和pitch */
    double roll, pitch;
    // Because the estimated gravity is represented in the b-frame, we can
    // directly solve more accurate roll and pitch angles to correct the global
    // state
    calculateRPfromGravity(filter_->state_.gn_, roll, pitch);
    correctRollPitch(roll, pitch);

    /** 
     * 对当前帧特征点云进行变换，使所有点向帧的结束时刻看齐，然后添加到kdTree中，
     * 为与下一帧点云的匹配做好准备。
     */
    // Undistort point cloud using estimated relative transform
    updatePointCloud();

    /** 当前帧点云切换为上一帧点云 */
    // Slide the new scan to last scan
    scan_last_.swap(scan_new_);
    scan_new_.reset(new Scan());

    return true;
  }

  /** 
   * ESKF的更新过程（低频）
   * 更新当前(k+1)帧点云相对于上一(k)帧终点的位姿变换，或者说：
   * 更新当前帧点云终点相对于当前帧点云起点的位姿变换。
   */
  void performIESKF() {
	  
	/** 获得当前的协方差P和状态变量，状态变量是自上一帧点云终点起到当前
	 * 帧点云终点累计的位姿预测（积分）结果 */
    // Store current state and perform initialization
    Pk_ = filter_->covariance_;
    GlobalState filterState = filter_->state_;
    linState_ = filterState;

    double residualNorm = 1e6;
    bool hasConverged = false;
    bool hasDiverged = false;
    const unsigned int DIM_OF_STATE = GlobalState::DIM_OF_STATE_;
	
	/**
	 * 下面进行迭代ESKF，即IESKF过程
	 * 在每一次迭代中，首先用状态变量的标称状态linState_对点云进行帧内校正，使当
	 * 前帧内所有点向帧的起始时刻看齐，然后通过观测方程计算两帧点云的位姿偏差及其
	 * 雅可比，基于偏差和雅可比获得误差状态的后验，并将后验注入标称状态linState_
	 * 如果误差状态的后验没有趋近于零，则重复上述过程，直至趋近于零。
	 */
    for (int iter = 0; iter < NUM_ITER && !hasConverged && !hasDiverged;
         iter++) {
      keypointSurfs_->clear();
      jacobianCoffSurfs->clear();
      keypointCorns_->clear();
      jacobianCoffCorns->clear();

      /**
       * 遍历特征点云，基于linState_提供的帧间位姿变换量，对点进行帧内校正，使帧
	   * 内所有点向帧的起始时刻看齐。然后在上一帧特征点云的kdtree中找到当前点p的
	   * 最近点a和次近点b，再获得点p到线段ab的距离，也就是观测方程的结果，并求得
	   * 观测方程的雅可比，保存在jacobianCoff中，对应的当前点p保存在keypoints中。
	   * 
	   * 由于上一帧点云已经向帧的结束时刻（即当前帧的起始时刻）看齐，因此理论上两
	   * 帧点云应该完全重合，观测方程的结果为0才对。但是位姿的估计值肯定是存在误差
	   * 的，这个误差只有通过引入观测值才能显现出来。
	   */
      // Find corresponding features
      findCorrespondingSurfFeatures(scan_last_, scan_new_, keypointSurfs_,
                                    jacobianCoffSurfs, iter);
      if (keypointSurfs_->points.size() < 10) {
        if (VERBOSE) {
          ROS_WARN("Insufficient matched surfs...");
        }
      }
	  
      findCorrespondingCornerFeatures(scan_last_, scan_new_, keypointCorns_,
                                      jacobianCoffCorns, iter);
      if (keypointCorns_->points.size() < 5) {
        if (VERBOSE) {
          ROS_WARN("Insufficient matched corners...");
        }
      }

	  /* 把所有的keypoints和jacobians汇总在一起 */
      // Sum up jocobians and residuals
      keypoints_->clear();
      jacobians_->clear();
      (*keypoints_) += (*keypointSurfs_);
      (*keypoints_) += (*keypointCorns_);
      (*jacobians_) += (*jacobianCoffSurfs);
      (*jacobians_) += (*jacobianCoffCorns);

	  /* 根据keypoints的数量，为ESKF的update变量H、R、K、S分配空间
       * 在这里，每个keypoints中的p点与上一帧点云对应的最近点的距离就是一个观测量，
	   * 与一般的卡尔曼滤波只有几个观测量不同，这个ESKF的观测量的规模有点大。*/
      // Memery allocation
      const unsigned int DIM_OF_MEAS = keypoints_->points.size();
      residual_.resize(DIM_OF_MEAS);
	  /* 测量函数 H i×18 */
      Hk_.resize(DIM_OF_MEAS, DIM_OF_STATE);
	  /* 测量噪声 R */
      Rk_.resize(DIM_OF_MEAS, DIM_OF_MEAS);
	  /* 卡尔曼增益 K */
      Kk_.resize(DIM_OF_STATE, DIM_OF_MEAS);
	  /* 系统不确定性 S */
      Py_.resize(DIM_OF_MEAS, DIM_OF_MEAS);
      Pyinv_.resize(DIM_OF_MEAS, DIM_OF_MEAS);

	  /* 构造观测变量z（即residual_），构造观测函数H */
      Hk_.setZero();
      V3D axis = Quat2axis(linState_.qbn_);
      for (int i = 0; i < DIM_OF_MEAS; ++i) {
		  
		/* keypoints_中保存的是当前特征点云中的关键点 */
        // Point represented in 2-frame (e.g., the end frame) in a
        // xyz-convention
        V3D P2xyz(keypoints_->points[i].x, keypoints_->points[i].y,
                  keypoints_->points[i].z);
				  
		/* jacobians_的(x,y,z)中保存的是观测方程的雅可比 */
        V3D coff_xyz(jacobians_->points[i].x, jacobians_->points[i].y,
                     jacobians_->points[i].z);

		/* jacobians_的intensity中保存的是当前特征点云中每个点p到上一帧点云
		 * 最近线段ab的距离，也就是观测值z */
        residual_(i) = LIDAR_SCALE * jacobians_->points[i].intensity;

		/* 构造观测方程的雅可比 */
		/* H[i,6] = coeff^T * (-R * X) * J(-θ)^(-1), */
        Hk_.block<1, 3>(i, GlobalState::att_) =
            coff_xyz.transpose() *
            (-linState_.qbn_.toRotationMatrix() * skew(P2xyz)) *
            Rinvleft(-axis);
		/* H[i,0] = coeff^T * I */
        Hk_.block<1, 3>(i, GlobalState::pos_) =
            coff_xyz.transpose() * M3D::Identity();
      }

	  /* 设置测量协方差矩阵，设置为雷达的标准差 */
      // Set the measurement covariance matrix
      VXD cov = VXD::Zero(DIM_OF_MEAS);
      for (int i = 0; i < DIM_OF_MEAS; ++i) {
        cov[i] = LIDAR_STD * LIDAR_STD;
      }
      Rk_ = cov.asDiagonal();

      // Kalman filter update. Details can be referred to ROVIO
      Py_ = Hk_ * Pk_ * Hk_.transpose() + Rk_;// S = H * P * H.transpose() + R;
	  /**
	   * 通过Cholesky分解（也叫LLT分解）求S的逆矩阵，令AX=I，求解X，X就是A的逆。
	   * 下面首先令Pyinv_=I，然后通过LLT分解求解Py_*Pyinv_=I，Pyinv_就是Py_的逆。 */
      Pyinv_.setIdentity();                   // solve Ax=B
      Py_.llt().solveInPlace(Pyinv_);
      Kk_ = Pk_ * Hk_.transpose() * Pyinv_;  // K = P*H.transpose()*S.inverse()

	  /**
	   * filterState 是位姿状态的估计值或者先验，对应ESKF中的真实状态
	   * linState_ 是基于观测值进行多轮迭代优化的后验，对应ESKF中的标称状态。
	   * linState_ 的初值与filterState相同，这与ESKF的预测过程刚结束、更新过程尚未
	   *	开始时误差状态为0相对应。
	   * difVecLinInv_ 是位姿状态先验与后验的差值，也就是误差状态的先验，初始值为0。
	   * updateVec_ 是误差状态的后验，也就是引入观测值，更新后的误差状态。
	   *
	   * 在predict刚结束，update尚未开始的时候，误差状态为0，也就是还没有观测到。
	   * 随着观测结果的引入，后验开始偏离先验，误差状态开始产生。然后通过将误差
	   * 状态注入标称状态，并进行IESKF迭代优化，最终使得误差状态后验趋近于零。
	   *
	   * IESKF迭代优化的目标是误差状态的后验updateVec_趋近于0。
	   */
	  /**
       * 通过先验（真实状态）减去后验（标称状态），获得误差状态的先验
	   * FIXME：刚开始的时候后验与先验一致，误差状态为0，随着迭代的推进，误差显现，
	   * 后验与先验分离，误差状态的先验也越来越大。这个先验会对后验的计算造成怎样
	   * 的影响？
	   */
	  /** difVecLinInv_ = filterState - linState_ */
      filterState.boxMinus(linState_, difVecLinInv_);

	  /* 下面的式子等效于：
	   *	y=z-H*x
	   *	x=x+K*y
	   * 其中
	   *    z      <-- residual_
	   *    H      <-- Hk_
	   *    δx先验 <-- difVecLinInv_
	   *    K      <-- Kk_
	   *    δx后验 <-- updateVec_
	   */
	  /* 通过引入观测值，更新误差状态，获得误差状态的后验 */
      updateVec_ = -Kk_ * (residual_ + Hk_ * difVecLinInv_) + difVecLinInv_;
   
	  /* 通过检查误差状态后验中是否存在无效值来判断是否发散 */
      // Divergence determination
      bool hasNaN = false;
      for (int i = 0; i < updateVec_.size(); i++) {
        if (isnan(updateVec_[i])) {
          updateVec_[i] = 0;
          hasNaN = true;
        }
      }
      if (hasNaN == true) {
        ROS_WARN("System diverges Because of NaN...");
        hasDiverged = true;
        break;
      }

      // Check whether the filter converges
      if (residual_.norm() > residualNorm * 10) {
        ROS_WARN("System diverges...");
        hasDiverged = true;
        break;
      }

	  /* 将误差状态后验注入标称状态
	   * linState_ = linState_ + updateVec_ */
      // Update the state
      linState_.boxPlus(updateVec_, linState_);

	  /** 
	   * 判断误差状态的后验是否趋近于零，如果没有，则用更新后的标称状态linState_
	   * 对当前点云进行帧内校正，用校正后的点云重新计算观测方程的结果及其雅可比，
       * 再计算出新的误差状态。重复上述过程，直到误差状态后验趋近于零。*/
      updateVecNorm_ = updateVec_.norm();
      if (updateVecNorm_ <= 1e-2) {
        hasConverged = true;
      }

      residualNorm = residual_.norm();
    }

    // If diverges, swtich to traditional ICP method to get a rough relative
    // transformation. Otherwise, update the error-state covariance matrix
    if (hasDiverged == true) {
      ROS_WARN("======Using ICP Method======");
      V3D t = filterState.rn_;
      Q4D q = filterState.qbn_;
      estimateTransform(scan_last_, scan_new_, t, q);
      filterState.rn_ = t;
      filterState.qbn_ = q;
      filter_->update(filterState, Pk_);
    } else {
	  /** 
	   * 更新协方差P，注意：在IEKF的更新过程中，状态变量更新多次，但是P只更新
	   * 一次 */
      // Update only one time
      IKH_ = Eigen::Matrix<double, 18, 18>::Identity() - Kk_ * Hk_;
      Pk_ = IKH_ * Pk_ * IKH_.transpose() + Kk_ * Rk_ * Kk_.transpose();
	  /* 将P强制成对称矩阵 */
      enforceSymmetry(Pk_);
	  /* 将迭代优化后的标称状态和协方差写入滤波器 */
      filter_->update(linState_, Pk_);
    }
  }

  void calculateRPfromGravity(const V3D& fbib, double& roll, double& pitch) {
    pitch = -sign(fbib.z()) * asin(fbib.x() / G0);
    roll = sign(fbib.z()) * asin(fbib.y() / G0);
  }

  /**
   * 将IESKF的update的当前帧结果叠加到位姿的全局状态变量中
   */
  // Update the gloabl state by the new relative transformation
  void integrateTransformation() {
    GlobalState filterState = filter_->state_;
    globalState_.rn_ = globalState_.qbn_ * filterState.rn_ + globalState_.rn_;
    globalState_.qbn_ = globalState_.qbn_ * filterState.qbn_;
    globalState_.vn_ =
        globalState_.qbn_ * filterState.qbn_.inverse() * filterState.vn_;
    globalState_.ba_ = filterState.ba_;
    globalState_.bw_ = filterState.bw_;
    globalState_.gn_ = globalState_.qbn_ * filterState.gn_;
  }

  /**
   * 将点云从雷达坐标系转到车辆坐标系，并获得每个点的相对时间戳，保存在
   * intensity字段的小数部分中。
   */
  void undistortPcl(ScanPtr scan) {
    bool halfPassed = false;
    scan->undistPointCloud_->clear();
    pcl::PointCloud<PointType>::Ptr distPointCloud = scan->distPointCloud_;
    cloud_msgs::cloud_info::Ptr segInfo = scan->cloudInfo_;
    int size = distPointCloud->points.size();
    PointType point;
    for (int i = 0; i < size; i++) {
      
	  /* 将点云从点云坐标系转到车辆坐标系 */
	  // If LiDAR frame does not align with Vehic frame, we transform the point
      // cloud to the vehicle frame
      rotatePoint(&distPointCloud->points[i], &point);

	  /* 根据点的水平方位确定该点的时间戳 */
      double ori = -atan2(point.y, point.x);
      if (!halfPassed) {
        if (ori < segInfo->startOrientation - M_PI / 2)
          ori += 2 * M_PI;
        else if (ori > segInfo->startOrientation + M_PI * 3 / 2)
          ori -= 2 * M_PI;

        if (ori - segInfo->startOrientation > M_PI) halfPassed = true;
      } else {
        ori += 2 * M_PI;

        if (ori < segInfo->endOrientation - M_PI * 3 / 2)
          ori += 2 * M_PI;
        else if (ori > segInfo->endOrientation + M_PI / 2)
          ori -= 2 * M_PI;
      }
	  /* 根据该点的方位获得在帧内的时间比例系数，再乘以帧周期就是该点在帧内的相对时间戳 */
      double relTime =
          (ori - segInfo->startOrientation) / segInfo->orientationDiff;
	  /* 将点的时间戳写入intensity字段的小数部分，这个时间戳是帧内相对时间戳 */
      point.intensity =
          int(distPointCloud->points[i].intensity) + SCAN_PERIOD * relTime;

      scan->undistPointCloud_->push_back(point);
    }
  }

  /**
   * 计算点云的曲率
   */
  void calculateSmoothness(ScanPtr scan) {
    int cloudSize = scan->undistPointCloud_->points.size();
    cloud_msgs::cloud_info::Ptr segInfo = scan->cloudInfo_;

	/* segmentedCloudRange中保存的是每个点的距离，即Range，
	 * segmentedCloudRange中的点是按照扫描线分段组织好的，
	 * 通过计算当前点与两侧10个点的距离差，获得当前点的曲率*/
    for (int i = 5; i < cloudSize - 5; i++) {
      double diffRange = segInfo->segmentedCloudRange[i - 5] +
                         segInfo->segmentedCloudRange[i - 4] +
                         segInfo->segmentedCloudRange[i - 3] +
                         segInfo->segmentedCloudRange[i - 2] +
                         segInfo->segmentedCloudRange[i - 1] -
                         segInfo->segmentedCloudRange[i] * 10 +
                         segInfo->segmentedCloudRange[i + 1] +
                         segInfo->segmentedCloudRange[i + 2] +
                         segInfo->segmentedCloudRange[i + 3] +
                         segInfo->segmentedCloudRange[i + 4] +
                         segInfo->segmentedCloudRange[i + 5];
      scan->cloudCurvature_[i] = diffRange * diffRange;

      scan->cloudNeighborPicked_[i] = 0;
      scan->cloudLabel_[i] = 0;
      scan->cloudSmoothness_[i].value = scan->cloudCurvature_[i];
      scan->cloudSmoothness_[i].ind = i;
    }
  }

  /**
   * 对点云扫描线上存在凹陷的区域进行标识，排除在提取特征点的范围之外
   * 对点云扫描线上的离点进行标识，排除在提取特征点的范围之外
   * 所谓凹陷是指从雷达的视角看过去，某个扫描线上的凹陷区域，凹陷程度
   * （即阈值）＞0.3米
   */
  void markOccludedPoints(ScanPtr scan) {
    int cloudSize = scan->undistPointCloud_->points.size();
    cloud_msgs::cloud_info::Ptr segInfo = scan->cloudInfo_;
    for (int i = 5; i < cloudSize - 6; ++i) {
      float depth1 = segInfo->segmentedCloudRange[i];
      float depth2 = segInfo->segmentedCloudRange[i + 1];
	  /* 获得相邻两个点在各自扫描线内的序号差，如果同属于一个扫描线，
	   * 则columnDiff应该等于1，否则columnDiff等于扫描线的长度减一 */
      int columnDiff = std::abs(int(segInfo->segmentedCloudColInd[i + 1] -
                                    segInfo->segmentedCloudColInd[i]));
	  /* 确保当前点不是扫描线的端点 */
      if (columnDiff < 10) {
		/* 以距离为尺度，标识出扫描线上存在凹陷的区域（±12个点），排除在特征点之外 */
        if (depth1 - depth2 > 0.3) {
          scan->cloudNeighborPicked_[i - 5] = 1;
          scan->cloudNeighborPicked_[i - 4] = 1;
          scan->cloudNeighborPicked_[i - 3] = 1;
          scan->cloudNeighborPicked_[i - 2] = 1;
          scan->cloudNeighborPicked_[i - 1] = 1;
          scan->cloudNeighborPicked_[i] = 1;
        } else if (depth2 - depth1 > 0.3) {
          scan->cloudNeighborPicked_[i + 1] = 1;
          scan->cloudNeighborPicked_[i + 2] = 1;
          scan->cloudNeighborPicked_[i + 3] = 1;
          scan->cloudNeighborPicked_[i + 4] = 1;
          scan->cloudNeighborPicked_[i + 5] = 1;
          scan->cloudNeighborPicked_[i + 6] = 1;
        }
      }
	  /* 如果当前点与两侧点的距离偏差同时超过2%，即离点，排除在特征点之外 */
      float diff1 = std::abs(segInfo->segmentedCloudRange[i - 1] -
                             segInfo->segmentedCloudRange[i]);
      float diff2 = std::abs(segInfo->segmentedCloudRange[i + 1] -
                             segInfo->segmentedCloudRange[i]);
      if (diff1 > 0.02 * segInfo->segmentedCloudRange[i] &&
          diff2 > 0.02 * segInfo->segmentedCloudRange[i])
        scan->cloudNeighborPicked_[i] = 1;
    }
  }

 /**
  * 对点云进行特征提取，提取Sharp、LessSharp、Flat、LessFlat四种特征点云
  */
  /********Relative Variables*********/
  pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan;
  pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScanDS;
  /***********************************/
  void extractFeatures(ScanPtr scan) {
    cloud_msgs::cloud_info::Ptr segInfo = scan->cloudInfo_;

    scan->cornerPointsSharp_->clear();
    scan->cornerPointsLessSharp_->clear();
    scan->surfPointsFlat_->clear();
    scan->surfPointsLessFlat_->clear();

    for (int i = 0; i < LINE_NUM; i++) {
      surfPointsLessFlatScan->clear();

      for (int j = 0; j < 6; j++) {
        int sp = (segInfo->startRingIndex[i] * (6 - j) +
                  segInfo->endRingIndex[i] * j) / 6;
        int ep = (segInfo->startRingIndex[i] * (5 - j) +
                  segInfo->endRingIndex[i] * (j + 1)) /
                     6 - 1;

        if (sp >= ep) continue;

        std::sort(scan->cloudSmoothness_.begin() + sp,
                  scan->cloudSmoothness_.begin() + ep, byValue());

        int largestPickedNum = 0;
        for (int k = ep; k >= sp; k--) {
          int ind = scan->cloudSmoothness_[k].ind;
          if (scan->cloudNeighborPicked_[ind] == 0 &&
              scan->cloudCurvature_[ind] > EDGE_THRESHOLD &&
              segInfo->segmentedCloudGroundFlag[ind] == false) {
            largestPickedNum++;
            if (largestPickedNum <= 2) {
              scan->cloudLabel_[ind] = 2;
              scan->cornerPointsSharp_->push_back(
                  scan->undistPointCloud_->points[ind]);
              scan->cornerPointsLessSharp_->push_back(
                  scan->undistPointCloud_->points[ind]);
            } else if (largestPickedNum <= 20) {
              scan->cloudLabel_[ind] = 1;
              scan->cornerPointsLessSharp_->push_back(
                  scan->undistPointCloud_->points[ind]);
            } else {
              break;
            }

            scan->cloudNeighborPicked_[ind] = 1;
            for (int l = 1; l <= 5; l++) {
              int columnDiff =
                  std::abs(int(segInfo->segmentedCloudColInd[ind + l] -
                               segInfo->segmentedCloudColInd[ind + l - 1]));
              if (columnDiff > 10) break;
              scan->cloudNeighborPicked_[ind + l] = 1;
            }
            for (int l = -1; l >= -5; l--) {
              int columnDiff =
                  std::abs(int(segInfo->segmentedCloudColInd[ind + l] -
                               segInfo->segmentedCloudColInd[ind + l + 1]));
              if (columnDiff > 10) break;
              scan->cloudNeighborPicked_[ind + l] = 1;
            }
          }
        }

        int smallestPickedNum = 0;
        for (int k = sp; k <= ep; k++) {
          int ind = scan->cloudSmoothness_[k].ind;
          if (scan->cloudNeighborPicked_[ind] == 0 &&
              scan->cloudCurvature_[ind] < SURF_THRESHOLD &&
              segInfo->segmentedCloudGroundFlag[ind] == true) {
            scan->cloudLabel_[ind] = -1;
            scan->surfPointsFlat_->push_back(
                scan->undistPointCloud_->points[ind]);
            smallestPickedNum++;
            if (smallestPickedNum >= 4) {
              break;
            }

            scan->cloudNeighborPicked_[ind] = 1;
            for (int l = 1; l <= 5; l++) {
              int columnDiff =
                  std::abs(int(segInfo->segmentedCloudColInd[ind + l] -
                               segInfo->segmentedCloudColInd[ind + l - 1]));
              if (columnDiff > 10) break;

              scan->cloudNeighborPicked_[ind + l] = 1;
            }
            for (int l = -1; l >= -5; l--) {
              int columnDiff =
                  std::abs(int(segInfo->segmentedCloudColInd[ind + l] -
                               segInfo->segmentedCloudColInd[ind + l + 1]));
              if (columnDiff > 10) break;

              scan->cloudNeighborPicked_[ind + l] = 1;
            }
          }
        }

        for (int k = sp; k <= ep; k++) {
          if (scan->cloudLabel_[k] <= 0) {
            surfPointsLessFlatScan->push_back(
                scan->undistPointCloud_->points[k]);
          }
        }
      }
      surfPointsLessFlatScanDS->clear();
      downSizeFilter_.setInputCloud(surfPointsLessFlatScan);
      downSizeFilter_.filter(*surfPointsLessFlatScanDS);
      *(scan->surfPointsLessFlat_) += *surfPointsLessFlatScanDS;
    }
  }

  void findCorrespondingSurfFeatures(
      ScanPtr lastScan, ScanPtr newScan,
      pcl::PointCloud<PointType>::Ptr keypoints,
      pcl::PointCloud<PointType>::Ptr jacobianCoff, int iterCount) {
    int surfPointsFlatNum = newScan->surfPointsFlat_->points.size();

    for (int i = 0; i < surfPointsFlatNum; i++) {
      PointType pointSel;
      PointType coeff, tripod1, tripod2, tripod3;

      transformToStart(&newScan->surfPointsFlat_->points[i], &pointSel);

      pcl::PointCloud<PointType>::Ptr laserCloudSurfLast =
          lastScan->surfPointsLessFlat_;

      if (iterCount % ICP_FREQ == 0) {
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        kdtreeSurf_->nearestKSearch(pointSel, 1, pointSearchInd,
                                    pointSearchSqDis);
        int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;

        if (pointSearchSqDis[0] < NEAREST_FEATURE_SEARCH_SQ_DIST) {
          closestPointInd = pointSearchInd[0];
          int closestPointScan =
              int(laserCloudSurfLast->points[closestPointInd].intensity);

          float pointSqDis, minPointSqDis2 = NEAREST_FEATURE_SEARCH_SQ_DIST,
                            minPointSqDis3 = NEAREST_FEATURE_SEARCH_SQ_DIST;

          for (int j = closestPointInd + 1; j < surfPointsFlatNum; j++) {
            if (int(laserCloudSurfLast->points[j].intensity) >
                closestPointScan + 2.5) {
              break;
            }

            pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                             (laserCloudSurfLast->points[j].x - pointSel.x) +
                         (laserCloudSurfLast->points[j].y - pointSel.y) *
                             (laserCloudSurfLast->points[j].y - pointSel.y) +
                         (laserCloudSurfLast->points[j].z - pointSel.z) *
                             (laserCloudSurfLast->points[j].z - pointSel.z);
            if (int(laserCloudSurfLast->points[j].intensity) <=
                closestPointScan) {
              if (pointSqDis < minPointSqDis2) {
                minPointSqDis2 = pointSqDis;
                minPointInd2 = j;
              }
            } else {
              if (pointSqDis < minPointSqDis3) {
                minPointSqDis3 = pointSqDis;
                minPointInd3 = j;
              }
            }
          }

          for (int j = closestPointInd - 1; j >= 0; j--) {
            if (int(laserCloudSurfLast->points[j].intensity) <
                closestPointScan - 2.5) {
              break;
            }

            pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                             (laserCloudSurfLast->points[j].x - pointSel.x) +
                         (laserCloudSurfLast->points[j].y - pointSel.y) *
                             (laserCloudSurfLast->points[j].y - pointSel.y) +
                         (laserCloudSurfLast->points[j].z - pointSel.z) *
                             (laserCloudSurfLast->points[j].z - pointSel.z);

            if (int(laserCloudSurfLast->points[j].intensity) >=
                closestPointScan) {
              if (pointSqDis < minPointSqDis2) {
                minPointSqDis2 = pointSqDis;
                minPointInd2 = j;
              }
            } else {
              if (pointSqDis < minPointSqDis3) {
                minPointSqDis3 = pointSqDis;
                minPointInd3 = j;
              }
            }
          }
        }
        pointSearchSurfInd1[i] = closestPointInd;
        pointSearchSurfInd2[i] = minPointInd2;
        pointSearchSurfInd3[i] = minPointInd3;
      }

      if (pointSearchSurfInd2[i] >= 0 && pointSearchSurfInd3[i] >= 0) {
        tripod1 = laserCloudSurfLast->points[pointSearchSurfInd1[i]];
        tripod2 = laserCloudSurfLast->points[pointSearchSurfInd2[i]];
        tripod3 = laserCloudSurfLast->points[pointSearchSurfInd3[i]];

        V3D P0xyz(pointSel.x, pointSel.y, pointSel.z);
        V3D P1xyz(tripod1.x, tripod1.y, tripod1.z);
        V3D P2xyz(tripod2.x, tripod2.y, tripod2.z);
        V3D P3xyz(tripod3.x, tripod3.y, tripod3.z);

        V3D M = math_utils::skew(P1xyz - P2xyz) * (P1xyz - P3xyz);
        double r = (P0xyz - P1xyz).transpose() * M;
        double m = M.norm();
        float res = r / m;

        V3D jacxyz = M.transpose() / (m);

        float s = 1;
        if (iterCount >= ICP_FREQ) {
          s = 1 -
              1.8 * fabs(res) /
                  sqrt(sqrt(pointSel.x * pointSel.x + pointSel.y * pointSel.y +
                            pointSel.z * pointSel.z));
        }

        if (s > 0.1 && res != 0) {
          coeff.x = s * jacxyz(0);
          coeff.y = s * jacxyz(1);
          coeff.z = s * jacxyz(2);
          coeff.intensity = s * res;

          keypoints->push_back(newScan->surfPointsFlat_->points[i]);
          jacobianCoff->push_back(coeff);
        }
      }
    }
  }

  /**
   * 遍历Corner特征点云，基于linState_提供的帧间位姿变换量，对点进行帧内校正。
   * 然后在上一帧Corner特征点云中找到当前点p的最近点a和次近点b，再获得点p到线段
   * ab的距离，也就是观测方程的结果，并求得观测方程的雅可比，保存在jacobianCoff
   * 中，对应的当前点p保存在keypoints中。
   * @param lastScan 上一帧特征点云
   * @param newScan 当前特征点云
   * @param keypoints 在上一帧点云中找到了最近点的当前点，也就是点p
   * @param jacobianCoff 每个点的intensity保存点p到线段ab的距离，也就是观测方程
   *   的结果; 每个点的x,y,z中保存观测方程的雅可比向量。
   * @param iterCount 迭代次数
   * @param linState_ 这个参数是全局变量，因此没有出现在参数表中，但是非常重要，
   *	保存了两帧点云间的位姿变换量。
   * @return 无
   */
  void findCorrespondingCornerFeatures(
      ScanPtr lastScan, ScanPtr newScan,
      pcl::PointCloud<PointType>::Ptr keypoints,
      pcl::PointCloud<PointType>::Ptr jacobianCoff, int iterCount) {
    int cornerPointsSharpNum = newScan->cornerPointsSharp_->points.size();

	/* 遍历Corner特征点中的每一个点 */
    for (int i = 0; i < cornerPointsSharpNum; i++) {
      PointType pointSel;
      PointType coeff, tripod1, tripod2;

	  /* 基于linState_提供的帧间位姿变换量，对点进行帧内校正，对齐到帧内第一个点 */
      transformToStart(&newScan->cornerPointsSharp_->points[i], &pointSel);

      pcl::PointCloud<PointType>::Ptr laserCloudCornerLast =
          lastScan->cornerPointsLessSharp_;

      if (iterCount % ICP_FREQ == 0) {
		/* 在上一帧点云中找到与当前点最近的一个点， kdtreeCorner_是上一帧点云*/
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        kdtreeCorner_->nearestKSearch(pointSel, 1, pointSearchInd,
                                      pointSearchSqDis);
        int closestPointInd = -1, minPointInd2 = -1;

		/* 在最近的的基础上，遍历最近点附近的点，查找次近点 */
        if (pointSearchSqDis[0] < NEAREST_FEATURE_SEARCH_SQ_DIST) {
		  /* 记录最近点的index */
          closestPointInd = pointSearchInd[0];
		  /* 记录最近点的时间戳 */
          int closestPointScan =
              int(laserCloudCornerLast->points[closestPointInd].intensity);

		  /* 从最近点开始，向后遍历上一帧点云中的点，找到与当前点最近的点，记录序号 */
          float pointSqDis, minPointSqDis2 = NEAREST_FEATURE_SEARCH_SQ_DIST;
          for (int j = closestPointInd + 1; j < cornerPointsSharpNum; j++) {
			  
			/* 如果该点距离最近点的时间戳超过2.5，则停止遍历 */
            if (int(laserCloudCornerLast->points[j].intensity) >
                closestPointScan + 2.5) {
              break;
            }

			/* 求该点与当前点的平方和，也就是距离平方 */
            pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                         (laserCloudCornerLast->points[j].x - pointSel.x) +
                         (laserCloudCornerLast->points[j].y - pointSel.y) *
                         (laserCloudCornerLast->points[j].y - pointSel.y) +
                         (laserCloudCornerLast->points[j].z - pointSel.z) *
                         (laserCloudCornerLast->points[j].z - pointSel.z);

			/* 记录与当前点距离最小的点的序号在minPointInd2中 */
            if (int(laserCloudCornerLast->points[j].intensity) >
                closestPointScan) {
              if (pointSqDis < minPointSqDis2) {
                minPointSqDis2 = pointSqDis;
                minPointInd2 = j;
              }
            }
          }
		  
		  /* 从最近点开始，向前遍历上一帧点云中的点，找到与当前点最近的点，记录序号 */
          for (int j = closestPointInd - 1; j >= 0; j--) {
            if (int(laserCloudCornerLast->points[j].intensity) <
                closestPointScan - 2.5) {
              break;
            }

            pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                         (laserCloudCornerLast->points[j].x - pointSel.x) +
                         (laserCloudCornerLast->points[j].y - pointSel.y) *
                         (laserCloudCornerLast->points[j].y - pointSel.y) +
                         (laserCloudCornerLast->points[j].z - pointSel.z) *
                         (laserCloudCornerLast->points[j].z - pointSel.z);

            if (int(laserCloudCornerLast->points[j].intensity) <
                closestPointScan) {
              if (pointSqDis < minPointSqDis2) {
                minPointSqDis2 = pointSqDis;
                minPointInd2 = j;
              }
            }
          }
        }

		/* 通过kdtree找到的最近点 */
        pointSearchCornerInd1[i] = closestPointInd;
		/* 通过遍历最近点附近的点找到的次近点 */
        pointSearchCornerInd2[i] = minPointInd2;
      }

      if (pointSearchCornerInd2[i] >= 0) {
        tripod1 = laserCloudCornerLast->points[pointSearchCornerInd1[i]];
        tripod2 = laserCloudCornerLast->points[pointSearchCornerInd2[i]];

		/* P0是当前帧点云中的当前点，P1和P2是上一帧点云中的最近点和次近点，构成线段P1P2 */
        V3D P0xyz(pointSel.x, pointSel.y, pointSel.z);
        V3D P1xyz(tripod1.x, tripod1.y, tripod1.z);
        V3D P2xyz(tripod2.x, tripod2.y, tripod2.z);

		/* 求线段P0P1和P0P2的叉乘结果 */
        V3D P = math_utils::skew(P0xyz - P1xyz) * (P0xyz - P2xyz);
		/* r是平行四边形的面积 */
        float r = P.norm();
		/* 求P1P2长度 */
        float d12 = (P1xyz - P2xyz).norm();
		/* 平行四边形面积除以P1P2长度，得到点P0到线段P1P2的距离 */
        float res = r / d12;

		/* 叉乘向量P再次和P2P1叉乘，得到点P0到线段P1P2的垂线 */
        V3D jacxyz =
            P.transpose() * math_utils::skew(P2xyz - P1xyz) / (d12 * r);

		/* 设置阻尼因子 */
        float s = 1;
        if (iterCount >= ICP_FREQ) {
          s = 1 - 1.8 * fabs(res);
        }

        if (s > 0.1 && res != 0) {
          coeff.x = s * jacxyz(0);
          coeff.y = s * jacxyz(1);
          coeff.z = s * jacxyz(2);
          coeff.intensity = s * res; /* intensity记录当前点P到特征点线段ab的距离，带阻尼 */

		  /* keypoints记录哪些在上一帧点云中找到了最近点的当前点 */
          keypoints->push_back(newScan->cornerPointsSharp_->points[i]);
		  /* jacobianCoff记录当前点p到上一帧点云最近点线段ab的垂线向量，以及距离 */
          jacobianCoff->push_back(coeff);
        }
      }
    }
  }

 /**
  * 完成点云的帧内校正，将所有点对齐到帧内第一个点
  */
  // Undistort point cloud to the start frame
  void transformToStart(PointType const* const pi, PointType* const po) {
	  
	/* 获得当前点相对于整个点云帧的时间比例系数，intensity的小数部分存储的
	 * 是每个点的时间戳 */
    double s = (1.f / SCAN_PERIOD) * (pi->intensity - int(pi->intensity));

	/* 构造点向量 */
    V3D P2xyz(pi->x, pi->y, pi->z);
	/* 将四元数形式的旋转量转成类似轴角的向量，乘以时间比例系数，再还原成四元数R21xyz */
    V3D phi = Quat2axis(linState_.qbn_);
    Q4D R21xyz = axis2Quat(s * phi);
    R21xyz.normalized();
	/* 平移量乘以时间比例系数 */
    V3D T112xyz = s * linState_.rn_;
	/* 对当前点进行旋转平移变换 */
    V3D P1xyz = R21xyz * P2xyz + T112xyz;

    po->x = P1xyz.x();
    po->y = P1xyz.y();
    po->z = P1xyz.z();
	/* 输出点的intensity仍然记录对应的时间戳 */
    po->intensity = pi->intensity;
  }

  /**
   * 完成点云的帧内校正，将所有点对齐到帧内最后一个点
   */
  // Undistort point cloud to the end frame
  void transformToEnd(PointType const* const pi, PointType* const po) {
    double s = (1.f / SCAN_PERIOD) * (pi->intensity - int(pi->intensity));

    V3D P2xyz(pi->x, pi->y, pi->z);
    V3D phi = Quat2axis(linState_.qbn_);
    Q4D R21xyz = axis2Quat(s * phi);
    R21xyz.normalized();
    V3D T112xyz = s * linState_.rn_;
    V3D P1xyz = R21xyz * P2xyz + T112xyz;

    R21xyz = linState_.qbn_;
    T112xyz = linState_.rn_;
    P2xyz = R21xyz.inverse() * (P1xyz - T112xyz);

    po->x = P2xyz.x();
    po->y = P2xyz.y();
    po->z = P2xyz.z();
    po->intensity = pi->intensity;
  }

  // Coordinate transformation from LiDAR frame to Vehicle frame
  void rotatePoint(PointType const* const pi, PointType* const po) {
    V3D rpy;
    rpy << deg2rad(0.0), deg2rad(0.0), deg2rad(IMU_LIDAR_EXTRINSIC_ANGLE);
    M3D R = rpy2R(rpy);
    V3D Pi(pi->x, pi->y, pi->z);
    V3D Po = R * Pi;
    po->x = Po.x();
    po->y = Po.y();
    po->z = Po.z();
    po->intensity = pi->intensity;
  }

  /** 
   * 对当前帧特征点云进行变换，使所有点向帧内最后一个点看齐
   * 然后添加到kdTree中，为与下一帧点云的匹配做好准备
   */
  void updatePointCloud() {
    scan_new_->cornerPointsLessSharpYZX_->clear();
    scan_new_->surfPointsLessFlatYZX_->clear();
    scan_new_->outlierPointCloudYZX_->clear();

	/** 
     * 对当前帧特征点云进行帧内校正，使所有点向最后一个点看齐
	 */
    PointType point;
    for (int i = 0; i < scan_new_->cornerPointsLessSharp_->points.size(); i++) {
      transformToEnd(&scan_new_->cornerPointsLessSharp_->points[i],
                     &scan_new_->cornerPointsLessSharp_->points[i]);
      point.x = scan_new_->cornerPointsLessSharp_->points[i].y;
      point.y = scan_new_->cornerPointsLessSharp_->points[i].z;
      point.z = scan_new_->cornerPointsLessSharp_->points[i].x;
      point.intensity = scan_new_->cornerPointsLessSharp_->points[i].intensity;
      scan_new_->cornerPointsLessSharpYZX_->push_back(point);
    }
    for (int i = 0; i < scan_new_->surfPointsLessFlat_->points.size(); i++) {
      transformToEnd(&scan_new_->surfPointsLessFlat_->points[i],
                     &scan_new_->surfPointsLessFlat_->points[i]);
      point.x = scan_new_->surfPointsLessFlat_->points[i].y;
      point.y = scan_new_->surfPointsLessFlat_->points[i].z;
      point.z = scan_new_->surfPointsLessFlat_->points[i].x;
      point.intensity = scan_new_->surfPointsLessFlat_->points[i].intensity;
      scan_new_->surfPointsLessFlatYZX_->push_back(point);
    }
    for (int i = 0; i < scan_new_->outlierPointCloud_->points.size(); i++) {
      // transformToEnd(&scan_new_->outlierPointCloud_->points[i],
      //                &scan_new_->outlierPointCloud_->points[i]);
      point.x = scan_new_->outlierPointCloud_->points[i].y;
      point.y = scan_new_->outlierPointCloud_->points[i].z;
      point.z = scan_new_->outlierPointCloud_->points[i].x;
      point.intensity = scan_new_->outlierPointCloud_->points[i].intensity;
      scan_new_->outlierPointCloudYZX_->push_back(point);
    }

    // Transform XYZ-convention to YZX-convention to meet the mapping module's
    // requirement
    globalStateYZX_.rn_ = Q_xyz_to_yzx * globalState_.rn_;
    globalStateYZX_.qbn_ =
        Q_xyz_to_yzx * globalState_.qbn_ * Q_xyz_to_yzx.inverse();

	/** 
     * 将已经完成转换的特征点云添加到kdTree中，以便下一帧点云进行位姿匹配
	 */
    if (scan_new_->cornerPointsLessSharp_->points.size() >= 5 &&
        scan_new_->surfPointsLessFlat_->points.size() >= 20) {
      kdtreeCorner_->setInputCloud(scan_new_->cornerPointsLessSharp_);
      kdtreeSurf_->setInputCloud(scan_new_->surfPointsLessFlat_);
    }
  }

  void estimateTransform(ScanPtr lastScan, ScanPtr newScan, V3D& t, Q4D& q) {
    double sum_dt = preintegration_->sum_dt;
    linState_.rn_ = t;
    linState_.qbn_ = q;
    for (int iter = 0; iter < NUM_ITER; iter++) {
      keypointSurfs_->clear();
      jacobianCoffSurfs->clear();
      keypointCorns_->clear();
      jacobianCoffCorns->clear();

      findCorrespondingSurfFeatures(lastScan, newScan, keypointSurfs_,
                                    jacobianCoffSurfs, iter);
      if (keypointSurfs_->points.size() < 10) {
        ROS_WARN("Insufficient matched surfs...");
        continue;
      }
	  
	  /* 在上一帧点云中与当前 */
      findCorrespondingCornerFeatures(lastScan, newScan, keypointCorns_,
                                      jacobianCoffCorns, iter);
      if (keypointCorns_->points.size() < 5) {
        ROS_WARN("Insufficient matched corners...");
        continue;
      }

      if (calculateTransformation(lastScan, newScan, keypointCorns_,
                                  jacobianCoffCorns, keypointSurfs_,
                                  jacobianCoffSurfs, iter)) {
        ROS_INFO_STREAM("System Converges after " << iter << " iterations");
        break;
      }
    }

    t = linState_.rn_;
    q = linState_.qbn_;  // qbn_ is quaternion rotation from b-frame to n-frame
  }

  bool calculateTransformation(ScanPtr lastScan, ScanPtr newScan,
                               pcl::PointCloud<PointType>::Ptr corners,
                               pcl::PointCloud<PointType>::Ptr jacoCornersCoff,
                               pcl::PointCloud<PointType>::Ptr surfs,
                               pcl::PointCloud<PointType>::Ptr jacoSurfsCoff,
                               int iterCount) {
    keypoints_->clear();
    jacobians_->clear();
    (*keypoints_) += (*surfs);
    (*keypoints_) += (*corners);
    (*jacobians_) += (*jacoSurfsCoff);
    (*jacobians_) += (*jacoCornersCoff);

    const int stateNum = 6;
    const int pointNum = keypoints_->points.size();
    const int imuNum = 0;
    const int row = pointNum + imuNum;
    Eigen::Matrix<double, Eigen::Dynamic, stateNum> J(row, stateNum);
    Eigen::Matrix<double, stateNum, Eigen::Dynamic> JT(stateNum, row);
    Eigen::Matrix<double, stateNum, stateNum> JTJ;
    Eigen::VectorXd b(row);
    Eigen::Matrix<double, stateNum, 1> JTb;
    Eigen::Matrix<double, stateNum, 1> x;
    J.setZero();
    JT.setZero();
    JTJ.setZero();
    b.setZero();
    JTb.setZero();
    x.setZero();

    for (int i = 0; i < pointNum; ++i) {
      // Select keypoint i
      const PointType& keypoint = keypoints_->points[i];
      const PointType& coeff = jacobians_->points[i];

      V3D P2xyz(keypoint.x, keypoint.y, keypoint.z);
      V3D coff_xyz(coeff.x, coeff.y, coeff.z);

      double s =
          (1.f / SCAN_PERIOD) * (keypoint.intensity - int(keypoint.intensity));

      V3D phi = Quat2axis(linState_.qbn_);
      // Rotation matrix from frame2 (new) to frame1 (last)
      Q4D R21xyz = axis2Quat(s * phi);
      R21xyz.normalized();
      // Translation vector from frame1 to frame2 represented in frame1
      V3D T112xyz = s * linState_.rn_;

      V3D jacobian1xyz =
          coff_xyz.transpose() *
          (-R21xyz.toRotationMatrix() * skew(P2xyz));  // rotation jacobian
      V3D jacobian2xyz =
          coff_xyz.transpose() * M3D::Identity();  // translation jacobian
      double residual = coeff.intensity;

      J.block<1, 3>(i, O_R) = jacobian1xyz;
      J.block<1, 3>(i, O_P) = jacobian2xyz;

      // Set the overall residual
      b(i) = -0.05 * residual;
    }

    // Solve x
    JT = J.transpose();
    JTJ = JT * J;
    JTb = JT * b;
    x = JTJ.colPivHouseholderQr().solve(JTb);

    // Determine whether x is degenerated
    bool isDegenerate = false;
    Eigen::Matrix<double, stateNum, stateNum> matP;
    if (iterCount == 0) {
      Eigen::Matrix<double, 1, stateNum> matE;
      Eigen::Matrix<double, stateNum, stateNum> matV;
      Eigen::Matrix<double, stateNum, stateNum> matV2;

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, stateNum, stateNum> >
          esolver(JTJ);
      matE = esolver.eigenvalues().real();
      matV = esolver.eigenvectors().real();

      matV2 = matV;

      isDegenerate = false;
      std::vector<double> eignThre(stateNum, 10.);
      for (int i = 0; i < stateNum; i++) {
        // if eigenvalue is less than 10, set the corresponding eigenvector to 0
        // vector
        if (matE(0, i) < eignThre[i]) {
          for (int j = 0; j < stateNum; j++) {
            matV2(i, j) = 0;
          }
          isDegenerate = true;
        } else {
          break;
        }
      }
      matP = matV.inverse() * matV2;
    }

    if (isDegenerate) {
      cout << "System is Degenerate." << endl;
      Eigen::Matrix<double, stateNum, 1> matX2(x);
      x = matP * matX2;
    }

    // Update state linState_
    Q4D dq = rpy2Quat(x.segment<3>(O_R));
    linState_.qbn_ = (linState_.qbn_ * dq).normalized();
    linState_.rn_ += x.segment<3>(O_P);

    // Determine whether should it stop
    V3D rpy_rad = x.segment<3>(O_R);
    V3D rpy_deg = math_utils::rad2deg(rpy_rad);
    double deltaR = rpy_deg.norm();
    V3D trans = 100 * x.segment<3>(O_P);
    double deltaT = trans.norm();
    if (deltaR < 0.1 && deltaT < 0.1) {
      return true;
    }

    return false;
  }

  void estimateInitialState1(const V3D& p, const Q4D& q, V3D& v0, V3D& v1,
                             V3D& ba, V3D& bw) {
    ba = INIT_BA;
    bw = INIT_BW;

    solveGyroscopeBias(q, bw);

    double sum_dt = preintegration_->sum_dt;
    v0 =
        (p - 0.5 * linState_.gn_ * sum_dt * sum_dt - preintegration_->delta_p) /
        sum_dt;
    v1 = v0 + sum_dt * linState_.gn_ + preintegration_->delta_v;

    cout << "v0: " << v0.transpose() << endl;
    cout << "v1: " << v1.transpose() << endl;
    cout << "ba0: " << INIT_BA.transpose() << endl;
    cout << "bw0: " << INIT_BW.transpose() << endl;
    cout << "bw0: " << bw.transpose() << endl;
  }

  void estimateInitialState2(const V3D& p, const Q4D& q, V3D& v0, V3D& v1,
                             V3D& ba, V3D& bw) {
    const int DIM_OF_STATE = 1 + 1 + 3;
    const int DIM_OF_MEAS = 3 + 3;
    Eigen::Matrix<double, Eigen::Dynamic, DIM_OF_STATE> J(DIM_OF_MEAS,
                                                          DIM_OF_STATE);
    Eigen::Matrix<double, DIM_OF_STATE, Eigen::Dynamic> JT(DIM_OF_STATE,
                                                           DIM_OF_MEAS);
    Eigen::Matrix<double, DIM_OF_STATE, DIM_OF_STATE> JTJ;
    Eigen::VectorXd b(DIM_OF_MEAS);
    Eigen::Matrix<double, DIM_OF_STATE, 1> JTb;
    Eigen::Matrix<double, DIM_OF_STATE, 1> x;

    J.setZero();
    JT.setZero();
    JTJ.setZero();
    b.setZero();
    JTb.setZero();
    x.setZero();

    double sum_dt = preintegration_->sum_dt;
    b.block<3, 1>(0, 0) =
        (p - preintegration_->delta_p) / sum_dt - 0.5 * sum_dt * linState_.gn_;
    b.block<3, 1>(3, 0) = sum_dt * linState_.gn_ + preintegration_->delta_v;

    V3D L(1, 0, 0);
    J.block<3, 1>(0, 0) = L;
    J.block<3, 3>(0, 2) = -0.5 * sum_dt * M3D::Identity();
    J.block<3, 1>(3, 0) = -L;
    J.block<3, 1>(3, 1) = L;
    J.block<3, 3>(3, 2) = sum_dt * M3D::Identity();

    JT = J.transpose();
    JTJ = JT * J;
    JTb = JT * b;

    x = JTJ.colPivHouseholderQr().solve(JTb);
    v0 = x(0) * L;
    v1 = x(1) * L;
    ba = INIT_BA;
    bw = INIT_BW;

    V3D test_ba = linState_.gn_ + preintegration_->delta_v / sum_dt;
    cout << "test_ba: " << test_ba.transpose() << endl;
  }

  void estimateInitialState3(const V3D& p, const Q4D& q, V3D& v0, V3D& v1,
                             V3D& ba, V3D& bw) {
    double sum_dt = preintegration_->sum_dt;
    V3D v = p / sum_dt;
    // v * sum_dt = (p - 0.5*linState_.gn_*sum_dt*sum_dt -
    // preintegration_->delta_p + 0.5*ba*sum_dt*sum_dt);
    ba = (v * sum_dt - p + 0.5 * linState_.gn_ * sum_dt * sum_dt +
          preintegration_->delta_p) *
         2 * (1.0 / sum_dt * sum_dt);

    solveGyroscopeBias(q, bw);

    v0 = v;
    v1 = v;
    cout << "v0: " << v0.transpose() << endl;
    cout << "v1: " << v1.transpose() << endl;
    cout << "ba0: " << ba.transpose() << endl;
    cout << "bw0: " << bw.transpose() << endl;
  }

  void estimateInitialState(const V3D& p, const Q4D& q, V3D& v0, V3D& v1,
                            V3D& ba, V3D& bw) {
    double sum_dt = preintegration_->sum_dt;
    // Calculate a rough velocity using relative translation
    V3D v = p / sum_dt;
    v0 = v;
    v1 = v;
    // TODO(charles): calibrate initial ba and bw using two consecutive scans
    // and IMU preintegration results
    ba = INIT_BA;
    bw = INIT_BW;
  }

  // Estimate gyroscope bias using a similar methoed provided in VINS-Mono
  void solveGyroscopeBias(const Q4D& q, V3D& bw) {
    Matrix3d A;
    V3D b;
    V3D delta_bg;
    A.setZero();
    b.setZero();

    MatrixXd tmp_A(3, 3);
    tmp_A.setZero();
    VectorXd tmp_b(3);
    tmp_b.setZero();
    Eigen::Quaterniond q_ij = q;
    tmp_A = preintegration_->jacobian.template block<3, 3>(GlobalState::att_,
                                                           GlobalState::gyr_);
    tmp_b = 2 * (preintegration_->delta_q.inverse() * q_ij).vec();
    A += tmp_A.transpose() * tmp_A;
    b += tmp_A.transpose() * tmp_b;

    delta_bg = A.ldlt().solve(b);
    ROS_WARN_STREAM("gyroscope bias initial calibration "
                    << delta_bg.transpose());

    bw += delta_bg;
  }

 public:
  FusionStatus status_;     // system status
  StatePredictor* filter_;  // Kalman filter pointer
  ScanPtr scan_new_;        // current scan information
  ScanPtr scan_last_;       // last scan information

  // !@KD tree relatives
  pcl::VoxelGrid<PointType> downSizeFilter_;
  pcl::KdTreeFLANN<PointType>::Ptr kdtreeCorner_;
  pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurf_;

  // !@Feature matching relatives
  std::vector<int> pointSelCornerInd;
  std::vector<double> pointSearchCornerInd1;
  std::vector<double> pointSearchCornerInd2;
  std::vector<int> pointSelSurfInd;
  std::vector<double> pointSearchSurfInd1;
  std::vector<double> pointSearchSurfInd2;
  std::vector<double> pointSearchSurfInd3;

  // !@Jacobians and keypoints
  pcl::PointCloud<PointType>::Ptr keypoints_;
  pcl::PointCloud<PointType>::Ptr jacobians_;
  pcl::PointCloud<PointType>::Ptr keypointCorns_;
  pcl::PointCloud<PointType>::Ptr keypointSurfs_;
  pcl::PointCloud<PointType>::Ptr jacobianCoffCorns;
  pcl::PointCloud<PointType>::Ptr jacobianCoffSurfs;

  // !@Global transformation from the original scan-frame to current scan-frame
  GlobalState globalState_;
  // !@Relative transformation from scan0-frame t0 scan1-frame
  GlobalState linState_;
  Eigen::Matrix<double, GlobalState::DIM_OF_STATE_, 1> difVecLinInv_;
  Eigen::Matrix<double, GlobalState::DIM_OF_STATE_, 1> updateVec_;
  double updateVecNorm_ = 0.0;

  // !@Kalman filter relatives
  VXD residual_;
  MXD Fk_;
  MXD Gk_;
  MXD Pk_;
  MXD Qk_;
  MXD Rk_;
  MXD Hk_;
  MXD Jk_;
  MXD Kk_;
  MXD IKH_;
  MXD Py_;
  MXD Pyinv_;

  // !@ IMU preintegration
  integration::IntegrationBase* preintegration_;
  Imu imu_last_;

  // !@Rotation matrices between XYZ-convention and YZX-convention
  Eigen::Matrix3d R_yzx_to_xyz;
  Eigen::Matrix3d R_xyz_to_yzx;
  Eigen::Quaterniond Q_yzx_to_xyz;
  Eigen::Quaterniond Q_xyz_to_yzx;
  GlobalState globalStateYZX_;
};

}  // namespace fusion

#endif  // INCLUDE_STATEESTIMATOR_HPP_
