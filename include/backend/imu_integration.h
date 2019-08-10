#pragma once

#include "eigen_types.h"
#include "../thirdparty/Sophus/sophus/se3.hpp"

namespace myslam {
namespace backend {

class IMUIntegration {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /**
     * constructor, with initial bias a and bias g
     * @param ba
     * @param bg
     */
    explicit IMUIntegration(const Vec3 &ba, const Vec3 &bg) : ba_(ba), bg_(bg) {
        const Mat33 i3 = Mat33::Identity();
        noise_measurement_.block<3, 3>(0, 0) = (acc_noise_ * acc_noise_) * i3;
        noise_measurement_.block<3, 3>(3, 3) = (gyr_noise_ * gyr_noise_) * i3;
        noise_random_walk_.block<3, 3>(0, 0) = (acc_random_walk_ * acc_random_walk_) * i3;
        noise_random_walk_.block<3, 3>(3, 3) = (gyr_random_walk_ * gyr_random_walk_) * i3;
    }

    ~IMUIntegration() {}

    /**
     * propage pre-integrated measurements using raw IMU data
     * @param dt
     * @param acc
     * @param gyr_1
     */
    void Propagate(double dt, const Vec3 &acc, const Vec3 &gyr);

    /**
     * according to pre-integration, when bias is updated, pre-integration should also be updated using
     * first-order expansion of ba and bg
     *
     * @param delta_ba
     * @param delta_bg
     */
    void Correct(const Vec3 &delta_ba, const Vec3 &delta_bg);

    void SetBiasG(const Vec3 &bg) { bg_ = bg; }

    void SetBiasA(const Vec3 &ba) { ba_ = ba; }

    /// if bias is update by a large value, redo the propagation
    void Repropagate();

    /// reset measurements
    /// NOTE ba and bg will not be reset, only measurements and jacobians will be reset!
    void Reset() {
        sum_dt_ = 0;
        delta_r_ = Sophus::SO3d();  // dR
        delta_v_ = Vec3::Zero();    // dv
        delta_p_ = Vec3::Zero();    // dp

        // jacobian w.r.t bg and ba
        dr_dbg_ = Mat33::Zero();
        dv_dbg_ = Mat33::Zero();
        dv_dba_ = Mat33::Zero();
        dp_dbg_ = Mat33::Zero();
        dp_dba_ = Mat33::Zero();

        // noise propagation
        covariance_measurement_ = Mat99::Zero();
        covariance_random_walk_ = Mat66::Zero();
        A_ = Mat99::Zero();
        B_ = Mat96::Zero();
    }

    /**
     * get the jacobians from r,v,p w.r.t. biases
     * @param _dr_dbg
     * @param _dv_dbg
     * @param _dv_dba
     * @param _dp_dbg
     * @param _dp_dba
     */
    void GetJacobians(Mat33 &dr_dbg, Mat33 &dv_dbg, Mat33 &dv_dba, Mat33 &dp_dbg, Mat33 &dp_dba) const {
        dr_dbg = dr_dbg_;
        dv_dbg = dv_dbg_;
        dv_dba = dv_dba_;
        dp_dbg = dp_dbg_;
        dp_dba = dp_dba_;
    }

    Mat33 GetDrDbg() const { return dr_dbg_; }

    /// get propagated noise covariance
    Mat99 GetCovarianceMeasurement() const {
        return covariance_measurement_;
    }

    /// get random walk covariance
    Mat66 GetCovarianceRandomWalk() const {
        return noise_random_walk_ * sum_dt_;
    }

    /// get sum of time
    double GetSumDt() const {
        return sum_dt_;
    }

    /**
     * get the integrated measurements
     * @param delta_r
     * @param delta_v
     * @param delta_p
     */
    void GetDeltaRVP(Sophus::SO3d &delta_r, Vec3 &delta_v, Vec3 &delta_p) const {
        delta_r = delta_r_;
        delta_v = delta_v_;
        delta_p = delta_p_;
    }

    Vec3 GetDv() const { return delta_v_; }

    Vec3 GetDp() const { return delta_p_; }

    Sophus::SO3d GetDr() const { return delta_r_; }

private:
    // raw data from IMU
    std::vector<double> dt_buf_;
    VecVec3 acc_buf_;
    VecVec3 gyr_buf_;

    // pre-integrated IMU measurements
    double sum_dt_ = 0;
    Sophus::SO3d delta_r_;  // dR
    Vec3 delta_v_ = Vec3::Zero();    // dv
    Vec3 delta_p_ = Vec3::Zero();    // dp

    // gravity, biases
    static Vec3 gravity_;
    Vec3 bg_ = Vec3::Zero();    // initial bias of gyro
    Vec3 ba_ = Vec3::Zero();    // initial bias of accelerator

    // jacobian w.r.t bg and ba
    Mat33 dr_dbg_ = Mat33::Zero();
    Mat33 dv_dbg_ = Mat33::Zero();
    Mat33 dv_dba_ = Mat33::Zero();
    Mat33 dp_dbg_ = Mat33::Zero();
    Mat33 dp_dba_ = Mat33::Zero();

    // noise propagation
    Mat99 covariance_measurement_ = Mat99::Zero();
    Mat66 covariance_random_walk_ = Mat66::Zero();
    Mat99 A_ = Mat99::Zero();
    Mat96 B_ = Mat96::Zero();

    // raw noise of imu measurement
    Mat66 noise_measurement_ = Mat66::Identity();
    Mat66 noise_random_walk_ = Mat66::Identity();

    /**@brief accelerometer measurement noise standard deviation*/
    constexpr static double acc_noise_ = 0.2;
    /**@brief gyroscope measurement noise standard deviation*/
    constexpr static double gyr_noise_ = 0.02;
    /**@brief accelerometer bias random walk noise standard deviation*/
    constexpr static double acc_random_walk_ = 0.0002;
    /**@brief gyroscope bias random walk noise standard deviation*/
    constexpr static double gyr_random_walk_ = 2.0e-5;
};

}
}

