//
// Created by heyijia on 19-1-30.
//

#include "../thirdparty/Sophus/sophus/se3.hpp"

#include "backend/edge_prior.h"
#include "backend/vertex_pose.h"

#include <iostream>

#define USE_SO3_JACOBIAN 1

namespace myslam {
namespace backend {

#ifndef USE_SO3_JACOBIAN
template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 3, 3> skewSymmetric(const Eigen::MatrixBase<Derived> &q)
{
    Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
    ans << typename Derived::Scalar(0), -q(2), q(1),
            q(2), typename Derived::Scalar(0), -q(0),
            -q(1), q(0), typename Derived::Scalar(0);
    return ans;
}

template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qleft(const Eigen::QuaternionBase<Derived> &q)
{
    Eigen::Quaternion<typename Derived::Scalar> qq = q;
    Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
    ans(0, 0) = qq.w(), ans.template block<1, 3>(0, 1) = -qq.vec().transpose();
    ans.template block<3, 1>(1, 0) = qq.vec(), ans.template block<3, 3>(1, 1) = qq.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() + skewSymmetric(qq.vec());
    return ans;
}
#endif

void EdgeSE3Prior::ComputeResidual() {
    VecX param_i = verticies_[0]->Parameters();
    Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
    Vec3 Pi = param_i.head<3>();

//    std::cout << Qi.vec().transpose() <<" "<< Qp_.vec().transpose()<<std::endl;
    // rotation error
#ifdef USE_SO3_JACOBIAN
    Sophus::SO3d ri(Qi);
    Sophus::SO3d rp(Qp_);
    Sophus::SO3d res_r = rp.inverse() * ri;
    residual_.block<3,1>(0,0) = Sophus::SO3d::log(res_r);
#else
    residual_.block<3,1>(0,0) = 2 * (Qp_.inverse() * Qi).vec();
#endif
    // translation error
    residual_.block<3,1>(3,0) = Pi - Pp_;
//    std::cout << residual_.transpose() <<std::endl;
}

void EdgeSE3Prior::ComputeJacobians() {

    VecX param_i = verticies_[0]->Parameters();
    Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);

    // w.r.t. pose i
    Eigen::Matrix<double, 6, 6> jacobian_pose_i = Eigen::Matrix<double, 6, 6>::Zero();

#ifdef USE_SO3_JACOBIAN
    Sophus::SO3d ri(Qi);
    Sophus::SO3d rp(Qp_);
    Sophus::SO3d res_r = rp.inverse() * ri;
    // http://rpg.ifi.uzh.ch/docs/RSS15_Forster.pdf  公式A.32
    jacobian_pose_i.block<3,3>(0,3) = Sophus::SO3d::JacobianRInv(res_r.log());
#else
    jacobian_pose_i.block<3,3>(0,3) = Qleft(Qp_.inverse() * Qi).bottomRightCorner<3, 3>();
#endif
    jacobian_pose_i.block<3,3>(3,0) = Mat33::Identity();

    jacobians_[0] = jacobian_pose_i;
//    std::cout << jacobian_pose_i << std::endl;
}

}
}