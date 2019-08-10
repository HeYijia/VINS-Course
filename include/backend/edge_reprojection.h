#ifndef MYSLAM_BACKEND_VISUALEDGE_H
#define MYSLAM_BACKEND_VISUALEDGE_H

#include <memory>
#include <string>

#include <Eigen/Dense>

#include "eigen_types.h"
#include "edge.h"

namespace myslam {
namespace backend {

/**
 * 此边是视觉重投影误差，此边为三元边，与之相连的顶点有：
 * 路标点的逆深度InveseDepth、第一次观测到该路标点的source Camera的位姿T_World_From_Body1，
 * 和观测到该路标点的mearsurement Camera位姿T_World_From_Body2。
 * 注意：verticies_顶点顺序必须为InveseDepth、T_World_From_Body1、T_World_From_Body2。
 */
class EdgeReprojection : public Edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeReprojection(const Vec3 &pts_i, const Vec3 &pts_j)
        : Edge(2, 4, std::vector<std::string>{"VertexInverseDepth", "VertexPose", "VertexPose", "VertexPose"}) {
        pts_i_ = pts_i;
        pts_j_ = pts_j;
    }

    /// 返回边的类型信息
    virtual std::string TypeInfo() const override { return "EdgeReprojection"; }

    /// 计算残差
    virtual void ComputeResidual() override;

    /// 计算雅可比
    virtual void ComputeJacobians() override;

//    void SetTranslationImuFromCamera(Eigen::Quaterniond &qic_, Vec3 &tic_);

private:
    //Translation imu from camera
//    Qd qic;
//    Vec3 tic;

    //measurements
    Vec3 pts_i_, pts_j_;
};

/**
* 此边是视觉重投影误差，此边为二元边，与之相连的顶点有：
* 路标点的世界坐标系XYZ、观测到该路标点的 Camera 的位姿T_World_From_Body1
* 注意：verticies_顶点顺序必须为 XYZ、T_World_From_Body1。
*/
class EdgeReprojectionXYZ : public Edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeReprojectionXYZ(const Vec3 &pts_i)
        : Edge(2, 2, std::vector<std::string>{"VertexXYZ", "VertexPose"}) {
        obs_ = pts_i;
    }

    /// 返回边的类型信息
    virtual std::string TypeInfo() const override { return "EdgeReprojectionXYZ"; }

    /// 计算残差
    virtual void ComputeResidual() override;

    /// 计算雅可比
    virtual void ComputeJacobians() override;

    void SetTranslationImuFromCamera(Eigen::Quaterniond &qic_, Vec3 &tic_);

private:
    //Translation imu from camera
    Qd qic;
    Vec3 tic;

    //measurements
    Vec3 obs_;
};

/**
 * 仅计算重投影pose的例子
 */
class EdgeReprojectionPoseOnly : public Edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeReprojectionPoseOnly(const Vec3 &landmark_world, const Mat33 &K) :
        Edge(2, 1, std::vector<std::string>{"VertexPose"}),
        landmark_world_(landmark_world), K_(K) {}

    /// 返回边的类型信息
    virtual std::string TypeInfo() const override { return "EdgeReprojectionPoseOnly"; }

    /// 计算残差
    virtual void ComputeResidual() override;

    /// 计算雅可比
    virtual void ComputeJacobians() override;

private:
    Vec3 landmark_world_;
    Mat33 K_;
};

}
}

#endif
