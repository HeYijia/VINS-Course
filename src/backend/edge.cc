#include "backend/vertex.h"
#include "backend/edge.h"
//#include <glog/logging.h>
#include <iostream>

using namespace std;

namespace myslam {
namespace backend {

unsigned long global_edge_id = 0;

Edge::Edge(int residual_dimension, int num_verticies,
           const std::vector<std::string> &verticies_types) {
    residual_.resize(residual_dimension, 1);
//    verticies_.resize(num_verticies);      // TODO:: 这里可能会存在问题，比如这里resize了3个空,后续调用edge->addVertex. 使得vertex前面会存在空元素
    if (!verticies_types.empty())
        verticies_types_ = verticies_types;
    jacobians_.resize(num_verticies);
    id_ = global_edge_id++;

    Eigen::MatrixXd information(residual_dimension, residual_dimension);
    information.setIdentity();
    information_ = information;

    lossfunction_ = NULL;
//    cout<<"Edge construct residual_dimension="<<residual_dimension
//            << ", num_verticies="<<num_verticies<<", id_="<<id_<<endl;
}

Edge::~Edge() {}

double Edge::Chi2() const{
    // TODO::  we should not Multiply information here, because we have computed Jacobian = sqrt_info * Jacobian
    return residual_.transpose() * information_ * residual_;
//    return residual_.transpose() * residual_;   // 当计算 residual 的时候已经乘以了 sqrt_info, 这里不要再乘
}

double Edge::RobustChi2() const{

    double e2 = this->Chi2();
    if(lossfunction_)
    {
        Eigen::Vector3d rho;
        lossfunction_->Compute(e2,rho);
        e2 = rho[0];
    }
    return e2;
}
void Edge::RobustInfo(double &drho, MatXX &info) const{
    if(lossfunction_)
    {
//        double e2 = residual_.transpose() * information_ * residual_;
        double e2 = this->Chi2();
        Eigen::Vector3d rho;
        lossfunction_->Compute(e2,rho);
        VecX weight_err = sqrt_information_ * residual_;

        MatXX robust_info(information_.rows(), information_.cols());
        robust_info.setIdentity();
        robust_info *= rho[1];
        if(rho[1] + 2 * rho[2] * e2 > 0.)
        {
            robust_info += 2 * rho[2] * weight_err * weight_err.transpose();
        }

        info = robust_info * information_;
        drho = rho[1];
    }else
    {
        drho = 1.0;
        info = information_;
    }
}

bool Edge::CheckValid() {
    if (!verticies_types_.empty()) {
        // check type info
        for (size_t i = 0; i < verticies_.size(); ++i) {
            if (verticies_types_[i] != verticies_[i]->TypeInfo()) {
                cout << "Vertex type does not match, should be " << verticies_types_[i] <<
                     ", but set to " << verticies_[i]->TypeInfo() << endl;
                return false;
            }
        }
    }
/*
    CHECK_EQ(information_.rows(), information_.cols());
    CHECK_EQ(residual_.rows(), information_.rows());
    CHECK_EQ(residual_.rows(), observation_.rows());

    // check jacobians
    for (size_t i = 0; i < jacobians_.size(); ++i) {
        CHECK_EQ(jacobians_[i].rows(), residual_.rows());
        CHECK_EQ(jacobians_[i].cols(), verticies_[i]->LocalDimension());
    }
    */
    return true;
}

}
}