//
// Created by gaoxiang19 on 11/10/18.
//

#include "backend/loss_function.h"

namespace myslam {
namespace backend {

    void HuberLoss::Compute(double e, Eigen::Vector3d& rho) const {
        double dsqr = delta_ * delta_;
        if (e <= dsqr) { // inlier
            rho[0] = e;
            rho[1] = 1.;
            rho[2] = 0.;
        } else { // outlier
            double sqrte = sqrt(e); // absolut value of the error
            rho[0] = 2*sqrte*delta_ - dsqr; // rho(e)   = 2 * delta * e^(1/2) - delta^2
            rho[1] = delta_ / sqrte;        // rho'(e)  = delta / sqrt(e)
            rho[2] = - 0.5 * rho[1] / e;    // rho''(e) = -1 / (2*e^(3/2)) = -1/2 * (delta/e) / e
        }
    }

    void CauchyLoss::Compute(double err2, Eigen::Vector3d& rho) const {
        double dsqr = delta_ * delta_;       // c^2
        double dsqrReci = 1. / dsqr;         // 1/c^2
        double aux = dsqrReci * err2 + 1.0;  // 1 + e^2/c^2
        rho[0] = dsqr * log(aux);            // c^2 * log( 1 + e^2/c^2 )
        rho[1] = 1. / aux;                   // rho'
        rho[2] = -dsqrReci * std::pow(rho[1], 2); // rho''
    }

    void TukeyLoss::Compute(double e2, Eigen::Vector3d& rho) const
    {
        const double e = sqrt(e2);
        const double delta2 = delta_ * delta_;
        if (e <= delta_) {
            const double aux = e2 / delta2;
            rho[0] = delta2 * (1. - std::pow((1. - aux), 3)) / 3.;
            rho[1] = std::pow((1. - aux), 2);
            rho[2] = -2. * (1. - aux) / delta2;
        } else {
            rho[0] = delta2 / 3.;
            rho[1] = 0;
            rho[2] = 0;
        }
    }
}
}
