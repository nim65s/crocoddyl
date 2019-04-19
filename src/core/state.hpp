///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CORE_STATE_HPP_
#define CORE_STATE_HPP_


#include <Eigen/Dense>


namespace crocoddyl {

template <class ConfigVectorType, class TangentVectorType>
class StateAbstract {
 public:
    StateAbstract() {}
    ~StateAbstract() {}
    // void diff(
    //     const Eigen::MatrixBase<ConfigVectorType>& x1,
    //     const Eigen::MatrixBase<ConfigVectorType>& x2) {}
    virtual double tmp() = 0;
    void print() {std::cout << tmp() << std::endl;}
};
}  // namespace crocoddyl

#endif  // CORE_STATE_HPP_
