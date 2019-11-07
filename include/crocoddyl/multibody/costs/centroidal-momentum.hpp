///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_MOMENTUM_HPP_
#define CROCODDYL_MULTIBODY_COSTS_MOMENTUM_HPP_

#include "crocoddyl/multibody/cost-base.hpp"

namespace crocoddyl {

class CostModelCentroidalMomentum : public CostModelAbstract {
 public:
  typedef Eigen::Matrix<double, 6, 1> Vector6d;

  CostModelCentroidalMomentum(boost::shared_ptr<StateMultibody> state,
                              boost::shared_ptr<ActivationModelAbstract> activation, const Vector6d& ref,
                              const std::size_t& nu);
  CostModelCentroidalMomentum(boost::shared_ptr<StateMultibody> state,
                              boost::shared_ptr<ActivationModelAbstract> activation, const Vector6d& ref);
  CostModelCentroidalMomentum(boost::shared_ptr<StateMultibody> state, const Vector6d& ref, const std::size_t& nu);
  CostModelCentroidalMomentum(boost::shared_ptr<StateMultibody> state, const Vector6d& ref);
  ~CostModelCentroidalMomentum();

  void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true);
  boost::shared_ptr<CostDataAbstract> createData(pinocchio::Data* const data);

  const Eigen::VectorXd& get_ref() const;

 private:
  Vector6d ref_;
};

struct CostDataCentroidalMomentum : public CostDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  CostDataCentroidalMomentum(Model* const model, pinocchio::Data* const data)
      : CostDataAbstract(model, data),
        hdot_partial_dq(6, model->get_state()->get_nv()),
        hdot_partial_dv(6, model->get_state()->get_nv()) {
    hdot_partial_dq.fill(0);
    hdot_partial_dv.fill(0);
  }

  pinocchio::Data::Matrix6x hdot_partial_dq;
  pinocchio::Data::Matrix6x hdot_partial_dv;
  pinocchio::Data::Matrix6x Arr_Rx;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_COSTS_MOMENTUM_HPP_
