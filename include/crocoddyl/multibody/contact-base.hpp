///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_CONTACT_BASE_HPP_
#define CROCODDYL_MULTIBODY_CONTACT_BASE_HPP_

#include "crocoddyl/multibody/states/multibody.hpp"
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/spatial/force.hpp>

namespace crocoddyl {

struct ContactDataAbstract;  // forward declaration

class ContactModelAbstract {
 public:
  ContactModelAbstract(StateMultibody& state, unsigned int const& nc, unsigned int const& nu);
  ContactModelAbstract(StateMultibody& state, unsigned int const& nc);
  ~ContactModelAbstract();

  virtual void calc(const boost::shared_ptr<ContactDataAbstract>& data,
                    const Eigen::Ref<const Eigen::VectorXd>& x) = 0;
  virtual void calcDiff(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                        const bool& recalc = true) = 0;

  virtual void updateForce(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::VectorXd& force) = 0;
  void updateForceDiff(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::MatrixXd& df_dx,
                       const Eigen::MatrixXd& df_du) const;
  virtual boost::shared_ptr<ContactDataAbstract> createData(pinocchio::Data* const data);

  StateMultibody& get_state() const;
  unsigned int const& get_nc() const;
  unsigned int const& get_nu() const;

 protected:
  StateMultibody& state_;
  unsigned int nc_;
  unsigned int nu_;

#ifdef PYTHON_BINDINGS

 public:
  void calc_wrap(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::VectorXd& x) { calc(data, x); }

  void calcDiff_wrap(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::VectorXd& x,
                     const bool& recalc = true) {
    calcDiff(data, x, recalc);
  }

#endif
};

struct ContactDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  ContactDataAbstract(Model* const model, pinocchio::Data* const data)
      : pinocchio(data),
        joint(0),
        Jc(model->get_nc(), model->get_state().get_nv()),
        a0(model->get_nc()),
        da0_dx(model->get_nc(), model->get_state().get_ndx()),
        df_dx(model->get_nc(), model->get_state().get_ndx()),
        df_du(model->get_nc(), model->get_nu()),
        f(pinocchio::Force::Zero()) {
    Jc.fill(0);
    a0.fill(0);
    da0_dx.fill(0);
    df_dx.fill(0);
    df_du.fill(0);
  }

  pinocchio::Data* pinocchio;
  pinocchio::JointIndex joint;
  Eigen::MatrixXd Jc;
  Eigen::VectorXd a0;
  Eigen::MatrixXd da0_dx;
  Eigen::MatrixXd df_dx;
  Eigen::MatrixXd df_du;
  pinocchio::Force f;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_CONTACT_BASE_HPP_
