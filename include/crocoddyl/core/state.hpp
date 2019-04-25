///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_STATE_HPP_
#define CROCODDYL_CORE_STATE_HPP_

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <string>

namespace crocoddyl {

template <bool Type, typename Scalar, int Rows = -1, int Cols = -1>
using MatrixType =
    typename std::conditional<Type, Eigen::Matrix<Scalar, Rows, Cols>,
                              Eigen::SparseMatrix<Scalar>>::type;
template <bool Type, typename Scalar, int Rows = -1>
using VectorType = MatrixType<Type, Scalar, Rows, 1>;

template <bool Type, typename Scalar, int NX = -1, int NDX = -1>
class StateAbstract {
public:
  StateAbstract(int nx, int ndx);

  virtual ~StateAbstract();

  virtual VectorType<Type, Scalar, NX> zero() = 0;

  virtual VectorType<Type, Scalar, NX> rand() = 0;

  virtual VectorType<Type, Scalar, NDX>
  diff(const VectorType<Type, Scalar, NX> &x0,
       const VectorType<Type, Scalar, NX> &x1) = 0;

  virtual VectorType<Type, Scalar, NX>
  integrate(const VectorType<Type, Scalar, NX> &x0,
            const VectorType<Type, Scalar, NDX> &dx) = 0;

  virtual MatrixType<Type, Scalar, NDX, NDX>
  Jdiff(const VectorType<Type, Scalar, NX> &x0,
        const VectorType<Type, Scalar, NX> &x1,
        std::string firstsecond = "both") = 0;

  virtual MatrixType<Type, Scalar, NDX, NDX>
  Jintegrate(const VectorType<Type, Scalar, NX> &x,
             const VectorType<Type, Scalar, NDX> &dx,
             std::string firstsecond = "both") = 0;

  int nx() const;

  int ndx() const;

  void printer(const VectorType<Type, Scalar, NX> &x0,
               const VectorType<Type, Scalar, NX> &x1,
               const VectorType<Type, Scalar, NDX> &dx);

private:
  int nx_;
  int ndx_;
};

} // namespace crocoddyl

#include <crocoddyl/core/state.hxx>

#endif // CROCODDYL_CORE_STATE_HPP_
