///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CORE_STATE_HPP_
#define CORE_STATE_HPP_

#include <string>
#include <Eigen/Dense>
#include <Eigen/Sparse>


namespace crocoddyl {

template <bool Type, typename Scalar, int Rows = -1, int Cols = -1>
using MatrixType = typename std::conditional<Type, Eigen::Matrix<Scalar, Rows, Cols>,
                                                   Eigen::SparseMatrix<Scalar>>::type;
template <bool Type, typename Scalar, int Rows = -1>
using VectorType = MatrixType<Type, Scalar, Rows, 1>;

template <bool Type, typename Scalar, int NX = -1, int NDX = -1>
class StateAbstract  {
 public:
    StateAbstract(int nx, int ndx) : nx_(nx), ndx_(ndx) {
        if (NX != -1) {
            assert(NX == nx);
        }
        if (NDX != -1) {
            assert(NDX == ndx);
        }
    }
    virtual ~StateAbstract() {}  // = default;

    virtual VectorType<Type, Scalar, NX> zero() = 0;

    virtual VectorType<Type, Scalar, NX> rand() = 0;

    virtual VectorType<Type, Scalar, NDX>
    diff(const VectorType<Type, Scalar, NX>& x0,
         const VectorType<Type, Scalar, NX>& x1) = 0;

    virtual VectorType<Type, Scalar, NX>
    integrate(const VectorType<Type, Scalar, NX>& x0,
              const VectorType<Type, Scalar, NDX>& dx) = 0;

    virtual MatrixType<Type, Scalar, NDX, NDX>
    Jdiff(const VectorType<Type, Scalar, NX>& x0,
          const VectorType<Type, Scalar, NX>& x1,
          std::string firstsecond = "both") = 0;

    virtual MatrixType<Type, Scalar, NDX, NDX>
    Jintegrate(const VectorType<Type, Scalar, NX>& x,
               const VectorType<Type, Scalar, NDX>& dx,
               std::string firstsecond = "both") = 0;

    int nx() const {
        return nx_;
    }

    int ndx() const {
        return ndx_;
    }

    void printer(const VectorType<Type, Scalar, NX>& x0,
                 const VectorType<Type, Scalar, NX>& x1,
                 const VectorType<Type, Scalar, NDX>& dx) {
        std::cout << "diff" << std::endl;
        std::cout << diff(x0, x1) << std::endl;
        std::cout << "integrate" << std::endl;
        std::cout << integrate(x0, dx) << std::endl;
        std::cout << "Jdiff" << std::endl;
        std::cout << Jdiff(x0, x1) << std::endl;
        std::cout << "Jintegrate" << std::endl;
        std::cout << Jintegrate(x0, dx) << std::endl;
    }

 private:
    int nx_;
    int ndx_;
};

}  // namespace crocoddyl

#endif  // CORE_STATE_HPP_
