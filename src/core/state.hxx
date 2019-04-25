///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CORE_STATE_HXX_
#define CORE_STATE_HXX_

#include <core/state.hpp>


namespace crocoddyl {

template <bool Type, typename Scalar, int NX, int NDX>
StateAbstract<Type, Scalar, NX, NDX>::StateAbstract(int nx, int ndx) : nx_(nx), ndx_(ndx) {
    if (NX != -1) {
        assert(NX == nx);
    }
    if (NDX != -1) {
        assert(NDX == ndx);
    }
}

template <bool Type, typename Scalar, int NX, int NDX>
StateAbstract<Type, Scalar, NX, NDX>::~StateAbstract() { }

template <bool Type, typename Scalar, int NX, int NDX>
int StateAbstract<Type, Scalar, NX, NDX>::nx() const {
    return nx_;
}

template <bool Type, typename Scalar, int NX, int NDX>
int StateAbstract<Type, Scalar, NX, NDX>::ndx() const {
    return ndx_;
}

template <bool Type, typename Scalar, int NX, int NDX>
void StateAbstract<Type, Scalar, NX, NDX>::printer(const VectorType<Type, Scalar, NX>& x0,
                                                   const VectorType<Type, Scalar, NX>& x1,
                                                   const VectorType<Type, Scalar, NDX>& dx) {
    std::cout << "zero" << std::endl;
    std::cout << zero() << std::endl;
    std::cout << "rand" << std::endl;
    std::cout << rand() << std::endl;
    std::cout << "diff" << std::endl;
    std::cout << diff(x0, x1) << std::endl;
    std::cout << "integrate" << std::endl;
    std::cout << integrate(x0, dx) << std::endl;
    std::cout << "Jdiff" << std::endl;
    std::cout << Jdiff(x0, x1) << std::endl;
    std::cout << "Jintegrate" << std::endl;
    std::cout << Jintegrate(x0, dx) << std::endl;
}

}  // namespace crocoddyl

#endif  // CORE_STATE_HXX_
