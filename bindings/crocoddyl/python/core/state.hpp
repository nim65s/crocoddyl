///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_PYTHON_CORE_STATE_HPP_
#define CROCODDYL_PYTHON_CORE_STATE_HPP_

#include <boost/python.hpp>
#include <crocoddyl/core/state.hpp>

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

template <bool Type, typename Scalar>
class StateAbstract_py : public StateAbstract<Type, Scalar>,
                         public bp::wrapper<StateAbstract<Type, Scalar>> {
public:
  StateAbstract_py(int nx, int ndx) : StateAbstract<Type, Scalar>(nx, ndx) {}

  VectorType<Type, Scalar> zero() {
    return bp::call<VectorType<Type, Scalar>>(this->get_override("zero").ptr());
  }

  VectorType<Type, Scalar> rand() {
    return bp::call<VectorType<Type, Scalar>>(this->get_override("rand").ptr());
  }

  VectorType<Type, Scalar> diff(const VectorType<Type, Scalar> &x0,
                                const VectorType<Type, Scalar> &x1) {
    return bp::call<VectorType<Type, Scalar>>(this->get_override("diff").ptr(),
                                              x0, x1);
  }

  VectorType<Type, Scalar> integrate(const VectorType<Type, Scalar> &x,
                                     const VectorType<Type, Scalar> &dx) {
    return bp::call<VectorType<Type, Scalar>>(
        this->get_override("integrate").ptr(), x, dx);
  }

  MatrixType<Type, Scalar> Jdiff(const VectorType<Type, Scalar> &x0,
                                 const VectorType<Type, Scalar> &x1,
                                 std::string firstsecond = "both") {
    return bp::call<MatrixType<Type, Scalar>>(this->get_override("Jdiff").ptr(),
                                              x0, x1, firstsecond);
  }

  MatrixType<Type, Scalar> Jintegrate(const VectorType<Type, Scalar> &x,
                                      const VectorType<Type, Scalar> &dx,
                                      std::string firstsecond = "both") {
    return bp::call<MatrixType<Type, Scalar>>(
        this->get_override("Jintegrate").ptr(), x, dx, firstsecond);
  }
};

template <bool Type, typename Scalar> void exposeStateAbstract() {
  bp::class_<StateAbstract_py<Type, Scalar>, boost::noncopyable>(
      "StateAbstract",
      R"(Abstract class for the state representation.

        A state is represented by its operators: difference, integrates and their derivatives.
        The difference operator returns the value of x1 [-] x2 operation. Instead the integrate
        operator returns the value of x [+] dx. These operators are used to compared two points
        on the state manifold M or to advance the state given a tangential velocity (Tx M).
        Therefore the points x, x1 and x2 belongs to the manifold M; and dx or x1 [-] x2 lie
        on its tangential space.)",
      bp::init<int, int>(bp::args(" self", " nx", " ndx"),
                         R"(Initialize the state dimensions.

:param nx: dimension of state configuration vector,
:param ndx: dimension of state tangent vector)"))
      .def("zero", pure_virtual(&StateAbstract_py<Type, Scalar>::zero),
           bp::args(" self"),
           R"(Return a zero reference state.

:return zero reference state)")
      .def("rand", pure_virtual(&StateAbstract_py<Type, Scalar>::rand),
           bp::args(" self"),
           R"(Return a random reference state.

:return random reference state)")
      .def("diff", pure_virtual(&StateAbstract_py<Type, Scalar>::diff),
           bp::args(" self", " x0", " x1"),
           R"(Operator that differentiates the two state points.

It returns the value of x1 [-] x0 operation. Note tha x0 and x1 are points in the state
manifold (in M). Instead the operator result lies in the tangent-space of M.
:param x0: current state
:param x1: next state
:return x1 [-] x0 value)")
      .def("integrate",
           pure_virtual(&StateAbstract_py<Type, Scalar>::integrate),
           bp::args(" self", " x", " dx"),
           R"(Operator that integrates the current state.

It returns the value of x [+] dx operation. x and dx are points in the state manifold (in M)
and its tangent, respectively. Note that the operator result lies on M too.
:param x: current state
:param dx: displacement of the state
:return x [+] dx value)")
      .def("Jdiff", pure_virtual(&StateAbstract_py<Type, Scalar>::Jdiff),
           bp::args(" self", " x0", " x1", " firstsecond = 'both'"),
           R"(Compute the partial derivatives of difference operator.

For a given state, the difference operator (x1 [-] x0) is defined by diff(x0, x1). Instead
here it is described its partial derivatives, i.e. \partial{diff(x0, x1)}{x0} and
\partial{diff(x0, x1)}{x1}. By default, this function returns the derivatives of the
first and second argument (i.e. firstsecond='both'). However we ask for a specific partial
derivative by setting firstsecond='first' or firstsecond='second'.
:param x0: current state
:param x1: next state
:param firstsecond: desired partial derivative
:return the partial derivative(s) of the diff(x0, x1) function)")
      .def("Jintegrate",
           pure_virtual(&StateAbstract_py<Type, Scalar>::Jintegrate),
           bp::args(" self", " x", " dx", " firstsecond = 'both'"),
           R"(Compute the partial derivatives of integrate operator.

For a given state, the integrate operator (x [+] dx) is defined by integrate(x, dx).
Instead here it is described its partial derivatives, i.e. \partial{integrate(x, dx)}{x}
and \partial{integrate(x, dx)}{dx}. By default, this function returns the derivatives of
the first and second argument (i.e. firstsecond='both'). However we ask for a specific
partial derivative by setting firstsecond='first' or firstsecond='second'.
:param x: current state
:param dx: displacement of the state
:param firstsecond: desired partial derivative
:return the partial derivative(s) of the integrate(x, dx) function)")
      .add_property("nx", &StateAbstract_py<Type, Scalar>::nx,
                    "dimension of state configuration vector")
      .add_property("ndx", &StateAbstract_py<Type, Scalar>::ndx,
                    "dimension of state tangent vector")
      .def("printer", &StateAbstract_py<Type, Scalar>::printer);
}

} // namespace python
} // namespace crocoddyl

#endif // CROCODDYL_PYTHON_CORE_STATE_HPP_