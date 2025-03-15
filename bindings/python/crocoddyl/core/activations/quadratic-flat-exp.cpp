///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2025, LAAS-CNRS, University of Edinburgh
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/activations/quadratic-flat-exp.hpp"

#include "python/crocoddyl/core/activation-base.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

template <typename Model>
struct ActivationModelQuadFlatExpVisitor
    : public bp::def_visitor<ActivationModelQuadFlatExpVisitor<Model>> {
  typedef typename Model::Scalar Scalar;
  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def("calc", &Model::calc, bp::args("self", "data", "r"),
           "Compute the 1 - exp(-||r||^2 / alpha).\n\n"
           ":param data: activation data\n"
           ":param r: residual vector")
        .def("calcDiff", &Model::calcDiff, bp::args("self", "data", "r"),
             "Compute the derivatives of a quadratic flat function.\n\n"
             "Note that the Hessian is constant, so we don't write again this "
             "value.\n"
             ":param data: activation data\n"
             ":param r: residual vector \n")
        .def("createData", &Model::createData, bp::args("self"),
             "Create the quadratic flat activation data.\n\n")
        .add_property("alpha", bp::make_function(&Model::get_alpha),
                      bp::make_function(&Model::set_alpha), "alpha");
  }
};

#define CROCODDYL_ACTIVATION_MODEL_QUADFLATEXP_PYTHON_BINDINGS(Scalar)    \
  typedef ActivationModelQuadFlatExpTpl<Scalar> Model;                    \
  typedef ActivationModelAbstractTpl<Scalar> ModelBase;                   \
  bp::register_ptr_to_python<std::shared_ptr<Model>>();                   \
  bp::class_<Model, bp::bases<ModelBase>>(                                \
      "ActivationModelQuadFlatExp",                                       \
      "Quadratic flat activation model.\n\n"                              \
      "A quadratic flat action describes a quadratic flat function that " \
      "depends on the residual, i.e., 1 - exp(-||r||^2 / alpha).",        \
      bp::init<std::size_t, Scalar>(                                      \
          bp::args("self", "nr", "alpha"),                                \
          "Initialize the activation model.\n\n"                          \
          ":param nr: dimension of the cost-residual vector"              \
          "param alpha: width of quadratic basin near zero"))             \
      .def(ActivationModelQuadFlatExpVisitor<Model>())                    \
      .def(CastVisitor<Model>())                                          \
      .def(PrintableVisitor<Model>())                                     \
      .def(CopyableVisitor<Model>());

void exposeActivationQuadFlatExp() {
  CROCODDYL_PYTHON_SCALARS(
      CROCODDYL_ACTIVATION_MODEL_QUADFLATEXP_PYTHON_BINDINGS)
}

}  // namespace python
}  // namespace crocoddyl
