///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <boost/python.hpp>
#include <crocoddyl/python/core.hpp>
#include <eigenpy/eigenpy.hpp>

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

BOOST_PYTHON_MODULE(libcrocoddyl_pywrap) {
  eigenpy::enableEigenPy();

  const bool Type = true;
  typedef double Scalar;
  eigenpy::enableEigenPySpecific<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>();
  eigenpy::enableEigenPySpecific<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>();

  exposeCore<Type, Scalar>();
}

} // namespace python
} // namespace crocoddyl
