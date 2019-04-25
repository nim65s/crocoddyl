///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_CROCODDYL_PYTHON_CORE_CORE_HPP_
#define BINDINGS_CROCODDYL_PYTHON_CORE_CORE_HPP_

#include <crocoddyl/python/core/state.hpp>


namespace crocoddyl {
namespace python {

template <bool Type, typename Scalar>
void exposeCore() {
    exposeStateAbstract<Type, Scalar>();
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_CROCODDYL_PYTHON_CORE_CORE_HPP_
