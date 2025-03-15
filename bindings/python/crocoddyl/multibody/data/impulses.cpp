///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/data/impulses.hpp"

#include "python/crocoddyl/multibody/multibody.hpp"

namespace crocoddyl {
namespace python {

template <typename Data>
struct DataCollectorImpulseVisitor
    : public bp::def_visitor<DataCollectorImpulseVisitor<Data>> {
  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.add_property(
        "impulses",
        bp::make_getter(&Data::impulses,
                        bp::return_value_policy<bp::return_by_value>()),
        "impulses data");
  }
};

#define CROCODDYL_DATA_COLLECTOR_IMPULSE_PYTHON_BINDINGS(Scalar) \
  typedef DataCollectorImpulseTpl<Scalar> Data;                  \
  typedef DataCollectorAbstractTpl<Scalar> DataBase;             \
  typedef ImpulseDataMultipleTpl<Scalar> ImpulseData;            \
  bp::register_ptr_to_python<std::shared_ptr<Data>>();           \
  bp::class_<Data, bp::bases<DataBase>>(                         \
      "DataCollectorImpulse", "Impulse data collector.\n\n",     \
      bp::init<std::shared_ptr<ImpulseData>>(                    \
          bp::args("self", "impulses"),                          \
          "Create impulse data collection.\n\n"                  \
          ":param impulses: impulses data"))                     \
      .def(DataCollectorImpulseVisitor<Data>())                  \
      .def(CopyableVisitor<Data>());

#define CROCODDYL_DATA_COLLECTOR_MULTIBODY_IMPULSE_PYTHON_BINDINGS(Scalar)     \
  typedef DataCollectorMultibodyInImpulseTpl<Scalar> Data;                     \
  typedef DataCollectorMultibodyTpl<Scalar> DataBase1;                         \
  typedef DataCollectorImpulseTpl<Scalar> DataBase2;                           \
  typedef pinocchio::DataTpl<Scalar> PinocchioData;                            \
  typedef ImpulseDataMultipleTpl<Scalar> ImpulseData;                          \
  bp::register_ptr_to_python<std::shared_ptr<Data>>();                         \
  bp::class_<Data, bp::bases<DataBase1, DataBase2>>(                           \
      "DataCollectorMultibodyInImpulse",                                       \
      "Data collector for multibody systems in impulse.\n\n",                  \
      bp::init<PinocchioData*, std::shared_ptr<ImpulseData>>(                  \
          bp::args("self", "pinocchio", "impulses"),                           \
          "Create multibody data collection.\n\n"                              \
          ":param pinocchio: Pinocchio data\n"                                 \
          ":param impulses: impulses data")[bp::with_custodian_and_ward<1,     \
                                                                        2>()]) \
      .def(CopyableVisitor<Data>());

void exposeDataCollectorImpulses() {
  CROCODDYL_PYTHON_SCALARS(CROCODDYL_DATA_COLLECTOR_IMPULSE_PYTHON_BINDINGS)
  CROCODDYL_PYTHON_SCALARS(
      CROCODDYL_DATA_COLLECTOR_MULTIBODY_IMPULSE_PYTHON_BINDINGS)
}

}  // namespace python
}  // namespace crocoddyl
