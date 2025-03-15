///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/data/contacts.hpp"

#include "python/crocoddyl/multibody/multibody.hpp"

namespace crocoddyl {
namespace python {

template <typename Data>
struct DataCollectorContactVisitor
    : public bp::def_visitor<DataCollectorContactVisitor<Data>> {
  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.add_property(
        "contacts",
        bp::make_getter(&Data::contacts,
                        bp::return_value_policy<bp::return_by_value>()),
        "contacts data");
  }
};

#define CROCODDYL_DATA_COLLECTOR_CONTACT_PYTHON_BINDINGS(Scalar) \
  typedef DataCollectorContactTpl<Scalar> Data;                  \
  typedef DataCollectorAbstractTpl<Scalar> DataBase;             \
  typedef ContactDataMultipleTpl<Scalar> ContactData;            \
  bp::register_ptr_to_python<std::shared_ptr<Data>>();           \
  bp::class_<Data, bp::bases<DataBase>>(                         \
      "DataCollectorContact", "Contact data collector.\n\n",     \
      bp::init<std::shared_ptr<ContactData>>(                    \
          bp::args("self", "contacts"),                          \
          "Create contact data collection.\n\n"                  \
          ":param contacts: contacts data"))                     \
      .def(DataCollectorContactVisitor<Data>())                  \
      .def(CopyableVisitor<Data>());

#define CROCODDYL_DATA_COLLECTOR_MULTIBODY_CONTACT_PYTHON_BINDINGS(Scalar)     \
  typedef DataCollectorMultibodyInContactTpl<Scalar> Data;                     \
  typedef DataCollectorMultibodyTpl<Scalar> DataBase1;                         \
  typedef DataCollectorContactTpl<Scalar> DataBase2;                           \
  typedef pinocchio::DataTpl<Scalar> PinocchioData;                            \
  typedef ContactDataMultipleTpl<Scalar> ContactData;                          \
  bp::register_ptr_to_python<std::shared_ptr<Data>>();                         \
  bp::class_<Data, bp::bases<DataBase1, DataBase2>>(                           \
      "DataCollectorMultibodyInContact",                                       \
      "Data collector for multibody systems in contact.\n\n",                  \
      bp::init<PinocchioData*, std::shared_ptr<ContactData>>(                  \
          bp::args("self", "pinocchio", "contacts"),                           \
          "Create multibody data collection.\n\n"                              \
          ":param pinocchio: Pinocchio data\n"                                 \
          ":param contacts: contacts data")[bp::with_custodian_and_ward<1,     \
                                                                        2>()]) \
      .def(CopyableVisitor<Data>());

#define CROCODDYL_DATA_COLLECTOR_ACTMULTIBODY_CONTACT_PYTHON_BINDINGS(Scalar)  \
  typedef DataCollectorActMultibodyInContactTpl<Scalar> Data;                  \
  typedef DataCollectorMultibodyInContactTpl<Scalar> DataBase1;                \
  typedef DataCollectorActuationTpl<Scalar> DataBase2;                         \
  typedef pinocchio::DataTpl<Scalar> PinocchioData;                            \
  typedef ActuationDataAbstractTpl<Scalar> ActuationData;                      \
  typedef ContactDataMultipleTpl<Scalar> ContactData;                          \
  bp::register_ptr_to_python<std::shared_ptr<Data>>();                         \
  bp::class_<Data, bp::bases<DataBase1, DataBase2>>(                           \
      "DataCollectorActMultibodyInContact",                                    \
      "Data collector for actuated multibody systems in contact.\n\n",         \
      bp::init<PinocchioData*, std::shared_ptr<ActuationData>,                 \
               std::shared_ptr<ContactData>>(                                  \
          bp::args("self", "pinocchio", "actuation", "contacts"),              \
          "Create multibody data collection.\n\n"                              \
          ":param pinocchio: Pinocchio data\n"                                 \
          ":param actuation: actuation data\n"                                 \
          ":param contacts: contacts data")[bp::with_custodian_and_ward<1,     \
                                                                        2>()]) \
      .def(CopyableVisitor<Data>());

#define CROCODDYL_DATA_COLLECTOR_JOINT_ACTMULTIBODY_CONTACT_PYTHON_BINDINGS(   \
    Scalar)                                                                    \
  typedef DataCollectorJointActMultibodyInContactTpl<Scalar> Data;             \
  typedef DataCollectorMultibodyInContactTpl<Scalar> DataBase1;                \
  typedef DataCollectorActuationTpl<Scalar> DataBase2;                         \
  typedef pinocchio::DataTpl<Scalar> PinocchioData;                            \
  typedef ActuationDataAbstractTpl<Scalar> ActuationData;                      \
  typedef JointDataAbstractTpl<Scalar> JointData;                              \
  typedef ContactDataMultipleTpl<Scalar> ContactData;                          \
  bp::register_ptr_to_python<std::shared_ptr<Data>>();                         \
  bp::class_<Data, bp::bases<DataBase1, DataBase2>>(                           \
      "DataCollectorJointActMultibodyInContact",                               \
      "Data collector for actuated-joint multibody systems in contact.\n\n",   \
      bp::init<PinocchioData*, std::shared_ptr<ActuationData>,                 \
               std::shared_ptr<JointData>, std::shared_ptr<ContactData>>(      \
          bp::args("self", "pinocchio", "actuation", "joint", "contacts"),     \
          "Create multibody data collection.\n\n"                              \
          ":param pinocchio: Pinocchio data\n"                                 \
          ":param actuation: actuation data\n"                                 \
          ":param joint: joint data\n"                                         \
          ":param contacts: contacts data")[bp::with_custodian_and_ward<1,     \
                                                                        2>()]) \
      .def(CopyableVisitor<Data>());

void exposeDataCollectorContacts() {
  CROCODDYL_PYTHON_SCALARS(CROCODDYL_DATA_COLLECTOR_CONTACT_PYTHON_BINDINGS)
  CROCODDYL_PYTHON_SCALARS(
      CROCODDYL_DATA_COLLECTOR_MULTIBODY_CONTACT_PYTHON_BINDINGS)
  CROCODDYL_PYTHON_SCALARS(
      CROCODDYL_DATA_COLLECTOR_ACTMULTIBODY_CONTACT_PYTHON_BINDINGS)
  CROCODDYL_PYTHON_SCALARS(
      CROCODDYL_DATA_COLLECTOR_JOINT_ACTMULTIBODY_CONTACT_PYTHON_BINDINGS)
}

}  // namespace python
}  // namespace crocoddyl
