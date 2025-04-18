///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/costs/cost-sum.hpp"

#include <functional>
#include <map>
#include <memory>

#include "python/crocoddyl/core/action-base.hpp"
#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/diff-action-base.hpp"
#include "python/crocoddyl/utils/copyable.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"
#include "python/crocoddyl/utils/map-converter.hpp"
#include "python/crocoddyl/utils/printable.hpp"

namespace crocoddyl {
namespace python {

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(CostModelSum_addCost_wrap,
                                       CostModelSum::addCost, 3, 4)

void exposeCostSum() {
  // Register custom converters between std::map and Python dict
  typedef std::shared_ptr<CostItem> CostItemPtr;
  typedef std::shared_ptr<CostDataAbstract> CostDataPtr;
  StdMapPythonVisitor<std::string, CostItemPtr, std::less<std::string>,
                      std::allocator<std::pair<const std::string, CostItemPtr>>,
                      true>::expose("StdMap_CostItem");
  StdMapPythonVisitor<std::string, CostDataPtr, std::less<std::string>,
                      std::allocator<std::pair<const std::string, CostDataPtr>>,
                      true>::expose("StdMap_CostData");

  bp::register_ptr_to_python<std::shared_ptr<CostItem>>();

  bp::class_<CostItem>(
      "CostItem", "Describe a cost item.\n\n",
      bp::init<std::string, std::shared_ptr<CostModelAbstract>, double,
               bp::optional<bool>>(
          bp::args("self", "name", "cost", "weight", "active"),
          "Initialize the cost item.\n\n"
          ":param name: cost name\n"
          ":param cost: cost model\n"
          ":param weight: cost weight\n"
          ":param active: True if the cost is activated (default true)"))
      .def_readwrite("name", &CostItem::name, "cost name")
      .add_property(
          "cost",
          bp::make_getter(&CostItem::cost,
                          bp::return_value_policy<bp::return_by_value>()),
          "cost model")
      .def_readwrite("weight", &CostItem::weight, "cost weight")
      .def_readwrite("active", &CostItem::active, "cost status")
      .def(CopyableVisitor<CostItem>())
      .def(PrintableVisitor<CostItem>());

  bp::register_ptr_to_python<std::shared_ptr<CostModelSum>>();

  bp::class_<CostModelSum>(
      "CostModelSum", bp::init<std::shared_ptr<StateAbstract>, std::size_t>(
                          bp::args("self", "state", "nu"),
                          "Initialize the total cost model.\n\n"
                          ":param state: state description\n"
                          ":param nu: dimension of control vector"))
      .def(bp::init<std::shared_ptr<StateAbstract>, std::size_t>(
          bp::args("self", "state", "nu"),
          "Initialize the total cost model.\n\n"
          "For this case the default nu is equals to model.nv.\n"
          ":param state: state description\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<std::shared_ptr<StateAbstract>>(
          bp::args("self", "state"),
          "Initialize the total cost model.\n\n"
          "For this case the default nu is equals to model.nv.\n"
          ":param state: state description"))
      .def("addCost", &CostModelSum::addCost,
           CostModelSum_addCost_wrap(
               bp::args("self", "name", "cost", "weight", "active"),
               "Add a cost item.\n\n"
               ":param name: cost name\n"
               ":param cost: cost model\n"
               ":param weight: cost weight\n"
               ":param active: True if the cost is activated (default true)"))
      .def("removeCost", &CostModelSum::removeCost, bp::args("self", "name"),
           "Remove a cost item.\n\n"
           ":param name: cost name")
      .def(
          "changeCostStatus", &CostModelSum::changeCostStatus,
          bp::args("self", "name", "active"),
          "Change the cost status.\n\n"
          ":param name: cost name\n"
          ":param active: cost status (true for active and false for inactive)")
      .def<void (CostModelSum::*)(const std::shared_ptr<CostDataSum>&,
                                  const Eigen::Ref<const Eigen::VectorXd>&,
                                  const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &CostModelSum::calc, bp::args("self", "data", "x", "u"),
          "Compute the total cost.\n\n"
          ":param data: cost-sum data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (CostModelSum::*)(const std::shared_ptr<CostDataSum>&,
                                  const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &CostModelSum::calc, bp::args("self", "data", "x"),
          "Compute the total cost value for nodes that depends only on the "
          "state.\n\n"
          "It updates the total cost based on the state only.\n"
          "This function is used in the terminal nodes of an optimal control "
          "problem.\n"
          ":param data: cost-sum data\n"
          ":param x: state point (dim. state.nx)")
      .def<void (CostModelSum::*)(const std::shared_ptr<CostDataSum>&,
                                  const Eigen::Ref<const Eigen::VectorXd>&,
                                  const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelSum::calcDiff,
          bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the total cost.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (CostModelSum::*)(const std::shared_ptr<CostDataSum>&,
                                  const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelSum::calcDiff, bp::args("self", "data", "x"),
          "Compute the Jacobian and Hessian of the total cost for nodes that "
          "depends on the state only.\n\n"
          "It updates the Jacobian and Hessian of the total cost based on the "
          "state only.\n"
          "This function is used in the terminal nodes of an optimal control "
          "problem.\n"
          ":param data: cost-sum data\n"
          ":param x: state point (dim. state.nx)")
      .def("createData", &CostModelSum::createData,
           bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the total cost data.\n\n"
           ":param data: shared data\n"
           ":return total cost data.")
      .add_property(
          "state",
          bp::make_function(&CostModelSum::get_state,
                            bp::return_value_policy<bp::return_by_value>()),
          "state description")
      .add_property(
          "costs",
          bp::make_function(&CostModelSum::get_costs,
                            bp::return_value_policy<bp::return_by_value>()),
          "stack of costs")
      .add_property("nu", bp::make_function(&CostModelSum::get_nu),
                    "dimension of control vector")
      .add_property("nr", bp::make_function(&CostModelSum::get_nr),
                    "dimension of the residual vector of active cost")
      .add_property("nr_total", bp::make_function(&CostModelSum::get_nr_total),
                    "dimension of the total residual vector")
      .add_property(
          "active",
          bp::make_function(
              &CostModelSum::get_active,
              deprecated<bp::return_value_policy<bp::return_by_value>>(
                  "Deprecated. Use property active_set")),
          "list of names of active contact items")
      .add_property(
          "inactive",
          bp::make_function(
              &CostModelSum::get_inactive,
              deprecated<bp::return_value_policy<bp::return_by_value>>(
                  "Deprecated. Use property inactive_set")),
          "list of names of inactive contact items")
      .add_property(
          "active_set",
          bp::make_function(&CostModelSum::get_active_set,
                            bp::return_value_policy<bp::return_by_value>()),
          "name of the active set of cost items")
      .add_property(
          "inactive_set",
          bp::make_function(&CostModelSum::get_inactive_set,
                            bp::return_value_policy<bp::return_by_value>()),
          "name of the inactive set of cost items")
      .def("getCostStatus", &CostModelSum::getCostStatus,
           bp::args("self", "name"),
           "Return the cost status of a given cost name.\n\n"
           ":param name: cost name")
      .def(CopyableVisitor<CostModelSum>())
      .def(PrintableVisitor<CostModelSum>());

  bp::register_ptr_to_python<std::shared_ptr<CostDataSum>>();

  bp::class_<CostDataSum>(
      "CostDataSum", "Class for total cost data.\n\n",
      bp::init<CostModelSum*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create total cost data.\n\n"
          ":param model: total cost model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 3>()])
      .def(
          "shareMemory",
          &CostDataSum::shareMemory<DifferentialActionDataAbstract>,
          bp::args("self", "model"),
          "Share memory with a given differential action data\n\n"
          ":param model: differential action data that we want to share memory")
      .def("shareMemory", &CostDataSum::shareMemory<ActionDataAbstract>,
           bp::args("self", "model"),
           "Share memory with a given action data\n\n"
           ":param model: action data that we want to share memory")
      .add_property(
          "costs",
          bp::make_getter(&CostDataSum::costs,
                          bp::return_value_policy<bp::return_by_value>()),
          "stack of costs data")
      .add_property("shared",
                    bp::make_getter(&CostDataSum::shared,
                                    bp::return_internal_reference<>()),
                    "shared data")
      .add_property(
          "cost",
          bp::make_getter(&CostDataSum::cost,
                          bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&CostDataSum::cost), "cost value")
      .add_property(
          "Lx",
          bp::make_function(&CostDataSum::get_Lx,
                            bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&CostDataSum::set_Lx), "Jacobian of the cost")
      .add_property(
          "Lu",
          bp::make_function(&CostDataSum::get_Lu,
                            bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&CostDataSum::set_Lu), "Jacobian of the cost")
      .add_property(
          "Lxx",
          bp::make_function(&CostDataSum::get_Lxx,
                            bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&CostDataSum::set_Lxx), "Hessian of the cost")
      .add_property(
          "Lxu",
          bp::make_function(&CostDataSum::get_Lxu,
                            bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&CostDataSum::set_Lxu), "Hessian of the cost")
      .add_property(
          "Luu",
          bp::make_function(&CostDataSum::get_Luu,
                            bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&CostDataSum::set_Luu), "Hessian of the cost")
      .def(CopyableVisitor<CostDataSum>());
}

}  // namespace python
}  // namespace crocoddyl
