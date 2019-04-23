#include <eigenpy/eigenpy.hpp>
// #include <eigenpy/geometry.hpp>
#include <core/state.hpp>
#include <boost/python.hpp>


namespace crocoddyl {

namespace bp = boost::python;

template <class ConfigVectorType, class TangentVectorType>
class StateAbstract_py : public StateAbstract<ConfigVectorType, TangentVectorType>,
                         public bp::wrapper<StateAbstract<ConfigVectorType, TangentVectorType> > {
 public:
    StateAbstract_py(int nx, int ndx) : StateAbstract<ConfigVectorType, TangentVectorType>(nx, ndx) {}

    TangentVectorType diff(const ConfigVectorType& x0, const ConfigVectorType& x1) {
        return this->get_override("diff")(x0, x1);
    }

    TangentVectorType integrate(const ConfigVectorType& x, const TangentVectorType& dx) {
        return this->get_override("integrate")(x, dx);
    }

};


BOOST_PYTHON_MODULE(libcrocoddyl_pywrap) {
    eigenpy::enableEigenPy();
    // eigenpy::exposeAngleAxis();
    // eigenpy::exposeQuaternion();
    eigenpy::enableEigenPySpecific<Eigen::VectorXd>();

    typedef Eigen::VectorXd ConfigVectorType;
    typedef Eigen::VectorXd TangentVectorType;
    bp::class_<StateAbstract_py<ConfigVectorType, TangentVectorType>, boost::noncopyable>("StateAbstract",
                                                                                          bp::init<int, int>())
    .def("diff", pure_virtual(&StateAbstract_py<ConfigVectorType, TangentVectorType>::diff))
    .def("integrate", pure_virtual(&StateAbstract_py<ConfigVectorType, TangentVectorType>::integrate))
    .add_property("nx", &StateAbstract_py<ConfigVectorType, TangentVectorType>::nx)
    .add_property("ndx", &StateAbstract_py<ConfigVectorType, TangentVectorType>::ndx)
    .def("printer",  &StateAbstract_py<ConfigVectorType, TangentVectorType>::printer);
}

}  // namespace crocoddyl
