#include <eigenpy/eigenpy.hpp>
#include <eigenpy/geometry.hpp>

#include <core/state.hpp>
#include <boost/python.hpp>



namespace crocoddyl
{
using namespace boost::python;

BOOST_PYTHON_MODULE(libcrocoddyl_pywrap)
{
    // eigenpy::enableEigenPy();
    // eigenpy::exposeAngleAxis();
    // eigenpy::exposeQuaternion();
    // eigenpy::enableEigenPySpecific<Eigen::VectorXd>();


    // typedef Eigen::VectorXd ConfigVectorType;
    // typedef Eigen::VectorXd TangentVectorType;
    typedef double ConfigVectorType;
    typedef double TangentVectorType;
    class_<StateAbstract<ConfigVectorType, TangentVectorType>, boost::noncopyable>("StateAbstract", init<>())
    // .def("diff", pure_virtual(&StateAbstract<ConfigVectorType, TangentVectorType>::diff));
    .def("tmp", pure_virtual(&StateAbstract<ConfigVectorType, TangentVectorType>::tmp))
    .def("printTest", &StateAbstract<ConfigVectorType, TangentVectorType>::print);
}

}