#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // needed because some backend gncpy functions retrun stl types
#include <pybind11/eigen.h>
#include <gncpy/filters/IBayesFilter.h>
#include <gncpy/filters/Kalman.h>

#include "Common.h"
#include "../Macros.h"

namespace py = pybind11;

// see https://pybind11.readthedocs.io/en/stable/advanced/classes.html#binding-classes-with-template-parameters for pybind with template class
// see https://stackoverflow.com/questions/62854830/error-by-wrapping-c-abstract-class-with-pybind11 for ImportError: generic_type: type "Derived" referenced unknown base type "Base" error
void initKalman(py::module& m) {

    using namespace lager;

    GNCPY_PY_CHILD_CLASS(gncpy::filters::Kalman, gncpy::filters::IBayesFilter)(m, "Kalman")
        .def(py::init())
        .def("set_state_model", &gncpy::filters::Kalman::setStateModel)
        .def("set_measurement_model", &gncpy::filters::Kalman::setMeasurementModel)
        .def("predict", &gncpy::filters::Kalman::predict)
        // make a small wrapper function so the pass by reference works
        .def("correct", [](gncpy::filters::Kalman& self, double timestep, const Eigen::VectorXd& meas, const Eigen::VectorXd& curState, const gncpy::filters::BayesCorrectParams* params) {
            double measFitProb;
            Eigen::VectorXd nextState = self.correct(timestep, meas, curState, measFitProb, params);
            return py::make_tuple(nextState, measFitProb);
        })
        .def_property("cov", [](gncpy::filters::Kalman& self) {
            return py::EigenDRef<Eigen::MatrixXd>(self.getCov());
        }, [](gncpy::filters::Kalman& self, py::EigenDRef<Eigen::MatrixXd> val){
            self.getCov() = val;
        }, py::return_value_policy::reference_internal)
        GNCPY_PICKLE(gncpy::filters::Kalman);
}