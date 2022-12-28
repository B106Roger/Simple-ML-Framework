#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <iostream>
#include <math.h>
#include <mkl.h>
#include "matrix.h"
#include "base_layer.h"
#include "linear.h"
#include "loss.h"
#include "network.h"
#include "optimizer.h"
#include "activation.h"
template<typename Type>
Matrix::Matrix(Type* ptr, size_t nrow, size_t ncol)
    :m_nrow(nrow), m_ncol(ncol), m_buffer(NULL)
{  
    // std::cout << 456 << std::endl;
    size_t nelement = nrow * ncol;
    m_buffer = new double[nelement];
    for(size_t i =0; i < nelement; i++) 
    {
        m_buffer[i] = (double)ptr[i];
    }
}

Matrix layer_forward(Linear &layer, const Matrix &input_tensor) {
    // convert (batch, in_feat)
    Matrix output = layer.forward(input_tensor);
    return output;
}

namespace py = pybind11;

PYBIND11_MODULE(_matrix, m) {
    m.doc() = "nsd21au hw3 pybind implementation"; // optional module docstring

    // *********************************************
    // isolation function
    // *********************************************
    m.def("mat_multiply", &mat_multiply);
    m.def("multiply_naive", &multiply_naive);
    m.def("multiply_mkl", &multiply_mkl);
    m.def("multiply_tile_modify", &multiply_tile_modify);
    // m.def("multiply_tile_modify_thread", &multiply_tile_modify_thread);
    m.def("multiply_tile_modify_pthread", &multiply_tile_modify_pthread);
    m.def("multiply_tile_SIMD_SSE", &multiply_tile_SIMD_SSE);
    m.def("multiply_tile_SIMD_AVX", &multiply_tile_SIMD_AVX);
    // Accelerate Part
    //////////////////////////////////////////////////////////////////////////////////////////
    // Register Your Own Matrix Multiplication Below
    //////////////////////////////////////////////////////////////////////////////////////////

    // Not Correct if the row and col is not 2's multiplier
    // m.def("multiply_tile", &multiply_tile);
    // m.def("multiply_tile_nb", &multiply_tile_nb);
    // m.def("multiply_tile_nb_reorder", &multiply_tile_nb_reorder);

    m.def("format_descriptor", &test);
    m.def("set_matrix_mode", &SetMatrixMode);
    m.def("get_matrix_mode", &GetMatrixMode);


    // *********************************************
    // Class
    // *********************************************
    py::class_<Matrix>(m, "Matrix", py::buffer_protocol())
        // given Matrix instance and convert to numpy object
        .def_buffer([](Matrix &m) -> py::buffer_info{
            return py::buffer_info(
                m.data(),
                sizeof(double),
                py::format_descriptor<double>::format(),
                2,
                {m.nrow(), m.ncol()},
                {sizeof(double) * m.ncol(), sizeof(double)}
            );
        })
        // given numpy array instance and convert to Matrix object
        .def(py::init([](py::buffer b)->Matrix {
            py::buffer_info info = b.request();
            size_t row = info.shape[0], col = info.shape[1];
            
            // Type Determine
            if (info.format == "L") // uint64
                return Matrix((uint64_t*)info.ptr, row, col);
            else if (info.format == "l") // int64
                return Matrix((int64_t*)info.ptr, row, col);
            else if (info.format == "I") // uint32
                return Matrix((uint32_t*)info.ptr, row, col);
            else if (info.format == "i") // int32
                return Matrix((int32_t*)info.ptr, row, col);
            else if (info.format == "H") // uint16
                return Matrix((uint16_t*)info.ptr, row, col);
            else if (info.format == "h") // int16
                return Matrix((int16_t*)info.ptr, row, col);
            else if (info.format == "B") // uint8
                return Matrix((uint8_t*)info.ptr, row, col);
            else if (info.format == "b") // int8
                return Matrix((int8_t*)info.ptr, row, col);
            else if (info.format == "f") // float32
                return Matrix((float*)info.ptr, row, col);
            else if (info.format == "d") // float64
                return Matrix((double*)info.ptr, row, col);
            else if (info.format == "g")  // float128
                return Matrix((long double*)info.ptr, row, col);
            else 
                // throw throw std::runtime_error("Incompatible buffer type!");
                return Matrix();
            return Matrix();
        }))
        .def(pybind11::init<int,int>())
        .def("__setitem__", [](Matrix &mat, std::pair<size_t, size_t> idx, double val) { return mat(idx.first, idx.second) = val; })
        .def("__getitem__", [](const Matrix &mat, std::pair<size_t, size_t> idx) { return mat(idx.first, idx.second); })
        .def("__eq__", [](const Matrix &mat1, const Matrix &mat2) { return mat1 == mat2; })

        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)

        .def(py::self + double())
        .def(py::self - double())
        .def(py::self * double())
        .def(py::self / double())

        .def("__add__", [](const Matrix &mat, int32_t num) {return mat + double(num);}, py::is_operator())
        .def("__sub__", [](const Matrix &mat, int32_t num) {return mat - double(num); }, py::is_operator())
        .def("__mul__", [](const Matrix &mat, int32_t num) {return mat * double(num); }, py::is_operator())
        .def("__truediv__", [](const Matrix &mat, int32_t num) {return mat / double(num); }, py::is_operator())

        .def("__add__", [](const Matrix &mat, int64_t num) {return mat + double(num);}, py::is_operator())
        .def("__sub__", [](const Matrix &mat, int64_t num) {return mat - double(num); }, py::is_operator())
        .def("__mul__", [](const Matrix &mat, int64_t num) {return mat * double(num); }, py::is_operator())
        .def("__truediv__", [](const Matrix &mat, int64_t num) {return mat / double(num); }, py::is_operator())

        .def(double() + py::self)
        .def(double() - py::self)
        .def(double() * py::self)
        .def(double() / py::self)

        .def(py::self += double())
        .def(py::self -= double())
        .def(py::self *= double())
        .def(py::self /= double())

        .def(py::self += py::self)
        .def(py::self -= py::self)
        .def(py::self *= py::self)
        .def(py::self /= py::self)

        .def("T", &Matrix::T)
        .def_property_readonly("nrow", &Matrix::nrow)
        .def_property_readonly("ncol", &Matrix::ncol)
        .def_property("array", &Matrix::get_array, nullptr);

    // *********************************************
    // Layers
    // *********************************************

    py::class_<BaseLayer>(m, "BaseLayer")
        .def(pybind11::init<bool,bool>())
        .def("get_config", &BaseLayer::get_config);

    py::class_<Linear, BaseLayer>(m, "Linear")
        .def(pybind11::init<int,int,bool,bool>()) 
            // py::arg("use_bias")=true, 
            // py::arg("trainable")=true)
        .def("forward", layer_forward)
        .def("__call__", layer_forward)
        .def("set_weight", &Linear::set_weight)
        .def("get_weight", &Linear::get_weight)
        .def_property_readonly("m_weight", &Linear::weight)
        .def_property_readonly("m_bias", &Linear::bias);

    // *********************************************
    // Activations
    // *********************************************

    py::class_<Sigmoid, BaseLayer>(m, "Sigmoid")
        .def(pybind11::init<>())
        .def("__call__", &Sigmoid::forward);

    py::class_<ReLU, BaseLayer>(m, "ReLU")
        .def(pybind11::init<>())
        .def("__call__", &ReLU::forward);

    // *********************************************
    // Model Structure
    // *********************************************

    py::class_<Network>(m, "Network")
        .def(pybind11::init<std::vector<BaseLayer*>>())
        .def("__call__", [](Network &net, const Matrix &mat1) {
            return net.forward(mat1); 
        })
        .def("backward", &Network::backward)
        .def_property("layers", &Network::get_layers, nullptr);

    // *********************************************
    // Losses
    // *********************************************

    py::class_<BaseLoss>(m, "BaseLoss")
        .def(pybind11::init<>())
        .def("__call__", &BaseLoss::operator())
        .def("forward", &BaseLoss::forward)
        .def("backward", &BaseLoss::backward);

    py::class_<MSE, BaseLoss>(m, "MSE")
        .def(pybind11::init<>())
        .def("forward", &MSE::forward)
        .def("backward", &MSE::backward);

    py::class_<CategoricalCrossentropy, BaseLoss>(m, "CategoricalCrossentropy")
        .def(pybind11::init<>())
        .def("forward", &CategoricalCrossentropy::forward)
        .def("backward", &CategoricalCrossentropy::backward);

    // *********************************************
    // Optimizers
    // *********************************************

    py::class_<SGD>(m, "SGD")
        .def(pybind11::init<double,double>())
        .def("apply_gradient", &SGD::apply_gradient);


}