#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <math.h>
#include <mkl.h>
#include "matrix.h"
// #include "linear.h"
template<typename Type>
Matrix::Matrix(Type* ptr, size_t nrow, size_t ncol)
    :m_nrow(nrow), m_ncol(ncol), m_buffer(NULL)
{  
    std::cout << 456 << std::endl;
    size_t nelement = nrow * ncol;
    m_buffer = new double[nelement];
    for(size_t i =0; i < nelement; i++) 
    {
        m_buffer[i] = (double)ptr[i];
    }
}
namespace py = pybind11;

PYBIND11_MODULE(_matrix, m) {
    m.doc() = "nsd21au hw3 pybind implementation"; // optional module docstring

    // *********************************************
    // isolation function
    // *********************************************
    // m.def("multiply_naive", &multiply_naive);
    // m.def("multiply_tile", &multiply_tile);
    // m.def("multiply_mkl", &multiply_mkl);
    // m.def("format_descriptor", &test);

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
            // if (info.ndim != 2)
            //     throw std::runtime_error("Incompatible buffer dimension!");
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
        .def("__add__", [](const Matrix &mat1, const Matrix &mat2) {return mat1 + mat2;})
        .def("__eq__", [](const Matrix &mat1, const Matrix &mat2) { return mat1 == mat2; })
        .def("T", &Matrix::T)
        .def_property_readonly("nrow", &Matrix::nrow)
        .def_property_readonly("ncol", &Matrix::ncol)
        .def_property("array", &Matrix::get_array, nullptr);

    // py::class_<Linear>(m, "Linear")
    //     .def(pybind11::init<int,int,bool,bool>()) 
    //         // py::arg("use_bias")=true, 
    //         // py::arg("trainable")=true)
    //     .def("forward", [](Linear &layer, const Matrix &input_tensor) {
    //         // convert (batch, in_feat) to (in_feat, batch)
    //         Matrix raw_input_tensor = input_tensor.T();
    //         Matrix output = layer.forward(raw_input_tensor);
    //         // convert (out_feat, batch) to (out_feat, batch)
    //         output = output.T();
    //         return output;
    //     })
    //     .def("set_weight", &Linear::set_weight);
}