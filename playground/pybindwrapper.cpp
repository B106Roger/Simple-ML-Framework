#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <math.h>
#include <mkl.h>
namespace py = pybind11;

class Animal {
public:
    virtual ~Animal() { }
    virtual std::string go(int n_times) {
        return "default animal";
    };
};
class PyAnimal : public Animal {
public:
    /* Inherit the constructors */
    using Animal::Animal;

    /* Trampoline (need one for each virtual function) */
    std::string go(int n_times) override {
        PYBIND11_OVERRIDE_PURE(
            std::string, /* Return type */
            Animal,      /* Parent class */
            go,          /* Name of function in C++ (must match Python name) */
            n_times      /* Argument(s) */
        );
    }
};


/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
class Dog : public Animal {
public:
    std::string go(int n_times) override {
        std::string result;
        for (int i=0; i<n_times; ++i)
            result += "woof! ";
        return result;
    }
};

class AnimalStore {
public:
    AnimalStore(std::vector<Animal*> list): animals(list) {}
    void print_animals() {
        for(size_t i = 0; i < animals.size(); i++) {
            std::cout << animals[i]->go(2) << std::endl;
        }
    }
private:
    std::vector<Animal*> animals;
};

std::string call_go(Animal *animal) {
    return animal->go(3);
}
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////

PYBIND11_MODULE(example, m) {
    py::class_<Animal>(m, "Animal")
        .def(py::init<>())
        .def("go", &Animal::go);

    py::class_<Dog, Animal>(m, "Dog")
        .def(py::init<>());

    py::class_<AnimalStore>(m, "AnimalStore")
        .def(py::init<std::vector<Animal*>>())
        .def("print_animals", &AnimalStore::print_animals);

    m.def("call_go", &call_go);
}