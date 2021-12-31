import unittest
import numpy as np
from example import *
# from testcase.test_util import (
#     print_matrix,
#     init_matrix,
#     print_ndarray,
#     test_equal,
# )


class TestStringMethods(unittest.TestCase):
    def test_layer_construct(self):
        d = Dog()
        print('call_go(d)',call_go(d))
        class Cat(Animal):
            def go(self, n_times):
                return "meow! " * n_times

        c = Cat()
        print('call_go(c)', call_go(c))
        
        animalstore=AnimalStore([
            d,c
        ])
        animalstore.print_animals()
        print('after print animal')
if __name__ == '__main__':
    unittest.main()