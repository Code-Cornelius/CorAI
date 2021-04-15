# Python Personal Libraries

### Available version

* *version 1.000* :  Released in September 2020. It was released in order for me to have a stable version that was compatible with my summer projects. There is still a lot to do on it. 
* *version 1.142* : Released in June 2021. The library is better documented, and better structured. It is now separated in the corresponding libraries, and there are tests for most of the functions.
* current work : incorporate new financial functions as well as neural networks functions.

One should download the latest version and add the path to it before running code.

### General information

The aim of this repository is to automatise and optimise classical python routines. We detail here the different directories available, which depends on the intended usage. 

Some functions are simple classical routines. Other files offer more advanced code, involving wrappers classes, classes objects, metaclasses…

Finally, we are trying to incorporate some C++ routines in the code for very efficient code. This part is still in the project phase.
 
### Structure of the Project and how to import

The main structure is the following:
```
Project
├── priv_lib_error 
│  ├── src
│  │  ├── error_convergence.py
│  │  ├── error_not_allowed_input.py
│  │  ├── error_not_enough_information.py
│  │  ├── error_not_yet_allowed.py
│  │  ├── error_type_setter.py
│  │  ├── numpy_function_used.py  (not really interesting for general purpose)
│  │  └── warning_deprecated.py
│  └── tests
│
├── priv_lib_estimator 
│  ├── src
│  │  ├── estimator
│  │  │  └── estimator.py
│  │  └── plot_estimator
│  │     ├── plot_estimator.py
│  │     ├── histogram_estimator.py
│  │     ├── statistic_plot_estimator.py
│  │     └── evolution_plot_estimator.py
│  └── tests
│
├── priv_lib_metaclass 
│  ├── src
│  │  └── register
│  │     ├── deco_register.py
│  │     └── register.py
│  └── tests
│
├── priv_lib_plot 
│  ├── src
│  │  ├── acolor
│  │  │  ├── acolorsetdiscrete.py
│  │  │  └── acolorsetcontinuous.py
│  │  └── aplot
│  │     ├── aplot.py
│  │     └── dict_ax_for_aplot.py
│  └── tests
│
└── priv_lib_util 
   ├── calculus
   │  ├── src
   │  │  ├── diff_eq.py
   │  │  ├── integration.py
   │  │  └── optimization.py
   │  └── tests
   ├── finance
   │  ├── src
   │  │  └── financial.py
   │  └── tests
   ├── ML  
   │  ├── src
   │  │  └── networkx.py
   │  └── tests
   └── tools
      ├── src
      │  ├── benchmarking.py
      │  ├── decorator.py
      │  ├── function_dict.py
      │  ├── function_iterable.py
      │  ├── function_recurrent.py
      │  ├── function_str.py
      │  ├── function_writer.py
      │  └── operator.py
      └── tests
```

However, one can import the meaningful objects in the following way, where one `from the_path import the_object`:


 Error_convergence
from .error_not_allowed_input import Error_not_allowed_input
from .error_not_enough_information import Error_not_enough_information
from .error_not_yet_allowed import Error_not_yet_allowed
from .error_type_setter import Error_type_setter
from .warning_deprecated import deprecated_function
from .numpy_function_used import numpy_function_used

```
Project
├── priv_lib_error 
│  ├── Error_convergence
│  ├── Error_not_allowed_input.py
│  ├── Error_not_enough_information.py
│  ├── Error_not_yet_allowed.py
│  ├── Error_type_setter.py
│  └── deprecated_function.py
│
├── priv_lib_estimator 
│  ├── Estimator
│  ├── Plot_estimator
│  ├── Histogram_estimator
│  ├── Statistic_plot_estimator
│  └── Evolution_plot_estimator
│
├── priv_lib_metaclass 
│  ├── deco_register
│  └── Register
│
├── priv_lib_plot 
│  ├── APlot
│  ├── AColorsetDiscrete
│  └── AColorsetContinuous
│
└── priv_lib_util 
   ├── calculus
   │  ├── diff_eq.py
   │  ├── integration.py
   │  └── optimization.py
   ├── finance
   │  └── financial.py
   ├── ML  
   │  └── networkx.py
   └── tools
      ├── benchmarking.py
      ├── decorator.py
      ├── function_dict.py
      ├── function_iterable.py
      ├── function_recurrent.py
      ├── function_str.py
      ├── function_writer.py
      └── operator.py
```

For example, in order to import `benchmarking.py`, one should write:  `from priv_lib_util.tools import benchmarking`

* All libraries start with the name *"priv_lib_{NAME LIBRARY}"*,
inside each library,  there is a source folder and a tests folder. In order to import any module, one should simply write:

```
from priv_lib import module
or
from priv_lib.extension import module
```

Then, the functions written in the module are callable with:

```
module.function()
```

if one wants to simply use the name of the function without refering to the private call table of the library, one can write:

```
function = module.function

function()
```



### Available version
## library_errors

Custom errors for better handling of errors in the library. They all inherit from the built-in exception and intends to make the code clearer.

* **Error_convergence** inherits from Exception,
* **Error_not_allowed_input** inherits from ValueError,
* **Error_not_enough_information** inherits from ValueError,
* **Error_not_yet_allowed** inherits from ValueError,
* **Error_type_setter** inherits from TypeError,
* **Warning_deprecated** function that rise a deprecation warning.

## library_estimator

## library_metaclass

## library_plot

## library_util



||||||||||||||||||||||||||||||||||||||||||||
We created two original objects. 

* **APlot** : A class that simplifies drawing using the library matplotlib ; 
* **Estimator** : A class that intends to make dataframes more accessible. 
