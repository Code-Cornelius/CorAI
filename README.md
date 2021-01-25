# Python Personal Libraries

### Available version

* *version 1.000* :  Released in September 2020. It was released in order for me to have a stable version that was compatible with my summer projects. There is still a lot to do on it. 
* *version 1.142* : Released in January 2021. The library is better documented, and better structured. It is now separated in the corresponding libraries, and there are tests for most of the functions.


### General information

The aim of this repository is to automatise and optimise classical python routines. We detail here the different directories available, which depends on the intended usage. 

Some functions are simple classical routines. Other files offer more advanced code, involving wrappers classes, classes objects, metaclasses…

Finally, we are trying to incorporate some C++ routines in the code for very efficient code. This part is still in the project phase.
 
### Naming convention and how to import


```
Project
├── priv_lib_error 
│  ├── src
│  │  ├── error_convergence
│  │  ├── error_not_allowed_input
│  │  ├── error_not_enough_information  
│  │  ├── error_not_yet_allowed
│  │  ├── error_type_setter  
│  │  ├── numpy_function_used  (not really interesting for general purpose)
│  │  └── warning_deprecated
│  └── tests
│
├── priv_lib_estimator 
│  ├── src
│  │  ├── estimator
│  │  └── plot_estimator
│  └── tests
│
├── priv_lib_metaclass 
│  ├── src
│  │  └── register
│  └── tests
│
├── priv_lib_plot 
│  ├── src
│  └── tests
│
└── priv_lib_util 
   ├── src
   │  ├── calculus
   │  ├── finance
   │  ├── ML  
   │  └── tools
   └── tests
```

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
