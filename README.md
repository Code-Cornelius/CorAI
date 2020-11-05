# Python Private Libraries

### Available version

* *version 1.000* :  first trial version. It was released in order for me to have a stable version that was compatible with my summer projects. There is still a lot to do on it. Released on September the 21st 2020.
* *version 1.142* : the newest version, available in December 2020. The library is better documented, and better structured.

### General information

The aim of this repository is to simplify classical python routines. We detail here the different directories available, which depends on the intended usage. 

Some functions are simple classical routines. Other files offer more advanced code, involving wrappers classes, classes objects, metaclasses…

Finally, we are trying to incorporate some C++ routines in the code for very efficient code. This part is still in the project phase.
 
### Naming convention

* All libraries start with the name *"library_{NAME LIBRARY}"*,
* Classes files start with *class_{NAME OF CLASS}.py*,
* Metaclasses files start with *metaclass_{NAME OF CLASS}.py*,
* Custom errors start with *error_{NAME OF ERROR}.py*,
* Custom warning start with *warning_{NAME OF WARNING}.py*,

In library_functions, one can find functions files. Only functions are defined inside. functions file should follow such pattern:  *{NAME SET}_functions.py*. There is one exception. One directory is called "tools" and the functions files inside are called: *classical_functions_{NAME SET}.py*.


### Available version
## library_classes

We created two original objects. 

* **APlot** : A class that simplifies drawing using the library matplotlib ; 
* **Estimator** : A class that intends to make dataframes more accessible. 

## library_errors

Custom errors for better handling of errors in the library. They all inherit from the built-in exception and intends to make the code clearer.

* **Error_convergence** inherits from Exception
* **Error_not_allowed_input** inherits from ValueError
* **Error_not_enough_information** inherits from ValueError
* **Error_not_yet_allowed** inherits from ValueError
* **Error_type_setter** inherits from TypeError
* **Warning_deprecated** inherits from DeprecationWarning

## library_functions

## library_metaclasses

