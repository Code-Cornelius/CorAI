# **Instructions for virtual environment and requirement files**

All the commands should be run from the root directory of the project.

1. ### Create a virtual environment
```commandline
python3 -m venv name_of_your_environment
```

2. ### Activate the virtual environment
To activate the environment, check the command here: https://docs.python.org/3/
library/venv.html. On unix systems using bash or zsh it is the following command:
```commandline
source name_of_your_environment/bin/activate
```

3. ### Install the requirements

    1. ####Update pip if necessary
         ```commandline
        pip install --upgrade pip
         ```

   2. #####Install the requirements for the module
   
      One can install a library with:
      
       ```commandline
       pip install <library_name> [<version>]
       ```
      
      Or if a requirement file is provided, simply use either, to install all the libraries from the requirement file:
    
         ```commandline
            pip install -r requirements.txt
         ```
       
       or if one simply wants some packages from the requirement file:
       
      ```commandline
      pip install -r <module_name>/requirements.txt
      ```

4. ### Run the tests

```commandline
python3 -m unittest discover <module_name>
```

5. ### Run any modules
```commandline
python3 -m <module_name>
```

6. ### Close the virtual environment
```commandline
deactivate
```

# **Instructions for requirements file**

1. ### Create a requirements file
    From the root of the project the requirements are written in a file requirements.txt. It can be created with:
    1. #### Using freeze
    
        check the installed packages:
    
        ```commandline
        pip freeze
        ```
        And to write them inside the requirements file:
    
        ```commandline
        pip freeze > requirements.txt
        ```
   
    2. #### Manually
        Just add a new file with the name requirements.txt.
        On every line, write the library with the required version:
        
        ```
        <library_name>==<version>
        ```

    The version for each library can be checked on: [https://pypi.org/][pypi].
    One can also use '>=' instead of '==', in order to allow for any version bigger or equal to the one specified. 

[pypi]: https://pypi.org/