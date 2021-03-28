# **Instructions for virtual environment**

All the commands should be run from the root directory of the project.

1. ### Create a virtual environment
```commandline
python3 -m venv venv
```

2. ### Activate the virtual environment
```commandline
source venv/bin/activate
```

3. ### Install the requirements

    1. ####Update pip if necessary
         ```commandline
        pip install --upgrade pip
         ```

   2. #####Install the requirements for the module
   
      Substitute **<module_name>** with the selected library.
   
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
    From the root of the project the requirements are needed for:
    1. #### Using freeze
        ```commandline
        pip freeze > requiremets.txt
        ```
    2. #### Manually
        Just add a new file with the name requirements.txt.
       
2. ### Adding requirements
Add a line in the requirements file with the format:
```
<library_name>==<version>
```
The version can be checked on: [https://pypi.org/][pypi].

Can also use '>=' instead of '=='. 

[pypi]: https://pypi.org/