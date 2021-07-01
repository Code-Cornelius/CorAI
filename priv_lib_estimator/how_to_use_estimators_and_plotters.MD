This is a tutorial on the how to use the classes from this library.

Estimators and Plotters are an efficient way to use dataframes. Dataframes from pandas are simple containers that lack
some global structure and efficient ways to plot data. In order to fill this gap, estimators are wrappers around
dataframes that $ allow storing metadata and grant some useful methods. Also, sometimes dataframe's extremely flexible
structure shall be restricted and one would wish to enforce a given pattern for all dataset of the same type. This is
doable with estimators.

On the other hand, plotters contain estimators and offer an automatic way to plot certain type of representation of the
data.

# ESTIMATORS

# ESTIM. PLOTTERS

The naming convention is:

Class | Naming Convention | 
------------ | ------------ |
Estimator Child Class                   | Estim_function_name | 
Plot_estimator Child Class              | Plot_estim_function_name | 
Relplot_estimator Child Class    | Relplot_function_name | 
Distplot_estimator Child Class         | Distplot_function_name | 

And the order for inheritance should be first the specific plot_estim, then the type of plot. 
That way, if one wants to rely on specific behavior from plot_estim, it is set before reaching the level of relplot.
The usage of `super().` is recommended.

Mermaid diagrams: https://mermaid-js.github.io/mermaid/#/
Easily instalable in Pycharm.

We follow UML class diagrams: https://mermaid-js.github.io/mermaid/#/classDiagram
https://en.wikipedia.org/wiki/Class_diagram

`*` means abstract, `$` means static or class method.


### Basic Structure diagram of the plotter. 

Typical diamond form where the left wing (`Plot_estim_function_name`) indicates the common behavior of plotting 
a certain function, and the right wing (`Typeplot_estimator`) indicates the way to plot such type of plot.
```mermaid
classDiagram
        
        
        Plot_estimator       <|-- Typeplot_estimator
        Plot_estimator       <|-- Plot_estim_function_name
        Plot_estim_function_name       <|-- Typeplot_function_name
        Typeplot_estimator       <|-- Typeplot_function_name

        
        <<abstract>> Plot_estimator
        <<abstract>> Typeplot_estimator
```


### Global Scheme, three types of plotters represented: evolution, histogram, statistic.

```mermaid
classDiagram
        
        Root_plot_estimator  <|-- Plot_estimator
        
        Plot_estimator       <|-- Relplot_estimator
        Plot_estimator       <|-- Distplot_estimator

        Plot_estimator       <|-- Plot_estim_function_name

        Plot_estim_function_name       <|-- Relplot_function_name
        Plot_estim_function_name       <|-- Distplot_function_name

        Relplot_estimator       <|-- Relplot_function_name
        Distplot_estimator            <|-- Distplot_function_name
        
        
        <<abstract>> Plot_estimator
        <<abstract>> Relplot_estimator
        <<abstract>> Distplot_estimator

        
     
        Plot_estimator:                +from_path_csv()$ 
        Plot_estimator:                +generate_title()$ 
        Plot_estimator:                +is_true_value_unique()$
        Plot_estimator:                +draw()* 

        Plot_estimator:                +Estimator estimator 
        Plot_estimator:                + List~str~ grouping_by
        Plot_estimator:                +AColorset COLORMAP
        Plot_estimator:                +is_grouping_by_subset_columns()


        
        Relplot_estimator:       +String EVOLUTION_COLUMN
        
        Relplot_estimator:       +get_data2true_evolution()
        Relplot_estimator:       +get_data2group_sliced()$   
        Relplot_estimator:       +get_values_evolution_column()$     
        Relplot_estimator:       +get_evolution_name_extremes()$  
        Relplot_estimator:       +get_data2evolution()* 
        Relplot_estimator:       +get_dict_fig()* 
        
        Relplot_estimator:       #_raise_if_separator_is_evolution()    
        Relplot_estimator:       +draw()     
        Relplot_estimator:       +lineplot()     
        Relplot_estimator:       +scatter()     

        
        Distplot_estimator:       +get_dict_fig()*
        Distplot_estimator:       +draw()
        Distplot_estimator:       +hist()



        
        
        Plot_estim_function_name:                +from_path_csv()$ 
        Plot_estim_function_name:                +generate_title()$ 
        Plot_estim_function_name:                +is_true_value_unique()$
        Plot_estim_function_name:                +draw()* 

        Plot_estim_function_name:                +Estimator estimator 
        Plot_estim_function_name:                + List~str~ grouping_by
        Plot_estim_function_name:                +AColorset COLORMAP


        
        Relplot_function_name:       +String EVOLUTION_COLUMN
        
        Relplot_function_name:       +get_dict_fig()* 
        Relplot_function_name:       +draw()     

        
        
        Distplot_function_name:       +get_dict_fig()*
        Distplot_function_name:       +hist()
        
```

### Relationship estimator and plotters.
```mermaid
classDiagram
        Estimator              <|-- Estim_function_name
        Plot_estimator         <|-- Plot_estim_function_name

        Plot_estim_function_name  o-- Estim_function_name
        Plot_estimator            o-- Estimator
        
        
        <<abstract>> Plot_estimator
        
     
        Estimator :                      +DataFrame df
        Estimator :                      +Index columns
        Estimator:                       +Set CORE_COL
        
        Estimator:                       +from_json_attributes()$
        Estimator:                       +from_json()
        Estimator:                       +to_json()
        Estimator:                       +to_csv()
        Estimator:                       +from_csv()
        
        Estimator:                       +groupby_data()$
        Estimator:                       +groupby()
        Estimator:                       +apply_function_upon_data_store_it()
        Estimator:                       +apply_function_upon_data()
        Estimator:                       +append()
        Estimator:                       +contains()



        Plot_estimator:                +from_csv()$ 
        Plot_estimator:                +generate_title()$ 
        Plot_estimator:                +is_true_value_unique()$
        Plot_estimator:                +draw()* 

        Plot_estimator:                +Estimator estimator 
        Plot_estimator:                + List~str~ grouping_by
        Plot_estimator:                +AColorset COLORMAP


        

        
        Estim_function_name :                      +Index columns
        Estim_function_name:                       +Set CORE_COL

        

        Plot_estim_function_name:                +generate_title()$ 
        Plot_estim_function_name:                +draw()* 

        Plot_estim_function_name:                +Estimator estimator 
        Plot_estim_function_name:                + List~str~ grouping_by
        Plot_estim_function_name:                +AColorset COLORMAP


        


        
        
```





[comment]: <> (Statistic_estimator:       -int sizeInFeet Statistic_estimator:       -canEat&#40;&#41;)

[comment]: <> (Statistic_estimator:       +int age Statistic_estimator:       +String gender Statistic_estimator:       +isMammal&#40;&#41;)

[comment]: <> (Statistic_estimator:       +mate&#40;&#41;       )
