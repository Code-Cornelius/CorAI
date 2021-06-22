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
todo : the order of inheritance has some importance. It has an impact at ....
The convention is:

Class | Naming Convention | 
------------ | ------------ |
Estimator Child Class                   | Estim_function_name | 
Plot_estimator Child Class              | Plot_estim_function_name | 
Evolution_plot_estimator Child Class    | Plot_evol_function_name | 
Histogram_estimator Child Class         | Plot_hist_function_name | 
Statistic_estimator Child Class         | Plot_stat_function_name | 

Mermaid diagrams: https://mermaid-js.github.io/mermaid/#/
Easily instalable in Pycharm.

We follow UML class diagrams: https://mermaid-js.github.io/mermaid/#/classDiagram
https://en.wikipedia.org/wiki/Class_diagram

'*' means abstract, '$' means static or class method.


### Basic Structure diagram of the plotter. 

Typical diamond form where the left wing (Plot_estim_function_name) indicates the common behavior of plotting 
a certain function, and the right wing (Type_plot_estimator) indicates the way to plot such type of plot.
```mermaid
classDiagram
        
        
        Plot_estimator       <|-- Type_plot_estimator
        Plot_estimator       <|-- Plot_estim_function_name
        Plot_estim_function_name       <|-- Plot_type_function_name
        Type_plot_estimator       <|-- Plot_type_function_name

        
        <<abstract>> Plot_estimator
        <<abstract>> Type_plot_estimator
```


### Global Scheme, three types of plotters represented: evolution, histogram, statistic.

```mermaid
classDiagram
        
        Root_plot_estimator  <|-- Plot_estimator
        
        Plot_estimator       <|-- Evolution_plot_estimator
        Plot_estimator       <|-- Histogram_estimator

        Plot_estimator       <|-- Statistic_estimator
        Plot_estimator       <|-- Plot_estim_function_name

        Plot_estim_function_name       <|-- Plot_hist_function_name
        Plot_estim_function_name       <|-- Plot_evol_function_name
        Plot_estim_function_name       <|-- Plot_stat_function_name

        Evolution_plot_estimator       <|-- Plot_evol_function_name
        Histogram_estimator            <|-- Plot_hist_function_name
        Statistic_estimator            <|-- Plot_stat_function_name
        
        
        <<abstract>> Plot_estimator
        <<abstract>> Evolution_plot_estimator
        <<abstract>> Histogram_estimator
        <<abstract>> Statistic_estimator

        
     
        Plot_estimator:                +from_path_csv()$ 
        Plot_estimator:                +generate_title()$ 
        Plot_estimator:                +is_true_value_unique()$
        Plot_estimator:                +draw()* 

        Plot_estimator:                +Estimator estimator 
        Plot_estimator:                + List~str~ grouping_by
        Plot_estimator:                +AColorset COLORMAP


        
        Evolution_plot_estimator:       +String EVOLUTION_COLUMN
        
        Evolution_plot_estimator:       +get_data2true_evolution()
        Evolution_plot_estimator:       +get_data2group_sliced()$   
        Evolution_plot_estimator:       +get_values_evolution_column()$     
        Evolution_plot_estimator:       +get_evolution_name_extremes()$  
        Evolution_plot_estimator:       +get_data2evolution()* 
        Evolution_plot_estimator:       +get_default_dict_fig()* 
        
        Evolution_plot_estimator:       #_raise_if_separator_is_evolution()    
        Evolution_plot_estimator:       #_plot_min_max()
        Evolution_plot_estimator:       #_plot_true_value()    
        Evolution_plot_estimator:       #_plot_finalisation()
        Evolution_plot_estimator:       +draw()     

        
        
        Histogram_estimator:       +int NB_OF_BINS* 
        Histogram_estimator:       +get_range()$ 
        Histogram_estimator:       +get_dict_plot_param()*
        Histogram_estimator:       +get_dict_fig()*
        Histogram_estimator:       +draw()
        


        
        
        Plot_estim_function_name:                +from_path_csv()$ 
        Plot_estim_function_name:                +generate_title()$ 
        Plot_estim_function_name:                +is_true_value_unique()$
        Plot_estim_function_name:                +draw()* 

        Plot_estim_function_name:                +Estimator estimator 
        Plot_estim_function_name:                + List~str~ grouping_by
        Plot_estim_function_name:                +AColorset COLORMAP


        
        Plot_evol_function_name:       +String EVOLUTION_COLUMN
        
        Plot_evol_function_name:       +get_data2true_evolution()
        Plot_evol_function_name:       +get_data2evolution()* 
        Plot_evol_function_name:       +get_default_dict_fig()* 
        
        Plot_evol_function_name:       +draw()     

        
        
        Plot_hist_function_name:       +int NB_OF_BINS* 
        Plot_hist_function_name:       +get_range()$ 
        Plot_hist_function_name:       +get_dict_plot_param()*
        Plot_hist_function_name:       +get_dict_fig()*
        Plot_hist_function_name:       +draw()
        
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
        
        Estimator:                       +groupby_data()$
        Estimator:                       +groupby()
        Estimator:                       +apply_function_upon_data_store_it()
        Estimator:                       +apply_function_upon_data()
        Estimator:                       +append()



        Plot_estimator:                +from_path_csv()$ 
        Plot_estimator:                +generate_title()$ 
        Plot_estimator:                +is_true_value_unique()$
        Plot_estimator:                +draw()* 

        Plot_estimator:                +Estimator estimator 
        Plot_estimator:                + List~str~ grouping_by
        Plot_estimator:                +AColorset COLORMAP


        

        
  

        
        
        Estim_function_name :                      +Index columns
        Estim_function_name:                       +Set CORE_COL

        

        Plot_estim_function_name:                +from_path_csv()$ 
        Plot_estim_function_name:                +generate_title()$ 
        Plot_estim_function_name:                +is_true_value_unique()$
        Plot_estim_function_name:                +draw()* 

        Plot_estim_function_name:                +Estimator estimator 
        Plot_estim_function_name:                + List~str~ grouping_by
        Plot_estim_function_name:                +AColorset COLORMAP


        


        
        
```





[comment]: <> (Statistic_estimator:       -int sizeInFeet Statistic_estimator:       -canEat&#40;&#41;)

[comment]: <> (Statistic_estimator:       +int age Statistic_estimator:       +String gender Statistic_estimator:       +isMammal&#40;&#41;)

[comment]: <> (Statistic_estimator:       +mate&#40;&#41;       )
