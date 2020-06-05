# Master thesis: Condition Monitoring of Heat Exchangers using Machine Learning
By Herman Wika Horn

This is the code repository for my thesis, which concludes a Master of Science degree at the Norwegian University of Science and Technology (NTNU), as part of the Engineering and ICT study program with a specialization in ICT and Mechanical Engineering. The presented study is a collaboration between Equinor ASA and the Department of Mechanical and Industrial Engineering (MTP), focusing on the application of machine learning methods for the oil and gas domain.

#### Repository intent
This repository hosts the code used through my research, and to obtain the final results. The code may be used by forking or otherwise downloading the code. Additionally, the same functionality is exported as a PyPi package which can be installed using pip

#### Contents
The repository contains:
- Notebook examples (".ipynb" files at the top level)
- Python examples ("py_examples" folder, NB: highly recommend using the notebook examples instead)
- Dataset profiling ("profiling" folder, NB: highly recommend using the "profiling" notebooks first)
- Code documentation ("docs" folder, NB: use [htmlpreview.github.io](https://htmlpreview.github.io/?) to view .html files)
- Source code ("src" folder)
- Installation guide (below)
- Thesis summary (below)

The code documentation can be viewed here: https://htmlpreview.github.io/?https://github.com/hermanwh/master-thesis/blob/master/docs/index.html

The source code is structured as follows:
- core.py: high-level stateful module (relevant)
- core_stateless.py: high-level stateless module (relevant)
- core_configs.py: used for specific datasets, which are not available for the reader (not relevant)
- data folder: runnable .py files for various preprocessing use cases (not relevant)
- ml folder: runnable .py files for various machine learning use cases + trained ML models (not relevant)
- utils: low-level implementation (relevant)

#### Installing required packages
To install the required packages for running this project, do the following:
1. Install Python 3.6 (no need to put in path)  

2. Clone repository  

3. Navigate to repository folder  

4. Update pip  
   python -m pip install --upgrade pip  

5. Install virtualenv  
   python -m pip install --user virtualenv

6. Create venv with Python 3.6  
   python -m virtualenv -p path\to\python\3.6\python.exe venv  
   e.g.  
   python -m virtualenv -p C:\Users\USERNAME\AppData\Local\Programs\Python\Python36\python.exe venv  

7. Activate virtual environment  
   venv\scripts\activate  

8. Install packages using requirements_local.txt  
   pip install -r requirements.txt

#### PyPi package of this project
A Python package with similar functionality is available at the following URL: [https://pypi.org/project/howiml/](https://pypi.org/project/howiml/).

To install and use the PyPi package, do the following:
1. Install Python 3.6 (no need to put in path)  

2. Create a new project folder  

3. Update pip  
   python -m pip install --upgrade pip  

4. Install virtualenv  
   python -m pip install --user virtualenv

5. Create venv with Python 3.6  
   python -m virtualenv -p path\to\python\3.6\python.exe venv  
   e.g.  
   python -m virtualenv -p C:\Users\USERNAME\AppData\Local\Programs\Python\Python36\python.exe venv  

6. Activate virtual environment  
   venv\scripts\activate  

7. Install PyPi package using pip  
   pip install howiml


## Abstract
Heat exchangers are among the key components in oil and gas processing, by facilitating heat transfer between separated fluids. Optimal utilisation of this capability is necessary to ensure high energy efficiency in processing facilities. During operation, heat exchangers experience accumulation of unwanted material on the heat transfer surfaces, which reduces thermal conductivity and hinders fluid flow. Because of this, implementing condition monitoring to estimate heat exchanger performance is vital. Traditional monitoring techniques have proven unreliable in practice. Hence, Equinor is looking to incorporate machine learning methods in the evaluation of heat exchanger performance. Additionally, it is desirable to determine a minimal set of sensors that suffice to monitor heat exchangers for future, potentially unmanned, processing facilities.

This thesis proposes a set of predictive models using a reduced number of input parameters to estimate heat exchanger performance, by calculating the deviation between predicted and measured coolant outlet temperature. A large selection of linear, multilayer perceptron and recurrent neural network regression models are tested for three different processing facilities. This is supported by the implementation of a software module, enabling rapid prototyping of and comparison between machine learning models.

Several discoveries are made which highlight the difficulty of applying machine learning methods to heat exchanger data, most importantly continuous decreases in system performance and lack of known performance measurements. Data obtained through process simulations are used to evaluate the capabilities of the predictive models for cases of known system degradation, for which encouraging results are obtained. When applied for real processing facilities, similar patterns are observed.

Overall, the proposed models are found to estimate heat exchanger performance well. Even so, much work remains before this can be used for maintenance planning in practice. Numerous topics for further development are identified, most importantly the inclusion of the proposed predictive models in a robust preventive maintenance scheme.

## Problem statement
Heat exchanger fouling has proven difficult to estimate using traditional monitoring techniques. Fouling estimation is vital in ensuring maintenance can be performed in a preventive rather than corrective manner. For new facilities in particular, it is desirable to implement condition monitoring of heat exchangers using a limited set of sensors. Based on the success of machine learning in order fields, its capabilities for the oil and gas domain should be investigated. More specifically, condition monitoring of heat exchangers should be implemented using appropriate machine learning methods, to enable the use of preventive maintenance schemes.

## Objectives
1. Determine the applicability, and potential limitations, of machine learning methods for the oil and gas domain
2. Develop and implement machine learning methods capable of estimating the fouling factor in heat exchangers, to assist with preventive maintenance decision making
3. Determine necessary measuring equipment to accurately estimate heat exchanger fouling for future, potentially unmanned, processing facilities.
4. Facilitate further research within these topics in Equinor

## Methodology
Because the fouling level for real processing facilities cannot be measured directly, system conditions are traditionally derived used other estimates obtained through thermodynamic models. This may require extensive monitoring and several assumptions regarding the fluid properties. Even when such calculations can be performed, the results are only estimates, from which it can be challenging to determine the corresponding fouling level. This makes fouling data from such methods difficult to use for the evaluation of machine learning algorithms.

To facilitate evaluation of the proposed fouling indicators for a controlled and deterministic fouling environment, the use of simulated data for model benchmarking is suggested. During system simulations using appropriate process modeling software, the fouling factor of a heat exchanger module can be set explicitly. Because fouling factor in itself is immeasurable in practice, using simulated data is the only way of comparing predictive results with factual data. Fouling indicators can be reevaluated and applied for real facilities based on their ability to predict the added level of fouling for simulated datasets. For this reason, datasets obtained both through simulations and gathered from real facilities are used throughout this thesis.

Equations are derived that justify the prediction of coolant outlet temperature with the use of coolant inlet temperture, process inlet temperture, process outlet temperture, and process flow rate. Predictive models are defined as follows:
![Predictive models A-E](https://github.com/hermanwh/master-thesis/blob/master/figures/predictive_models.PNG?raw=true)

The predictive models are based on the following hypotheses:
![Hypotheses](https://github.com/hermanwh/master-thesis/blob/master/figures/hypotheses.PNG?raw=true)

Performance is known for simulated dataset D, while for real facilities it can be roughly estimated based on inspection of heat exchanger data and knowledge of maintenance dates. Separate testing sets are not used to validate the trained models. Model performance is primarily evaluated based on empirical interpretation of predictions performed on a portion of data not used for training or validation, in addition to calculated metrics like validation loss and coefficient of determination. Using empirical analysis to evaluate model performance has obvious drawbacks, as it does not generalize well for arbitrary processing facilities and requires extensive domain knowledge. However, with the shortcomings of simple metrics and the otherwise large scope of this thesis, it is considered unproducable to implement more advanced evaluation metrics.

## Implementation
Implementing, analyzing and visualizing the performance of machine learning algorithms requires programming for tasks like preprocessing, building and training models, plotting, printing and more. During review of related work, it was noted that research projects focusing on machine learning often implement code as standalone notebooks or scripts, containing explicit code for these tasks in each document. While this approach provides a simple environment for prototyping, it makes large scale testing and comparison of different model configurations cumbersome by introducing potential boilerplate code. This suggests that modularization may be used to extract code for repeatedly used functionality, and encouraged the implementation of a top-level module. Because machine learning programming requires the use of many underlying frameworks, implementing a top-level module for commonly used tasks increases readability and usability for unfamiliar users by hiding the framework-specific implementation details.

The implemented solution intends to simplify and thus encourage continued research on the use of machine learning monitoring techniques. Despite this, generating interpretable and reliable results for the thesis itself has had top priority throughout. This means modular and reusable code comes as an addition to, and not at the expense of, the paramount thesis objectives.

Use of both the high-level module and additional functionality is demonstrated through a series of Jupyter Notebook files. These are located at the top-level of the GitHub repository. All results presented in this thesis are available in these notebooks, which may either be read directly from GitHub or run locally by forking the GitHub repository, installing the required packages and running Jupyter Notebook.

The reader is referred to the notebooks for the practical demonstration. Data profiling is performed for each facility. Unsupervised methods include PCA and correlation analysis. Supervised methods include applying the predictive models A-E to each facility, as well as extensive comparison of network architectures, regularization parameters and loss metrics. Experimental use cases include performing predictions with models trained on different datasets, as well as model uncertainty assessment.

## Results
Predictive models A and C are found to accurately estimate the level of fouling. Predictive models B, D and E are found to experience problems with overfitting for the coolant valve opening parameter.

### Linear model, dataset D
![Prediction linear model, dataset D](https://github.com/hermanwh/master-thesis/blob/master/figures/pred_1.png?raw=true)
![Deviation linear model, dataset D](https://github.com/hermanwh/master-thesis/blob/master/figures/dev_1.png?raw=true)

### MLP model, dataset D
![Prediction MLP model, dataset D](https://github.com/hermanwh/master-thesis/blob/master/figures/pred_2.png?raw=true)
![Deviation MLP model, dataset D](https://github.com/hermanwh/master-thesis/blob/master/figures/dev_2.png?raw=true)

### LSTM model, dataset G
![Prediction LSTM model, dataset G](https://github.com/hermanwh/master-thesis/blob/master/figures/pred_lstm_a.png?raw=true)
![Deviation LSTM, dataset G](https://github.com/hermanwh/master-thesis/blob/master/figures/dev_lstm_a.png?raw=true)

## Conclusions
Through the definition of a minimal predictive model A and subsequent fitting and evaluation of multiple regression models, the first part of hypothesis 1 concerning fouling estimation is considered credible. Results are presented for three different facilities, all for which considerable increases in thermal performance are identified following heat exchanger maintenance. Continuous performance degradation between each maintenance is also observed. An apparent linear relationship between added fouling factor and deviation in coolant outlet temperature is found for simulated data. 

Furthermore, predictive models B through E were defined to investigate the effects of using additional sensors. Increased predictive performance is found by the inclusion of the coolant flow rate parameter. Meanwhile, clear overfitting is observed when coolant valve opening is included for facility G. Coolant inlet pressure is found to have limited variation, and thus system influence, in practice. The second part of hypothesis 1 concerning addition sensors is also considered credible.

Necessary assumptions are that coolant flow rate of the heat exchanger in question is controlled according to a desired process outlet temperature, and that operating conditions such as fluid compositions do not change substantially. The fitting of regression models require data to be acquired with minimal levels of fouling.

Challenges were faced when defining the hydraulic effects of fouling for simulated data. As a result, no differences in pressure drop are observed for facility D with fouling on either side of the heat exchanger. Consequently, limited emphasis is placed on testing of this hypothesis. For facility G, no clear fouling patterns are discovered using predictive models with process pressure difference as output parameter. Still, hypothesis 2 is considered plausible based on heat exchanger theory. The measurement of pressure differences may be important in practice to account for cases of fouling not reflected in heat exchanger thermal performance.

Recommendations for future work are covered quite extensively to facilitate and encourage further work within the same Equinor research project, and to tie up any loose ends in the wide range of subjects discussed throughout the thesis. During development of the thesis methodology, there has been numerous ideas for models, estimators and schemes that could not be tested due to various constraints, primary the already large extent of the thesis.
