# water Quality Classification app

## Table of Content
* [1. Project Overview](#project-description)
* [2. Dataset](#dataset)
* [3. Packages](#packages)
* [4. Environment](#environment)
* [5. MLFlow](#mlflow)
* [6. Team Members](#team-members)

## 1. Project Overview <a class="anchor" id="project-description"></a>

This project aims to classify water quality into four categories based on parameters such as pH, turbidity, dissolved oxygen, conductivity, ambient temperature ect. while also considering variations across different locations, months, and hours of the day. Using a machine learning model, the system will predict water quality based on input data and provide users with an easy-to-use Streamlit-based web application for real-time analysis.

## 2. Dataset <a class="anchor" id="dataset"></a>

The dataset is comprised of river water quality data.
 * [river_water_parameters.csv contains sampling point, Date  and Time data was collected from, water parameters namely: Ambient temperature (°C),	Ambient humidity, Sample temperature (°C), pH, EC\n(µS/cm), TDS\n(mg/L), TSS\n(mL sed/L), DO\n(mg/L), Level (cm), Turbidity (NTU), Hardness\n(mg CaCO3/L), Hardness classification, Total Cl-\n(mg Cl-/L), an average rating based on views, and the number of members in the anime 'group'.] 
 You can find the `river_water_parameters.csv` datasets [here](https://www.google.com/url?q=https://www.kaggle.com/datasets/natanaelferran/river-water-parameters?select%3DRiver%2Bwater%2Bparameters.csv&sa=D&source=editors&ust=1744381141155687&usg=AOvVaw3e6sLkrseKwC-I2ibjPwN-).

## 3. Packages <a class="anchor" id="packages"></a>
|Packages                                                  | Uses
|----------------------------------------------------------|---------------------------------
|pandas                                                    | Data manipulation and analysis
|numpy                                                     | Numerical operations
|matplotlib                                                | Basic plotting
|seaborn                                                   | Statistical data visualization
|scikit-learn                                              | ML models, metrics, preprocessing
|streamlit                                                 | Web app interface


## 4. Environment <a class="anchor" id="environment"></a>

### Create the new evironment - you only need to do this once

```bash
# create the conda environment
conda create --name <env>
```

### This is how you activate the virtual environment in a terminal and install the project dependencies

```bash
# activate the virtual environment
conda activate <env>
# install the pip package
conda install pip
# install the requirements for this project
pip install -r requirements.txt
```

## 5. MLFlow <a class="anchor" id="mlflow"></a>

MLOps, which stands for Machine Learning Operations, is a practice focused on managing and streamlining the lifecycle of machine learning models. The modern MLOps tool, MLflow is designed to facilitate collaboration on data projects, enabling teams to track experiments, manage models, and streamline deployment processes. For experimentation, testing, and reproducibility of the machine learning models in this project, you will use MLflow. MLflow will help track hyperparameter tuning by logging and comparing different model configurations. This allows you to easily identify and select the best-performing model based on the logged metrics.

- Please have a look here and follow the instructions: https://www.mlflow.org/docs/2.7.1/quickstart.html#quickstart

## 6. Team Members <a class="anchor" id="team-members"></a>

| Name                                                                      | Email           
|---------------------------------------------------------------------------|------------------ 
|1. Bonakele Mdletshe                                                       | Bonasiwe@gmail.com    