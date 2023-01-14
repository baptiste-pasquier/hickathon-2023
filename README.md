# Hi!ckathon 3 – Hi! PARIS

## Installation

1. Clone the repository
```bash
git clone https://github.com/baptiste-pasquier/hickathon-2023
```

2. Install the project
- With `poetry` ([installation](https://python-poetry.org/docs/#installation)) :
```bash
poetry install
```
- With `pip` :
```bash
pip install -e .
```

## Usage

```bash
python run.py
```
```
optional arguments:  -h, --help  show this help message and exit
  --train     Train the model
  --predict   Predict on the test dataset
  --small     Use the small dataset
```
Arguments :

- `--train` : train the model
  - datasets : `datasets/train/train_features_sent.csv` and `datasets/train/train_labels_sent.csv`
  - output : trained model at `trained_model/trained_model.pickle`
- `--predict` : predict with the trained model
  - dataset : `datasets/test/test_features_sent.csv`
  - output : `datasets/test/submission_sent.csv`
- `--small` : use the datasets in the `small` subfolder

# I.	OVERVIEW
## 1.	Project Background and Description

One of the main GHG emissions sources is the real estate sector, in which energy sobriety remains underdeveloped. Indeed real estate owners are reluctant to undertake refurbishment works as they only see them as financial losses. To overcome the environmental challenge we’re faced with, dealing with this short-sightedness is a necessity. We believe that fair and accurate information is crucial to achieve energy sobriety. We think that there would be more refurbishments if owners knew precisely what they had to win. Our vision is thus to empower building owners with the fittest data to overcome short sighting. Our ambition is to develop a solution that would allow real estate owners to know what the optimal renovation is for their building, how much energy, and thus how much money, it would save them.

## 2.	Project Scope

We want to predict as accurately as possible the annual energy consumption per square foot of a building. That would allow us to simulate the annual energy savings for any kind of refurbishment like the adding of an external wall insulation or of window glazing. Hence, we could optimize and find the best renovation for any particular building.
## 3.	Presentation of the group

| First name | Last name | Year of studies & Profile | School      | Skills                                                                          | Roles/Tasks                                                            |
| ---------- | --------- | ------------------------- | ----------- | ------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Joséphine  | Gardette  | M1                        | HEC         | Communicational and business skills                                             | Video making and data exploration                                      | Answer |
| Nadia      | Zargouni  | M1                        | HEC         | Expertise in business modeling, business skills                                 | Business model, data preprocessing, video making                       |
| Benjamin   | Amsellem  | M1                        | HEC & ENSAE | Data science and business,  Capability to work with both teams to join any task | Coordinator, Linking business and technical units ; Data preprocessing |
| François   | Bertholom | M2                        | ENSAE       | Data Science and Pandas expert                                                  | Data preprocessing supervisor, model optimization, feature engineering |
| Baptiste   | Pasquier  | M2                        | ENSAE       | Code & project management                                                       | Git lord, Dataset optimization                                         |
| Coni       | Soret     | M2                        | ENSAE       | Strong theoretical knowledge                                                    | ML engineer ; feature engineering and model selection                  |

## 4.	Task Management

We used collaborative tools to optimize workflow: Slack to share files, code, and messages. Google Docs for the business pitch and ideas. For the preprocessing, we split the set of columns so that everyone could work on different variables and we used Google Sheet to summarize and visualize the progress: one line per variable, each person knew which variables to handle and could fill with useful informations.
Everybody in the team worked on data exploration. For the rest of the tasks we split the team in two units, technical data science unit and business and communication unit. Benjamin linked the two units and worked with both teams on specific tasks to help whenever there was a need.

# II.	PROJECT MANAGEMENT
## 1.	Data Understanding

We began our work with a comprehensive exploratory analysis: data types check, missing values and duplicates spotting for each variable. We then studied the distribution of each variable. For categorical ones, we examined the different modalities through barplots and value counts ; for numerical ones, we looked at various statistical characteristics such as the median, mean, and quantiles, and plotted histograms both with and without the 5% extremal quantiles.
During this exploration, we also computed the correlations between all variables and the target variable (for the categorical variables we used groupby. mean), i.e. the energy consumption per square foot over a year. We found the correlations to be very low, however, we did find mild correlations between numerical variables and the total energy consumption of the buildings.
This information helped us to identify potential areas of focus for the data preprocessing and model building.

## 2.	Data Pre-processing

This step was crucial to ensure that the data was clean, consistent, and ready for analysis.
First, we removed duplicate rows and outliers from the dataset. Categorical variables were either one-hot encoded or categorically encoded, and some required additional work (e.g. creating separate columns of different types of insulation, which can be combined). To save memory and improve processing times, we also changed the data types of some variables, often converting floats encoded on 64 bits to booleans.
Additionally, we addressed missing values in the dataset by deciding on an appropriate method for each column, such as using the median or the most frequent class.
To enhance the information available in the dataset and improve the performance of our models, we also created new features. One example is the total wall surface (calculated as if the building had a square base) multiplied by the thermal conductivity and thickness of the walls. This helped to include relevant information that would have been difficult to extract given the types of models we used. We did not take this step as far as we would have liked, though, as we believe including meteorological data and more precise location information could have dramatically improved the quality of our predictions but we weren’t able to find the actual location corresponding to the area codes.

## 3.	Modeling Development

Using our theoretical knowledge we restricted the set of interesting models we would like to test, for instance we excluded linear models and neural networks. We immediately focused on the tree models class which is known to be the most efficient for tabular data. We found out that the best performing model was XGBoost regression. To optimize the hyperparameters we started with random search and then switched to grid search once we determined a restricted space of parameters.

## 4.	 Deployment Strategy

We tried making every step as general and robust as possible by deploying our code as one large pipeline made of several smaller and more specific pipelines.

# III.	CARBON FOOTPRINT LIMITATION

In order to save computational power and energy and limit our carbon footprint, we carried out our random search on a small subset of the data and changed data types as early as possible to shrink the dataset size.

# IV.	CONCLUSION

We could enhance the granularity of information by exploiting the area code and adding geographical and meteorological data.
