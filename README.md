# Modeling FBS Team Performance to Forecast Wins in Sports Analytics

A small class project/stepping stone to future research in sports anayltics and the effects of rolling averages on the prediction of Win/Loss and Point Spread. 
A initial assignment report [Final_Report.pdf](Final_Report.pdf) shows the results of the first iteration of this ongoing project, and provides a more in-depth explanation of the overview + results of the project.
Below is an introduction of the files and what they do as far as my workflow (as well as images of the workflow and how they should render in RapidMiner) as well as the csv files used in this initial iteration.
All data preprocessing, mining, and results were conducted in Altair RapidMiner (AI Studio 2026.0.1), and recreation of these results are easily reproducible if the files are run in the software. 

## Introduction of Files + Data

### College Football Team Stats 2002 - January 2024
The original data set used/manipulated in the work can be found at this [link](https://www.kaggle.com/datasets/cviaxmiwnptr/college-football-team-stats-2002-to-january-2024/data) and the data includes 58 attributes and 18,909 samples.
Each sample/row in the dataset pertains to a specific game in a specific season (year) between two teams, and includes attributes such as `score_home`, `score_away`, `season`, `week`, etc. The relevant data sets used are below, but all of the derived data came from this dataset.  

### Feature Reduced Set + Training Set with Rolling Averages
[Feature_Reduced_Set.csv](Feature_Reduced_Set.csv) + [Training_Set_Rolling_Averages.csv](Training_Set_Rolling_Averages.csv) 

The original data set did not include rolling averages, as well as necessary variables I wanted to use for rolling averages such as `spread` (home team score - away team score), and many rates that were 
necessary for what I wanted to assess such as pass completion rates, third down completion rates, etc. The preprocessed flow starts with an encoding of a numerical value
to nominal (season) to ensure the semantics of the values are used rather than the numerical values. The
data allowed for empty values in the event that a team was unranked, and rankings are only awarded 1
through 25, so to treat all unranked teams as equal, a ’26’ ranking was given as a placeholder. The pipeline
continues by replacing empty postseason values with a week ‘17’, and to account for the newer era of playoffs
that began in 2014 and to reduce the data to a relevant subset, the data from 2002-2013 was removed.
To reduce the data further, two large conferences in the FBS, The Big Ten conference (B1G 10) and the
South Eastern Conference (SEC) teams were only accounted for as the basis for the home team statistics
(26 teams). In order to use rolling averages of the teams performance throughout the season, the data was
first sorted by team and concatenated with year to serve as the variable by which to determine the averages
per team/season for the weeks in their season. A table below shows plaintext/actual formulas of the relevant generated feautures that were then used to create rolling averages.

| Attribute Name                  | Calculation                        | Description                                     |
|---------------------------------|------------------------------------|-------------------------------------------------|
| `spread`                          | score_home - score_away            | Home Team Score - Away Team Score
| `home_win`                        | if (spread > 0, "Win", "Loss")     | If the spread is positive, it's a Win, else Loss
| `Home Third Down Completion Rate` | if (third_down_att_home > 0, third_down_comp_home/third_down_att_home, 0) | If the Third Down Attempts were not 0, compute Attempts / Completions, else 0
| `Home Fourth Down Completion Rate`| if(fourth_down_att_home > 0, fourth_down_comp_home/fourth_down_att_home, 0) | If the Fourth Down Attempts were not 0, compute Attempts / Completions, else 0
| `Home Pass Completion Rate`       | if(pass_att_home > 0, pass_comp_home/pass_att_home, 0) | If the Pass Attempts were not 0, compute Attempts / Completions, else 0
| `Run-Pass Ratio`                  | if(pass_att_home > 0, rush_att_home/pass_att_home, 0) | If the Pass Attempts were not 0, compute Rush Attempts / Pass Completions, else 0

The [Pre_Rolling_Averages_Big10_+_SEC.rmp](Pre_Rolling_Averages_Big10_+_SEC.rmp) and [Rolling_Averages_Big10_+_SEC.rmp](Rolling_Averages_Big10_+_SEC.rmp) RapidMiner files provide a further breakdown of the flow as to how the data set was manipulated and how rolling averages were calculated; the separate flows are both apart of the work that happens in the main flow ([Data_Preprocessing_+_Mining_+_Store_Models.rmp](Data_Preprocessing_+_Mining_+_Store_Models.rmp)).

### Michigan 2025 Compiled Box Score Data + Test Set
[Michigan_2025_Box_Scores.csv](Michigan_2025_Box_Scores.csv) + [Michigan_2025_Rolling_Averages.csv](Michigan_2025_Rolling_Averages.csv)

Cross Validation with 10 folds was used in all of the models tested, but in an effort to have to look at truly unseen data, the 2025 home season box scores from the Michigan Football Team were compiled in order to get the same generated variables as well as the rolling averages as further testing for the models. In the future, I hope to compile the data of all of the Big10/SEC teams from 2025 to get a further comprehensive test, but the compilation of Michigan's data sufficed for this assignment. 

The [Rolling_Averages_of_Michigan_2025_Home_Season.rmp](Rolling_Averages_of_Michigan_2025_Home_Season.rmp) RapidMiner file provides a further breakdown of the flow as to how the data set was manipulated and how rolling averages were calculated; the data manipulated here is apart of the work that happens in the model testing flow on the Michigan data ([Model_Tests_on_Michigan_2025.rmp](Model_Tests_on_Michigan_2025.rmp)).

### Main Flow + Michigan 2025 Data Testing + Models
[Data_Preprocessing_+_Mining_+_Store_Models.rmp](Data_Preprocessing_+_Mining_+_Store_Models.rmp) + [Model_Tests_on_Michigan_2025.rmp](Model_Tests_on_Michigan_2025.rmp)

These files include the entire flow of the preprocessing, mining, and results of the 10-fold cross validation, and the model testing on the 2025 Michigan Season (home games).

[Spread - Decision Tree Model](spread_dt_model.rmp.rmmodel) + [Spread - Linear Regression Model](spread_lr_model.rmp.rmmodel) + [Win/Loss - Decision Tree Model](winloss_dt_model.rmp.rmmodel)

These files include all three of the models that were stored from the initial preprocessing and mining and used in the testing on the 2025 Michigan Season. 


