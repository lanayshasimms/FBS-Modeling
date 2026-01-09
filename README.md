# Modeling FBS Team Performance to Forecast Wins in Sports Analytics

A small class project/stepping stone to future research in sports anayltics and the effects of rolling averages on the prediction of Win/Loss and Point Spread. 
A initial assignment report [INSERT NAME HERE] shows the results of the first iteration of this ongoing project, and provides a more in-depth explanation of the overview of the project.
Below is an introduction of the files and what they do as far as my workflow (as well as images of the workflow and how they should render in RapidMiner) as well as the csv files used in this initial iteration.
All data preprocessing, mining, and results were conducted in Altair RapidMiner (AI Studio 2026.0.1), and recreation of these results are easily reproducible if the files are run in the software. 

## Introduction of Files + Data

### College Football Team Stats 2002 - January 2024
The original data set used/manipulated in the work can be found at this [link](https://www.kaggle.com/datasets/cviaxmiwnptr/college-football-team-stats-2002-to-january-2024/data) and the data includes 58 attributes and 18,909 samples.
Each sample/row in the dataset pertains to a specific game in a specific season (year) between two teams, and includes attributes such as `score_home`, `score_away`, `season`, `week`, etc. 

### feature Reduced Set + Training Set with Rolling Averages

[Big10/SEC Scores](big10_sec_rolling_averages.csv)

The original data set did not include rolling averages, as well as necessary variables such as `spread` (home team score - away team score), and many rates that were 
necessary for what I wanted to assess such as pass completion rates, third down completion rates, etc. ------- and starts with an encoding of a numerical value
to nominal (season) to ensure the semantics of the values are used rather than the numerical values. The
data allowed for empty values in the event that a team was unranked, and rankings are only awarded 1
through 25, so to treat all unranked teams as equal, a ’26’ ranking was given as a placeholder. The pipeline
continues by replacing empty postseason values with a week ‘17’, and to account for the newer era of playoffs
that began in 2014 and to reduce the data to a relevant subset, the data from 2002-2013 was removed.
To reduce the data further, two large conferences in the FBS, The Big Ten conference (B1G 10) and the
South Eastern Conference (SEC) teams were only accounted for as the basis for the home team statistics
(26 teams). In order to use rolling averages of the teams performance throughout the season, the data was
first sorted by team and concatenated with year to serve as the variable by which to determine the averages
per team/season for the weeks in their season. A table below shows the 
