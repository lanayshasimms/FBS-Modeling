import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(2001)

def preprocess_data(df):
    """Initial data preprocessing steps"""
    
    # Convert season to categorical (string)
    df['season'] = df['season'].astype(str)
    
    # Replace missing ranks with 26 (unranked teams)
    df['rank_away'] = df['rank_away'].fillna(26)
    df['rank_home'] = df['rank_home'].fillna(26)
    
    # Replace missing week values with 17 (postseason)
    df['week'] = df['week'].fillna(17)
    
    # Filter out seasons 2002-2013
    seasons_to_remove = ['2002', '2003', '2004', '2005', '2006', '2007', 
                         '2008', '2009', '2010', '2011', '2012', '2013']
    df = df[~df['season'].isin(seasons_to_remove)].copy()
    
    # Filter for Big10/SEC teams
    big10_sec_teams = [
        'Washington', 'Michigan', 'Michigan State', 'Wisconsin', 'USC', 'UCLA',
        'Rutgers', 'Purdue', 'Penn State', 'Oregon', 'Ohio State', 'Northwestern',
        'Nebraska', 'Minnesota', 'Maryland', 'Iowa', 'Indiana', 'Illinois',
        'Alabama', 'Arkansas', 'Auburn', 'Florida', 'Georgia', 'Kentucky',
        'LSU', 'Mississippi State', 'Missouri', 'Oklahoma', 'Ole Miss',
        'South Carolina', 'Tennessee', 'Texas', 'Vanderbilt'
    ]
    df = df[df['home'].isin(big10_sec_teams)].copy()
    
    # Sort by home team, season, and week
    df = df.sort_values(['home', 'season', 'week']).reset_index(drop=True)
    
    return df

def generate_features(df):
    """Generate derived features"""
    
    # Calculate spread and win/loss
    df['spread'] = df['score_home'] - df['score_away']
    df['home_win'] = df['spread'].apply(lambda x: 'Win' if x > 0 else 'Loss')
    
    # Third down completion rates
    df['Home Third Down Completion Rate'] = np.where(
        df['third_down_att_home'] > 0,
        df['third_down_comp_home'] / df['third_down_att_home'],
        0
    )
    df['Away Third Down Completion Rate'] = np.where(
        df['third_down_att_away'] > 0,
        df['third_down_comp_away'] / df['third_down_att_away'],
        0
    )
    
    # Fourth down completion rates
    df['Home Fourth Down Completion Rate'] = np.where(
        df['fourth_down_att_home'] > 0,
        df['fourth_down_comp_home'] / df['fourth_down_att_home'],
        0
    )
    df['Away Fourth Down Completion Rate'] = np.where(
        df['fourth_down_att_away'] > 0,
        df['fourth_down_comp_away'] / df['fourth_down_att_away'],
        0
    )
    
    # Pass completion rates
    df['Home Pass Completion Rate'] = np.where(
        df['pass_att_home'] > 0,
        df['pass_comp_home'] / df['pass_att_home'],
        0
    )
    df['Away Pass Completion Rate'] = np.where(
        df['pass_att_away'] > 0,
        df['pass_comp_away'] / df['pass_att_away'],
        0
    )
    
    # Run-pass ratio
    df['Run-Pass Ratio'] = np.where(
        df['pass_att_home'] > 0,
        df['rush_att_home'] / df['pass_att_home'],
        0
    )
    
    # Rush yards to total yards ratio
    df['Rush Yards - Total Yards'] = np.where(
        df['total_yards_home'] > 0,
        df['rush_yards_home'] / df['total_yards_home'],
        0
    )
    
    # Create team-season identifier
    df['home_season'] = df['home'] + '_' + df['season']
    
    return df

def create_rolling_averages(df, window=3):
    """Create rolling averages for each team-season"""
    
    # Columns to create rolling averages for
    rolling_cols = [
        'spread', 'fum_home', 'q1_home', 'q2_home', 'q3_home', 'q4_home',
        'first_downs_home', 'Run-Pass Ratio', 'Rush Yards - Total Yards',
        'possession_home', 'int_home', 'pen_num_home', 'pen_yards_home',
        'Home Third Down Completion Rate', 'Home Pass Completion Rate'
    ]
    
    # Group by team-season and calculate rolling averages
    for col in rolling_cols:
        # Calculate rolling average
        df[f'average_{col.lower().replace(" ", "_").replace("-", "_")}'] = (
            df.groupby('home_season')[col]
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        )
    
    # For the first games in each team-season, use the actual value
    for col in rolling_cols:
        avg_col = f'average_{col.lower().replace(" ", "_").replace("-", "_")}'
        df[avg_col] = df[avg_col].fillna(df[col])
    
    return df

def select_features(df, remove_cols):
    """Remove unnecessary columns"""
    cols_to_keep = [col for col in df.columns if col not in remove_cols]
    return df[cols_to_keep]

def prepare_modeling_data(df):
    """Final preparation for modeling"""
    
    # Remove columns not needed for modeling
    cols_to_remove_first = [
        'attendance', 'away', 'Away Fourth Down Completion Rate',
        'Away Pass Completion Rate', 'Away Third Down Completion Rate',
        'conf_away', 'conf_home', 'date', 'first_downs_away',
        'fourth_down_att_away', 'fourth_down_att_home', 'fourth_down_comp_away',
        'fourth_down_comp_home', 'fum_away', 'home', 'Home Fourth Down Completion Rate',
        'int_away', 'ot_away', 'ot_home', 'pass_att_away', 'pass_att_home',
        'pass_comp_away', 'pass_comp_home', 'pass_yards_away', 'pen_num_away',
        'pen_yards_away', 'possession_away', 'q1_away', 'q2_away', 'q3_away',
        'q4_away', 'rush_att_away', 'rush_att_home', 'rush_yards_away',
        'rush_yards_home', 'score_away', 'score_home', 'season',
        'third_down_att_away', 'third_down_att_home', 'third_down_comp_away',
        'third_down_comp_home', 'time_et', 'total_yards_away', 'total_yards_home',
        'tv', 'pass_yards_home'
    ]
    
    df = select_features(df, cols_to_remove_first)
    
    # Remove the original columns (keep only averages)
    cols_to_remove_second = [
        'first_downs_home', 'fum_home', 'Home Pass Completion Rate',
        'Home Third Down Completion Rate', 'int_home', 'pen_num_home',
        'pen_yards_home', 'possession_home', 'q1_home', 'q2_home',
        'q3_home', 'q4_home', 'Run-Pass Ratio', 'Rush Yards - Total Yards'
    ]
    
    df = select_features(df, cols_to_remove_second)
    
    return df

def train_models(df):
    """Train and evaluate models using cross-validation"""
    
    results = {}
    
    # Prepare feature columns (exclude targets and identifiers)
    exclude_cols = ['spread', 'home_win', 'game_type', 'home_season', 
                    'neutral', 'rank_away', 'rank_home']
    
    # Model 1: Win/Loss Prediction (Decision Tree Classification)
    X_winloss = df.drop(columns=['spread', 'home_win'] + 
                        [c for c in exclude_cols if c in df.columns and c not in ['spread', 'home_win']])
    y_winloss = df['home_win']
    
    dt_classifier = DecisionTreeClassifier(
        criterion='entropy',  # gain_ratio approximation
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=2001
    )
    
    cv_scores_winloss = cross_val_score(dt_classifier, X_winloss, y_winloss, 
                                        cv=10, scoring='accuracy')
    results['winloss_accuracy'] = cv_scores_winloss.mean()
    
    # Model 2: Spread Prediction (Linear Regression)
    X_spread_lr = df.drop(columns=['spread', 'home_win'] + 
                          [c for c in exclude_cols if c in df.columns and c not in ['spread', 'home_win']])
    y_spread = df['spread']
    
    lr_model = LinearRegression()
    cv_scores_spread_lr = cross_validate(lr_model, X_spread_lr, y_spread, 
                                         cv=10, scoring=['r2', 'neg_mean_squared_error'])
    results['spread_lr_r2'] = cv_scores_spread_lr['test_r2'].mean()
    results['spread_lr_rmse'] = np.sqrt(-cv_scores_spread_lr['test_neg_mean_squared_error'].mean())
    
    # Model 3: Spread Prediction (Decision Tree Regression)
    X_spread_dt = df.drop(columns=['spread', 'home_win'] + 
                          [c for c in exclude_cols if c in df.columns and c not in ['spread', 'home_win']])
    
    dt_regressor = DecisionTreeRegressor(
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=2001
    )
    
    cv_scores_spread_dt = cross_validate(dt_regressor, X_spread_dt, y_spread, 
                                         cv=10, scoring=['r2', 'neg_mean_squared_error'])
    results['spread_dt_r2'] = cv_scores_spread_dt['test_r2'].mean()
    results['spread_dt_rmse'] = np.sqrt(-cv_scores_spread_dt['test_neg_mean_squared_error'].mean())
    
    return results

# main func
if __name__ == "__main__":
    df = pd.read_csv('cfb_box-scores_2002-2024.csv')
    
    # workflow
    df = preprocess_data(df)
    df = generate_features(df)
    df = create_rolling_averages(df, window=3)
    df = prepare_modeling_data(df)
    results = train_models(df)
    
    # print results
    print("\nModel Performance Results:")
    print(f"Win/Loss Accuracy: {results['winloss_accuracy']:.4f}")
    print(f"Spread Linear Regression R²: {results['spread_lr_r2']:.4f}")
    print(f"Spread Linear Regression RMSE: {results['spread_lr_rmse']:.4f}")
    print(f"Spread Decision Tree R²: {results['spread_dt_r2']:.4f}")
    print(f"Spread Decision Tree RMSE: {results['spread_dt_rmse']:.4f}")
    