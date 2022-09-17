# Function to summarize data
def summary_stats(data, title):  
    """
    Generates a Summary table containing the most relevant information of a dataset

    Parameters:
    ----------
    data : dataframe
        Data to summarize
    title : str
        Title of the graph

    Returns:
    --------
    Dataframe
    """ 
    # Generate a general summary of the variables
    df_missingval = pd.DataFrame(data.isna().any(), columns=['Missing vals'])                   # Check if there are any missing values
    df_types = pd.DataFrame(data.dtypes, columns=['Variable type'])                             # Obtain the datatypes of all colums
    df_describe = data.describe().round(decimals=2).transpose()                                 # Generate summary statistics
    _ = pd.merge(df_missingval, df_types, how='inner', left_index=True, right_index=True)       # Intermediate merge types and missing val
    df_var_summary = pd.merge(df_describe, _ , how='outer', left_index=True, right_index=True)  # Final merge 
    #df_var_summary.loc['date_of_birth', 'count'] = len(data.index)                             # Replace count of date_of_birth
    print(title.center(120))

    return df_var_summary

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Function that computes new variables for any of the datasets

def gen_variables(data):
    """ 
    Creates the list of new variables mentioned in section 1. 
    
    Parameters:
    ----------
    data : dataframe
        Original dataframe to transform
    
    Returns: 
    -------
    dataframe
        A dataframe with the new variables

    """
    df = data
    # Generate age
    today = pd.Timestamp(dt.date.today())                                                   # Get today's date
    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], infer_datetime_format=True)   # Convert column into datetime
    df['age'] = df['date_of_birth'].apply(lambda x: (today - pd.Timestamp(x)).days)         # Calculate dif between dates
    df['age'] = round(df['age'] / 365, 0)                                                   # Convert into years
    df['age'] = df['age'].astype(int)                                                       # Convert column into integer
    df = df.drop(columns=['date_of_birth', 'id'])                                           # Drop date of birth and id column                                             

    # Generate total debt
    df['total_debt'] = df['debt_to_income_ratio'] * df['monthly_income']

    # Generate number_accounts
    df['number_accounts'] = df['number_open_credit_lines'] + df['number_open_loans']

    # Generate avg loan
    df['avg_loan'] = df['total_debt'] / df['number_accounts']

    # Correct division with 0
    df['avg_loan'] = df['avg_loan'].replace(np.inf, 0)

    # Generate 90 days pct
    df['90_days_pct'] = df['number_90_days_past_due'] / df['number_accounts']

    # Correct division with 0
    df['90_days_pct'] = df['90_days_pct'].replace(np.inf, 0)
    df['90_days_pct'] = df['90_days_pct'].replace(np.nan, 0)

    # Generate charged_off_pct
    df['charged_off_pct'] = df['number_charged_off'] / df['number_accounts']

    # Correct division with 0
    df['charged_off_pct'] = df['charged_off_pct'].replace(np.inf, 0)
    df['charged_off_pct'] = df['charged_off_pct'].replace(np.nan, 0)

    # Generate avg score
    df['avg_score'] = (df['score2'] +  df['score1']) / 2

    # Generate score pct change
    df['score_change'] = (df['score2'] - df['score1'])
    
    return df

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Function that plots correlation matrix and correlation matrix in absolute value
def corr_matrix(data, file_name, annot_size = 9, fig_num = 3):
    """ 
    Plots correlation matrix and correlation matrix in absolute value 
    
    Parameters:
    ----------
    data : dataframe
        Original dataframe to transform

    file_name : str
        File to store the image

    annot_size : float
        Size of the annotations for the plot

    fig_num : float
        Number of the figure title
    
    Returns: 
    -------
    None

    """
    # Creating correlation matrix graph
    fig, ax = plt.subplots(1,2)   
    fig.set_size_inches((25,9))                                                                 # Create a 5x2 grid of subplots
    plt.suptitle('Figure {}. Correlation heatmaps'.format(fig_num), x=0.4, y=0.95)

    # Mask to remove diagonal to make scale more visually attractive
    l = len(data.columns) 
    mask = np.zeros((l, l), int)
    np.fill_diagonal(mask, 1)
    mask

    # Graphing correlation matrix with original sign
    ax[0].set_title('a) Original correlations')
    sns.heatmap(data.corr().round(2),                 
                annot=True, mask=mask, cmap='BrBG', center=0, ax=ax[0], annot_kws={"size": annot_size})
    # Graphing correlation matrix with absolute value
    ax[1].set_title('b) Absolute value correlations')
    sns.heatmap(data.corr().abs().round(2), 
                annot=True, mask=mask, cmap='BrBG', center=0, ax=ax[1], annot_kws={"size": annot_size})

    plt.show()

    plt.savefig('figures/{}'.format(file_name));

    return None

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Prints as summary of the issues found during the exploratory data analysis
def issues_recap(df_train, df_test):
    # Querying relevant information 
    count_missing_train = len(df_train[df_train['number_dependants'] == -1])
    count_missing_test = len(df_test[df_test['number_dependants'] == -1])

    count_no_hist_train = len(df_train[df_train['number_accounts'] == 0])
    count_no_hist_test = len(df_test[df_test['number_accounts'] == 0])

    count_95_yrs_train = len(df_train[df_train['age'] >= 95])
    count_95_yrs_test = len(df_train[df_train['age'] >= 95])

    print('Missing values')
    print('- Total missing values training : {}'.format(count_missing_train))
    print('- Total missing values testing : {}'.format(count_missing_test))
    print('No credit history')
    print('- Total clients without history training : {}'.format(count_no_hist_train))
    print('- Total clients without history testing : {}'.format(count_no_hist_test))
    print('Age')
    print('- Total clients over 95 years train: {}'.format(count_95_yrs_train))
    print('- Total clients over 95 years train: {}'.format(count_95_yrs_test))

    # Graphic representation of age disconituity
    df_train.groupby(['age']).agg({'number_dependants': np.mean})
    tail = df_train.groupby(['age'])['age'].count().tail(15)

    plt.figure(figsize=(10,5))
    plt.plot(tail.index, tail, color='lightslategrey')
    plt.bar(tail.index, tail, color='lightslategrey')
    plt.title('Figure 6. Age discointinuity')
    plt.show()
    plt.savefig('figures/age_discontinuity');

    return None

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Function that replaces misssing values of number of dependants 
def fill_missing(data):
    """" 
    Fills the missing values of (-1) in the number_dependants column

    Paramters:
    ----------
    data : dataframe
        Dataframe to correct
    
    Returns:
    --------
    dataframe
        Corrected dataframe

    """
    mode_dependants = data['number_dependants'].mode()[0]
    data['number_dependants'] = data['number_dependants'].replace(-1, mode_dependants)

    return data

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Function that misses wrong ages
def fix_age(data):
    """" 
    Fixes age column for individuals over 118 years

    Paramters:
    ----------
    data : dataframe
        Dataframe to correct
    
    Returns:
    --------
    dataframe
        Corrected dataframe
    """
    data['age'] = data['age'].apply(lambda x: (x - 100) if x >= 118 else x )

    return data

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def standarize(data): 
    """
    This function  standarizes the values of our datasets
    
    Parameters:
    ------------
    data : dataframe
        Dataset to transform
    
    Returns:
    --------
    dataframe
        A dataframe with the data standarized on the selected columns
    """
    # Rescaling selected attributed
    scaler = MinMaxScaler()                         # Create scaler object 
    scaler.fit(data)                                # Fit the data
    scalers = pd.DataFrame(scaler.transform(data))  # Transform data
    scalers.columns = data.columns                  # Rename columns for scaled frame
    scalers
    data = scalers

    return scalers

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Illustrative table of cases
def possible_outcomes():
    # Create table with all potential outcomes of a classification process
    concepts = pd.DataFrame(np.array([['True negative', 'False positive'],
                                    ['False negative', 'True positive']]))

    concepts.columns = ['Predicted Non-default', 'Predicted Default']
    concepts.index = ['Non-default', 'Default']

    print('Table 4. Possible outcomes'.center(48))
    return concepts

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Function to generate confusion matrix
def plot_confusion_matrix(y_target, y_predicted, title = 'Confusion Matrix of classifier'):
    """ 
    Creates a confusion matrix 
    
    Parameters:
    -----------
    y_target : array
        real values of y

    y_predicted : array
        predicted values of y

    Returns:
    --------
    Dictionary with metrics

    """
    # Genrate confusion matrix
    cm = confusion_matrix(y_target, y_predicted, normalize='true')
    # Generate heatmap
    sns.heatmap(cm, annot=True, cmap='BrBG')
    # Compute metrics
    accurate = round(accuracy_score(y_target, y_predicted)*100, 2)
    precision = round(precision_score(y_target, y_predicted)*100, 2)
    recall = round(recall_score(y_target, y_predicted)*100, 2)

    plt.title('{}\n' 
                'Accuracy = {}%\n'
                'Precision = {}%\n'
                'Recall = {}%'.format(title, accurate, precision, recall))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.show()

    return None