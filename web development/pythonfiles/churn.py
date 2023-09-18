import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import io
pd.set_option('display.max_columns', None)
import plotly.express as px #for visualization
import matplotlib.pyplot as plt #for visualization
# Upload the CSV file
data_df = pd.read_csv("E:/socket/web development/pythonfiles/dataset1.csv")


#Read the dataset
#data_df = pd.read_csv("../data/churn.csv")

#Get overview of the data
def dataoveriew(df, message):
    print(f'{message}:n')
    print('Number of rows: ', df.shape[0])
    print("nNumber of features:", df.shape[1])
    print("nData Features:")
    print(df.columns.tolist())
    print("nMissing values:", df.isnull().sum().values.sum())
    print("nUnique values:")
    print(df.nunique())
dataoveriew(data_df, 'Overview of the dataset')
target_instance = data_df["Unsubscribed"].value_counts().to_frame()
target_instance = target_instance.reset_index()
target_instance = target_instance.rename(columns={'index': 'Category'})
fig = px.pie(target_instance, values='Unsubscribed', names='Category', color_discrete_sequence=["green", "red"],
             title='Distribution of Churn')
fig.show(renderer="colab")
#Defining bar chart function
def bar(feature, df=data_df ):
    #Groupby the categorical feature
    temp_df = df.groupby([feature, 'Unsubscribed']).size().reset_index()
    temp_df = temp_df.rename(columns={0:'Count'})
    #Calculate the value counts of each distribution and it's corresponding Percentages
    value_counts_df = df[feature].value_counts().to_frame().reset_index()
    categories = [cat[1][0] for cat in value_counts_df.iterrows()]
    #Calculate the value counts of each distribution and it's corresponding Percentages
    num_list = [num[1][1] for num in value_counts_df.iterrows()]
    div_list = [element / sum(num_list) for element in num_list]
    percentage = [round(element * 100,1) for element in div_list]
    #Defining string formatting for graph annotation
    #Numeric section
    def num_format(list_instance):
        formatted_str = ''
        for index,num in enumerate(list_instance):
            if index < len(list_instance)-2:
                formatted_str=formatted_str+f'{num}%, ' #append to empty string(formatted_str)
            elif index == len(list_instance)-2:
                formatted_str=formatted_str+f'{num}% & '
            else:
                formatted_str=formatted_str+f'{num}%'
        return formatted_str
    #Categorical section
    def str_format(list_instance):
        formatted_str = ''
        for index, cat in enumerate(list_instance):
            if index < len(list_instance)-2:
                formatted_str=formatted_str+f'{cat}, '
            elif index == len(list_instance)-2:
                formatted_str=formatted_str+f'{cat} & '
            else:
                formatted_str=formatted_str+f'{cat}'
        return formatted_str
    #Running the formatting functions
    num_str = num_format(percentage)
    cat_str = str_format(categories)

    #Setting graph framework
    fig = px.bar(temp_df, x=feature, y='Count', color='Unsubscribed', title=f'Churn rate by {feature}', barmode="group", color_discrete_sequence=["green", "red"])
    fig.add_annotation(
                text=f'Value count of distribution of {cat_str} are<br>{num_str} percentage respectively.',
                align='left',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=1.4,
                y=1.3,
                bordercolor='black',
                borderwidth=1)
    fig.update_layout(
        # margin space for the annotations on the right
        margin=dict(r=400),
    )

    return fig.show(renderer="colab")

#Gender feature plot
bar('Gender')
bar('Genre')

#SeniorCitizen feature plot
data_df.loc[data_df.IsActiveMember==0,'IsActiveMember'] = "No"   #convert 0 to No in all data instances
data_df.loc[data_df.IsActiveMember==1,'IsActiveMember'] = "Yes"  #convert 1 to Yes in all data instances
bar('IsActiveMember')

# The customerID column isnt useful as the feature is used for identification of customers.
# Drop unnecessary columns
data_df.drop(['RowNumber', 'Subscriber_name'], axis=1, inplace=True)
# Handle missing values in specific columns
data_df['Age'].fillna(data_df['Age'].median(), inplace=True)
data_df['Streamedtime'].fillna(data_df['Streamedtime'].median(), inplace=True)
columns_with_missing_values = ['Gender','Genre','NumOfVideos','Like_Dislike','IsActiveMember','Unsubscribed','Subscriber_id']
for column in columns_with_missing_values:
    data_df[column].fillna(data_df[column].mode()[0], inplace=True)
# Check if there are any remaining missing values
#missing_values = data_df.isnull().sum()
#print("Missing Values:")
#print(missing_values)
# Encode categorical features
#Defining the map function
def binary_map(feature):
    return feature.map({'Yes':1, 'No':0})
## Encoding target feature
#data_df['Unsubscribed'] = data_df[['Unsubscribed']].apply(binary_map)
# Encoding gender category
data_df['Gender'] = data_df['Gender'].map({'male':1, 'female':0})

#Encoding other binary category
#binary_list = ['IsActiveMember', 'Like/Dislike']
#data_df[binary_list] = data_df[binary_list].apply(binary_map)

#Encoding the other categoric features with more than two categories
data_df = pd.get_dummies(data_df, drop_first=True)

# Checking the correlation between features
corr = data_df.corr()

fig = px.imshow(corr,width=1000, height=1000)
fig.show()


#import statsmodels.api as sm
#import statsmodels.formula.api as smf

#Change variable name separators to '_'
#all_columns = [column.replace(" ", "_").replace("(", "_").replace(")", "_").replace("-", "_") for column in data_df.columns]
#all_columns = ['Gender','Genre','NumOfVideos','Like_Dislike','IsActiveMember','Unsubscribed','RowNumber','Subscriber_id']
#all_columns = ['RowNumber', 'Subscriber_id', 'Gender', 'Genre', 'NumOfVideos', 'Like_Dislike', 'IsActiveMember', 'Unsubscribed', 'Age', 'Streamedtime','Subscriber_name']

#Effect the change to the dataframe column names
#data_df.columns = all_columns

#Prepare it for the GLM formula
#glm_columns = [e for e in all_columns if e not in ['Subscriber_id', 'Unsubscribed']]
#glm_columns = ' + '.join(map(str, glm_columns))

#Fiting it to the Generalized Linear Model
#glm_model = smf.glm(formula=f'Unsubscribed ~ {glm_columns}', data=data_df, family=sm.families.Binomial())
#res = glm_model.fit()
#print(res.summary())
#np.exp(res.params)



#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
data_df['Age'] = sc.fit_transform(data_df[['Age']])
data_df['NumOfVideos'] = sc.fit_transform(data_df[['NumOfVideos']])
data_df['Streamedtime'] = sc.fit_transform(data_df[['Streamedtime']])

# Import Machine learning algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

#Import metric for performance evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#Split data into train and test sets
from sklearn.model_selection import train_test_split
X = data_df.drop('Unsubscribed', axis=1)
y = data_df['Unsubscribed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

#Defining the modelling function
def modeling(alg, alg_name, params={}):
    model = alg(**params) #Instantiating the algorithm class and unpacking parameters if any
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #Performance evaluation

    def print_scores(alg, y_true, y_pred):
        print(alg_name)
        acc_score = accuracy_score(y_true, y_pred)
        print("accuracy: ",acc_score)
        pre_score = precision_score(y_true, y_pred)
        print("precision: ",pre_score)
        rec_score = recall_score(y_true, y_pred)
        print("recall: ",rec_score)
        f_score = f1_score(y_true, y_pred, average='weighted')
        print("f1_score: ",f_score)

    print_scores(alg, y_test, y_pred)
    return model


# Running logistic regression model
log_model = modeling(LogisticRegression, 'Logistic Regression')

### Trying other machine learning algorithms: SVC
svc_model = modeling(SVC, 'SVC Classification')

#Random forest
rf_model = modeling(RandomForestClassifier, "Random Forest Classification")

#Decision tree
dt_model = modeling(DecisionTreeClassifier, "Decision Tree Classification")

#Naive bayes
nb_model = modeling(GaussianNB, "Naive Bayes Classification")