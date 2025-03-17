import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

# adjust dataframe structure for better display in terminal
pd.set_option('display.max_columns', 30)  # display 30 columns
pd.set_option('display.width', None)  # display without linebreak

# Read CSV file/ Load dataset
df = pd.read_csv('Automobile_data.csv')
print(df.head())

# define a directory to store graphs and models
save_dir = '/Automobileprice/CarPrice/graph_pics'
model_dir = '/Automobileprice/CarPrice/models'

# data profiling and analysis
print(df.info())
print(df.describe())
print(df.describe(include='object'))

# Display all unique values
cols = df.columns

for col in cols:
    print(f'For {col}, datatype= {df[col].dtype} = {df[col].unique()}')


# Discovered "?" in the normalized loss, bore, stroke, horsepower, peak-rpm, num-of-doors and price column. Addressing...
# Display all the rows with "?" values
def find_rows_with_question_marks(frame):
    mask = (frame == '?').any(axis=1)  # Create a boolean mask where any column in a row is "?"
    rows_with_question_marks = frame[mask]
    return rows_with_question_marks


rows_containing_question_marks = find_rows_with_question_marks(df)
print(rows_containing_question_marks)

column_with_question = ['normalized-losses', 'bore', 'stroke', 'horsepower', 'peak-rpm', 'price']

for col_question in column_with_question:
    print(df[col_question].value_counts())  # checking the count of unique values in the column
    # Assuming we require this column, we try to fill the "?" values with mean data
    # replace "?" with NaN values first
    df[col_question] = df[col_question].replace('?', np.nan).astype(float)
    # can be done like this too instead of .astype(float)
    # df[col_question] = pd.to_numeric(df[col_question])
    # apply mean to the column
    df[col_question] = df[col_question].fillna(df[col_question].mean())

# for num-of-doors, since it is an object that cannot be converted into integer directly, we do this
df['num-of-doors'] = df['num-of-doors'].replace(['two', 'four', '?'], ['2', '4', np.nan]).astype(float)
df['num-of-doors'] = df['num-of-doors'].fillna(df['num-of-doors'].median())

# for num-of-cylinders, since it is an object that cannot be converted into integer directly, we do this
df['num-of-cylinders'] = df['num-of-cylinders'].replace(['four', 'six', 'five', 'three', 'twelve', 'two', 'eight'],
                                                        ['4', '6', '5', '3', '12', '2', '8']).astype(int)

print(df.info())


# EXPLORATORY DATA ANALYSIS(EDA)

# pair plot
# plt.figure(figsize=(20, 20))
# sns.pairplot(data=df, palette='coolwarm', hue='make')
# plt.savefig(os.path.join(save_dir, 'pairplot.png'))
# plt.close()
# print(plt.show())

# heatmap to check correlation
# plt.figure(figsize=(10, 10))
# sns.heatmap(data=df.corr(numeric_only=True), cmap='coolwarm', annot=True, fmt='.2f')
# plt.savefig(os.path.join(save_dir, 'heatmap.png'))
# plt.close()
# print(plt.show())

# count plot for categorical data columns
def count_plotter(col_name):
    sns.countplot(data=df, x=col_name, hue=col_name, palette='husl')
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(save_dir, col_name))
    plt.close()


# count_plotter('make')
# count_plotter('fuel-type')
# count_plotter('aspiration')
# count_plotter('body-style')
# count_plotter('drive-wheels')
# count_plotter('engine-location')
# count_plotter('engine-type')
# count_plotter('fuel-system')

def hist_plotter(col_name, color_c):
    sns.histplot(data=df, x=col_name, color=color_c, kde=True)
    plt.savefig(os.path.join(save_dir, f'{col_name}.png'))
    plt.tight_layout()
    plt.close()


# hist_plotter('normalized-losses', 'gold')
# hist_plotter('num-of-doors', 'deepskyblue')
# hist_plotter('wheel-base', 'slateblue')
# hist_plotter('length', 'violet')
# hist_plotter('width', 'springgreen')
# hist_plotter('height', 'lightcoral')
# hist_plotter('curb-weight', 'chocolate')
# hist_plotter('num-of-cylinders', 'orange')
# hist_plotter('bore', 'palegreen')
# hist_plotter('stroke', 'gray')
# hist_plotter('compression-ratio', 'tomato')
# hist_plotter('horsepower', 'salmon')
# hist_plotter('peak-rpm', 'peru')
# hist_plotter('city-mpg', 'navy')
# hist_plotter('highway-mpg', 'orchid')
# hist_plotter('price', 'pink')

sns.scatterplot(data=df, x='price', y='body-style', hue='body-style', palette='Paired')
plt.savefig(os.path.join(save_dir, 'test.png'))

sns.boxplot(data=df, y='make')
plt.savefig(os.path.join(save_dir, 'boxplot.png'))

# Encoding to convert categorical data into numerical data

# using label encoder for binary datas
to_transform = ['fuel-type', 'drive-wheels', 'aspiration', 'engine-location']
le = LabelEncoder()

for convert in to_transform:
    df[convert] = le.fit_transform(df[convert])

# using Ordinal Encoder to retain order as per importance
fuel_system_order = ['mfi', 'spfi', '4bbl', 'spdi', '1bbl', 'idi', '2bbl', 'mpfi']
oe_fuel_system = OrdinalEncoder(categories=[fuel_system_order])
df['fuel-system'] = oe_fuel_system.fit_transform(df[['fuel-system']])

body_style_order = ['hatchback', 'wagon', 'sedan', 'hardtop', 'convertible']
oe_body_style = OrdinalEncoder(categories=[body_style_order])
df['body-style'] = oe_body_style.fit_transform(df[['body-style']])

engine_type_order = ['ohcf', 'ohc', 'l', 'ohcv', 'rotor', 'dohc', 'dohcv']
oe_engine_type = OrdinalEncoder(categories=[engine_type_order])
df['engine-type'] = oe_engine_type.fit_transform(df[['engine-type']])

make_order = ['alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda', 'isuzu', 'jaguar',
              'mazda', 'mercedes-benz', 'mercury', 'mitsubishi', 'nissan', 'peugot',
              'plymouth', 'porsche', 'renault', 'saab', 'subaru', 'toyota', 'volkswagen',
              'volvo']

# categorizing car brands as low, medium and high range acc to current market status
categories = []
for brand in df['make']:
    if brand in ['chevrolet', 'dodge', 'plymouth', 'honda', 'isuzu', 'mitsubishi', 'nissan', 'toyota', 'volkswagen']:
        categories.append(0)  # Low-end
    elif brand in ['mazda', 'subaru', 'renault', 'mercury', 'saab', 'peugot', 'alfa-romero']:
        categories.append(1)  # Mid-range
    elif brand in ['audi', 'volvo', 'bmw', 'jaguar', 'mercedes-benz', 'porsche']:
        categories.append(2)  # High-end
    else:
        categories.append(-1)  # Unknown

df['make'] = categories

# feature selection
X = df.drop(['price'], axis=1)
y = df['price']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model selection
def train_and_evaluate_model(X_train, y_train, X_test, y_test, model, model_name, model_dir):
    pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])

    if isinstance(model, RandomForestRegressor):
        param_grid = {
            'model__n_estimators': [100, 500, 1000],
            'model__max_depth': [5, 10, 15],
            'model__min_samples_split': [2, 5, 10]
        }
    elif isinstance(model, DecisionTreeRegressor):
        param_grid = {
            'model__max_depth': [5, 10, 15],
            'model__min_samples_split': [2, 5, 10]
        }
    else:
        param_grid = {}

    if param_grid:
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2')
        grid_search.fit(X_train, y_train)
        print(f"Best parameters for {model_name}:", grid_search.best_params_)
        best_pipeline = grid_search.best_estimator_
    else:
        pipeline.fit(X_train, y_train)
        best_pipeline = pipeline

    y_pred = best_pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Results for {model_name}:")
    print(f"  MSE: {mse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R2: {r2:.2f}")

    joblib.dump(best_pipeline, os.path.join(model_dir, f'{model_name}_model.joblib'))
    print(f"  Model saved to {os.path.join(model_dir, f'{model_name}_model.joblib')}")


# Model calls
train_and_evaluate_model(X_train, y_train, X_test, y_test, LinearRegression(), 'LinearRegression', model_dir)
train_and_evaluate_model(X_train, y_train, X_test, y_test, RandomForestRegressor(), 'RandomForestRegressor', model_dir)
train_and_evaluate_model(X_train, y_train, X_test, y_test, DecisionTreeRegressor(), 'DecisionTreeRegressor', model_dir)

