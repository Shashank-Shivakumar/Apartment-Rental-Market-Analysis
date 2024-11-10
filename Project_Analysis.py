import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

file_path = os.path.join(os.getcwd(), "apartments_data.csv")
if not os.path.exists(file_path):
    raise FileNotFoundError("File not found in the current working directory!")

try:
    data = pd.read_csv(file_path, sep=';', encoding='ISO-8859-1')
except UnicodeDecodeError:
    data = pd.read_csv(file_path, sep=';', encoding='unicode_escape')

# print(data.head())
# print(data.info())
print(data.shape)

# Check for missing values in each column
missing_data = data.isnull().sum()
missing_data = missing_data[missing_data > 0].sort_values(ascending=False)  # Filter out columns with no missing values

plt.figure(figsize=(12, 8))  # Adjust the figure size as necessary
sns.barplot(x=missing_data.index, y=missing_data.values)
plt.xlabel('Columns')
plt.ylabel('Number of Missing Values (log scale)')
plt.title('Missing Data in Each Column')
plt.xticks(rotation=90)  # Rotates labels to make them readable
plt.yscale('log')  # Sets the y-axis to logarithmic scale
plt.show()

missing_data 

table = PrettyTable()
table.field_names = ["Column Name", "Missing Values"]
table.title = "Missing Data Summary"

# Populate the table
for column, value in missing_data.items():
    table.add_row([column, value])

print(table)

data['amenities'] = data['amenities'].fillna('Not Specified')

data['pets_allowed'] = data['pets_allowed'].fillna('Unknown')
data['pets_allowed'] = data['pets_allowed'].apply(lambda x: x if x in ["None", "Cats", "Dogs", "Cats,Dogs"] else "Unknown")

# 2. Replace missing values in 'bedrooms' and 'bathrooms' with their respective medians
data['bedrooms'] = data['bedrooms'].fillna(data['bedrooms'].median())
data['bathrooms'] = data['bathrooms'].fillna(data['bathrooms'].median())

# 3. Drop rows where 'latitude', 'longitude', 'price', or 'price_display' are missing
data = data.dropna(subset=['latitude', 'longitude', 'price', 'price_display'])

data['cityname'] = data['cityname'].fillna('Unknown')
data['state'] = data['state'].fillna('Unknown')

data = data.drop("address", axis=1)

cleaned_data_new = data

print(cleaned_data_new.head(5))

cleaned_missing_data = cleaned_data_new.isnull().sum()
cleaned_missing_data = cleaned_missing_data[cleaned_missing_data > 0].sort_values(ascending=False)

cleaned_missing_data
#No missing values
print(cleaned_data_new.shape)
cleaned_data_new.describe()

def detect_outliers_iqr(data):
    outliers = {}
    for column in data.columns:
        if data[column].dtype in ['int64', 'float64']:  # Ensuring we only check numerical data
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            outlier_step = 1.5 * IQR
            outlier_conditions = (data[column] < Q1 - outlier_step) | (data[column] > Q3 + outlier_step)
            outlier_data = data[outlier_conditions]
            if len(outlier_data) > 1:  # Only consider columns with more than one outlier
                outliers[column] = outlier_data
    return outliers

outliers = detect_outliers_iqr(cleaned_data_new)

table = PrettyTable()
table.title = 'Outlier Summary'
table.field_names = ["Column", "Number of Outliers"]
table.sortby = "Number of Outliers"
table.reversesort = False

for column, outlier_df in outliers.items():
    table.add_row([column, outlier_df.shape[0]])

print(table)

selected_columns = list(outliers.keys())[:6]  # This limits the plots to the first six variables with outliers

# Setup for subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))  # 2 rows and 3 columns of subplots
axes = axes.flatten()  # Flatten the array to make indexing easier

# Colors for the boxplots
colors = ['skyblue', 'lightgreen', 'tan', 'pink', 'purple', 'orange']

# Plotting each set of outliers in its own subplot, with color
for ax, column, color in zip(axes, selected_columns, colors):
    # Generate boxplot with a specified color
    box = ax.boxplot(outliers[column][column], vert=True, patch_artist=True)
    for patch in box['boxes']:
        patch.set_facecolor(color)
    ax.set_title(f'Boxplot of {column} with Outliers')
    ax.set_xlabel(column)
    ax.grid(True)  # Basic grid

# Hide any unused axes if there are less than 6 plots
for i in range(len(selected_columns), 6):
    axes[i].axis('off')

plt.tight_layout()
plt.show()

#Removing outliers
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Columns to clean
columns_to_clean = ['bathrooms', 'bedrooms', 'price', 'square_feet', 'latitude', 'longitude']

# Clean the data for each column
cleaned_data = cleaned_data_new.copy()
for column in columns_to_clean:
    cleaned_data = remove_outliers_iqr(cleaned_data, column)

cleaned_data.describe()

# Performing PCA
# Select numeric columns
numeric_data = cleaned_data.select_dtypes(include=[np.number])

# Handling missing values by filling them with the mean of each column
numeric_data_filled = numeric_data.fillna(numeric_data.mean())

# Standardize the data
scaler = StandardScaler()
numeric_data_scaled = scaler.fit_transform(numeric_data_filled)

# Fit PCA with a specified number of components
n_components = 6
pca = PCA(n_components=n_components)
pca.fit(numeric_data_scaled)

# Calculate cumulative explained variance
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Create a line plot for cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(
    range(1, len(cumulative_explained_variance) + 1),
    cumulative_explained_variance,
    marker='o',
    linestyle='-',
    label='Cumulative Explained Variance'
)

# Add a threshold line at 85% explained variance
threshold = 0.85
plt.axhline(y=threshold, color='red', linestyle='--', linewidth=1, label=f'{threshold * 100}% Variance')

# Annotations to highlight key points
plt.annotate(
    f'{cumulative_explained_variance[-1]:.2f}',
    xy=(n_components, cumulative_explained_variance[-1]),
    xytext=(n_components, cumulative_explained_variance[-1] + 0.05),
    textcoords='data',
    ha='center',
    arrowprops=dict(facecolor='black', shrink=0.05)
)

# Add plot labels and title
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance by Number of Components")
plt.legend()
plt.grid(True)
plt.show()

cov_matrix_pca = np.cov(numeric_data_scaled @ pca.components_.T, rowvar=False)

# Calculate the condition number of the covariance matrix.
condition_number_pca = np.linalg.cond(cov_matrix_pca)

# Retrieve the singular values from the PCA results.
singular_values_pca = pca.singular_values_

pca_metrics = (condition_number_pca, singular_values_pca)

print("Condition Number:", pca_metrics[0])
print("Singular Values:", pca_metrics[1])

#Normality test
from statsmodels.graphics.gofplots import qqplot

# Plotting QQ Plots for price and square_feet
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# QQ plot for price
qqplot(cleaned_data['price'], line='s', ax=ax1)
ax1.set_title('QQ Plot of Price')

# QQ plot for square_feet
qqplot(cleaned_data['square_feet'], line='s', ax=ax2)  # Dropping NA values for valid plot
ax2.set_title('QQ Plot of Square Feet')

plt.tight_layout()
plt.show()

from scipy.stats import kstest, shapiro, normaltest

# Perform Kolmogorov-Smirnov Test for normality on price and square_feet
ks_test_price = kstest(cleaned_data['price'], 'norm', args=(cleaned_data['price'].mean(), cleaned_data['price'].std()))
ks_test_square_feet = kstest(cleaned_data['square_feet'].dropna(), 'norm', 
                             args=(cleaned_data['square_feet'].dropna().mean(), cleaned_data['square_feet'].dropna().std()))

# Perform Shapiro-Wilk Test for normality
shapiro_test_price = shapiro(cleaned_data['price'])
shapiro_test_square_feet = shapiro(cleaned_data['square_feet'].dropna())

# Perform D'Agostino's K² Test for normality
dagostino_test_price = normaltest(cleaned_data['price'])
dagostino_test_square_feet = normaltest(cleaned_data['square_feet'].dropna())

table = PrettyTable()

# Set the column names
table.field_names = ["Test", "Statistic", "P-value", "Conclusion"]

# Populate the table with results
table.add_row(["K-S Test Price", f"{ks_test_price.statistic:.3f}", f"{ks_test_price.pvalue:.3e}", "Not Normal"])
table.add_row(["K-S Test Square Feet", f"{ks_test_square_feet.statistic:.3f}", f"{ks_test_square_feet.pvalue:.3e}", "Not Normal"])
table.add_row(["Shapiro Test Price", f"{shapiro_test_price[0]:.3f}", f"{shapiro_test_price[1]:.3e}", "Not Normal"])
table.add_row(["Shapiro Test Square Feet", f"{shapiro_test_square_feet[0]:.3f}", f"{shapiro_test_square_feet[1]:.3e}", "Not Normal"])
table.add_row(["D'Agostino's K² Test Price", f"{dagostino_test_price.statistic:.3f}", f"{dagostino_test_price.pvalue:.3e}", "Not Normal"])
table.add_row(["D'Agostino's K² Test Square Feet", f"{dagostino_test_square_feet.statistic:.3f}", f"{dagostino_test_square_feet.pvalue:.3e}", "Not Normal"])

table.align = "r"
table.title="Normality Test Results for Price and Square Feet"
table

# Heatmap & Pearson correlation coefficient matrix
corr_matrix = cleaned_data.corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 9))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Heatmap of Variable Correlations')
plt.show()

correlation_data = cleaned_data[['bathrooms', 'bedrooms', 'price', 'square_feet']]

# Calculate the Pearson correlation matrix
correlation_matrix = correlation_data.corr()

# Plotting the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Pearson Correlation Coefficient Matrix')
plt.show()

# Plotting the scatter plot matrix (Pair Plot)
sns.pairplot(correlation_data)
plt.show()

# +
cleaned_data['price_numeric'] = pd.to_numeric(cleaned_data['price_display'].str.replace('[\$,]', '', regex=True), errors='coerce')
df_clean = cleaned_data.dropna(subset=['price_numeric'])

# Recalculating the average prices with the cleaned data
average_prices_clean = df_clean.groupby('cityname').agg({'price_numeric': 'mean', 'latitude': 'mean', 'longitude': 'mean'}).reset_index()

# Plotting again
plt.figure(figsize=(10, 6))
scatter = plt.scatter(average_prices_clean['longitude'], average_prices_clean['latitude'], c=average_prices_clean['price_numeric'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Average Rental Price ($)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Average Rental Prices by City (Cleaned Data)')
plt.grid(True)
plt.show()

#Line plot
cleaned_data['datetime'] = pd.to_datetime(cleaned_data['time'], unit='s')

# Group data by 'datetime' and calculate the average price
price_trends = cleaned_data.groupby(cleaned_data['datetime'].dt.date)['price'].mean().reset_index()
price_trends['datetime'] = pd.to_datetime(price_trends['datetime'])  # Ensure datetime format for plotting

plt.figure(figsize=(12, 6))
sns.lineplot(data=price_trends, x='datetime', y='price')
plt.title('Average Price Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Average Price')
plt.grid(True)
plt.show()

average_price_per_state = cleaned_data.groupby('state')['price'].mean().reset_index()

# Sorting to get the top 20 states with the highest average prices
top_20_states = average_price_per_state.sort_values(by='price', ascending=False).head(20)

# Plotting
plt.figure(figsize=(12, 8))
sns.barplot(data=top_20_states, x='state', y='price', palette='coolwarm')
plt.title('Average Listing Price per State')
plt.xlabel('State')
plt.ylabel('Average Price')
plt.xticks(rotation=45)  # Rotates state labels for better readability
plt.show()

#Violin Plot
df_square_feet = cleaned_data[(cleaned_data['square_feet'] > 100) & (cleaned_data['square_feet'] < 5000)]

# Creating the violin plot
plt.figure(figsize=(12, 6))
sns.violinplot(x=df_square_feet['square_feet'], color="skyblue")
plt.title('Violin Plot of Apartment Square Footage Distribution', fontsize=14)
plt.xlabel('Square Footage', fontsize=12)
plt.ylabel('Density', fontsize=12)

plt.show()

#Bar plot(grouped)
average_price_per_state = cleaned_data.groupby('state')['price'].mean().reset_index()

# Plotting
plt.figure(figsize=(12, 8))
sns.barplot(data=average_price_per_state, x='state', y='price', palette='coolwarm')
plt.title('Average Listing Price per State')
plt.xlabel('State')
plt.ylabel('Average Price')
plt.xticks(rotation=45)  # Rotates state labels for better readability
plt.show()

#Bar plot(Stacked)

#Count plot
plt.figure(figsize=(14, 10))  # Increased figure size for better visibility
sns.countplot(data=cleaned_data, x='category', palette='viridis', order=cleaned_data['category'].value_counts().index)
plt.title('Count of Listings by Category')
plt.xlabel('Category')
plt.ylabel('Number of Listings (log scale)')
plt.xticks(rotation=45)  # Rotates category labels for better readability
plt.yscale('log')  # Applying logarithmic scale to the y-axis
plt.show()

#Pie chart
pets_counts = cleaned_data['pets_allowed'].value_counts()

# Creating the pie chart
plt.figure(figsize=(10, 8))
plt.pie(pets_counts, labels=pets_counts.index, autopct='%1.1f%%', startangle=40, colors=plt.cm.Set3.colors)
plt.title('Distribution of Listings by Pet Policy')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

pet_policy_counts = cleaned_data['pets_allowed'].value_counts()
bedroom_counts = cleaned_data['bedrooms'].value_counts()
bathroom_counts = cleaned_data['bathrooms'].value_counts()

# Start subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Pie Chart of Pet Policies
axes[0].pie(pet_policy_counts, labels=pet_policy_counts.index, autopct='%1.1f%%', startangle=140)
axes[0].set_title('Pet Policy Distribution')

axes[1].pie(bedroom_counts, labels=bedroom_counts.index, autopct='%1.1f%%', startangle=50)
axes[1].set_title('Bedrooms Distribution')

axes[2].pie(bathroom_counts, labels=bathroom_counts.index, autopct='%1.1f%%', startangle=200)
axes[2].set_title('Bathrooms Distribution')

plt.tight_layout()
plt.show()

#Distribution Plot
plt.figure(figsize=(10, 6))
sns.distplot(cleaned_data['price'], bins=30, kde=True, rug=True, color='skyblue')
plt.title('Distribution of Prices')
plt.xlabel('Price')
plt.ylabel('Density')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
# Plot only non-zero, positive values to apply a logarithmic transformation
sns.distplot(cleaned_data[cleaned_data['square_feet'] > 0]['square_feet'], bins=30, kde=True, rug=True, color='green')
plt.title('Log-scaled Distribution of Property Sizes (Square Feet)')
plt.xlabel('Square Feet (log scale)')
plt.ylabel('Density')
plt.xscale('log')  # Apply logarithmic scale
plt.grid(True)
plt.show()

#Histogram with KDE
plt.figure(figsize=(10, 6))
sns.histplot(cleaned_data['price'], kde=True, color='blue')
plt.title('Histogram of Prices with KDE')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")

# KDE plot with fill and transparency settings
ax = sns.kdeplot(cleaned_data['price'], fill=True, color="blue", alpha=0.6, linewidth=3)
ax.set_title('KDE Plot for Apartment Prices with Fill and Alpha')
ax.set_xlabel('Price ($)')
ax.set_ylabel('Density')

plt.show()

# Create an lmplot with regression line and scatter points
plot_data = cleaned_data[['square_feet', 'price']]

plt.figure(figsize=(10, 6))
lm_plot = sns.lmplot(x='square_feet', y='price', data=plot_data, height=6, aspect=1.5, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
lm_plot.set(title='Regression Plot of Price vs. Square Feet', xlabel='Square Feet', ylabel='Price ($)')

plt.show()

# Create a boxen plot
boxen_data = cleaned_data[['bedrooms', 'price']]

plt.figure(figsize=(12, 8))
boxen_plot = sns.boxenplot(x='bedrooms', y='price', data=boxen_data)
boxen_plot.set_title('Price Distribution Across Different Bedroom Counts')
boxen_plot.set_xlabel('Number of Bedrooms')
boxen_plot.set_ylabel('Price ($)')

plt.show()

# Prepare data for the area plot by sorting and calculating the cumulative distribution
area_data = cleaned_data['price'].sort_values()
cumulative_price = np.cumsum(area_data) / np.sum(area_data)

# Plotting the cumulative distribution of prices
plt.figure(figsize=(12, 8))
plt.fill_between(area_data, cumulative_price, color="skyblue", alpha=0.4)
plt.title('Cumulative Distribution of Apartment Prices')
plt.xlabel('Price ($)')
plt.ylabel('Cumulative Distribution')

plt.show()

# Create a joint plot with both KDE and scatter plots for 'square_feet' and 'price'
joint_plot = sns.jointplot(x='square_feet', y='price', data=plot_data, kind="scatter", height=8, ratio=4, space=0.5)
joint_plot.plot_joint(sns.kdeplot, color='r', zorder=0, levels=6)
joint_plot.set_axis_labels('Square Feet', 'Price ($)')
joint_plot.fig.suptitle('Joint Plot of Price and Square Feet with KDE', y=1.03)

plt.show()

# Create a rug plot for the 'price' data
plt.figure(figsize=(12, 4))
sns.rugplot(cleaned_data['price'], height=0.5, color='blue')
plt.title('Rug Plot of Apartment Prices')
plt.xlabel('Price ($)')

plt.show()

from scipy.stats import gaussian_kde

data_points = np.vstack([cleaned_data['square_feet'], cleaned_data['price']])
density = gaussian_kde(data_points)(data_points)

# Create a 3D plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 10))
x = cleaned_data['square_feet']
y = cleaned_data['price']
z = density
ax.scatter(x, y, z, c=z, cmap='viridis')

# The 2D histogram of x and y
hist, xedges, yedges = np.histogram2d(x, y, bins=30, weights=z, density=True)

# Construct arrays with the dimensions for the bars.
xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
dx = dy = np.ones_like(hist) * (xedges[1] - xedges[0])
dz = hist

# Plot the density as a surface
X, Y = np.meshgrid((xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2, indexing="ij")

# Use plot_surface to plot X, Y, Z
surf = ax.plot_surface(X, Y, dz, cmap='viridis', rstride=1, cstride=1, alpha=0.5, antialiased=True)

ax.set_xlabel('Square Feet')
ax.set_ylabel('Price ($)')
ax.set_zlabel('Density')

plt.title('3D Contour Plot of Price and Square Feet Density')
plt.show()

# Calculate the correlation matrix from relevant numerical variables
correlation_data = cleaned_data[['price', 'square_feet', 'bedrooms', 'bathrooms']]
corr_matrix = correlation_data.corr()

# Create a cluster map based on the correlation matrix
cluster_map = sns.clustermap(corr_matrix, annot=True, cmap='coolwarm', figsize=(8, 8))
plt.title('Clustered Heatmap of Correlation Matrix')
plt.show()

# Create a hexbin plot for 'square_feet' and 'price'
plt.figure(figsize=(10, 8))
hexbin_plot = plt.hexbin(cleaned_data['square_feet'], cleaned_data['price'], gridsize=50, cmap='Blues', bins='log')
plt.colorbar(hexbin_plot, label='Log10(N)')
plt.title('Hexbin Plot of Price vs. Square Feet')
plt.xlabel('Square Feet')
plt.ylabel('Price ($)')

plt.show()

# Create a strip plot for 'bedrooms' and 'price'
plt.figure(figsize=(10, 8))
strip_plot = sns.stripplot(x='bedrooms', y='price', data=boxen_data, jitter=0.1, alpha=0.5)
strip_plot.set_title('Strip Plot of Price by Number of Bedrooms')
strip_plot.set_xlabel('Number of Bedrooms')
strip_plot.set_ylabel('Price ($)')

plt.show()

############# PLEASE WAIT, IT TAKES TIME --> UNCOMMENT IF YOU HAVE PATIENCE #####################
# # Create a swarm plot for 'bedrooms' and 'price'
# plt.figure(figsize=(12, 8))
# swarm_plot = sns.swarmplot(x='bedrooms', y='price', data=boxen_data, size=4)
# swarm_plot.set_title('Swarm Plot of Price by Number of Bedrooms')
# swarm_plot.set_xlabel('Number of Bedrooms')
# swarm_plot.set_ylabel('Price ($)')

# plt.show()

unique_categories = cleaned_data['category'].unique()

# Print the unique values
print(unique_categories)

bedroom_counts = cleaned_data['bedrooms'].value_counts()
bathroom_counts = cleaned_data['bathrooms'].value_counts()

# Sorting the counts in descending order
sorted_bedroom_counts = bedroom_counts.sort_values(ascending=False)
sorted_bathroom_counts = bathroom_counts.sort_values(ascending=False)

# Creating a figure with two subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 8))  # Adjust the figsize as needed

# Plotting number of bedrooms
sns.barplot(ax=axes[0], x=sorted_bedroom_counts.index, y=sorted_bedroom_counts, palette='coolwarm')
axes[0].set_title('Number of Listings by Number of Bedrooms', fontsize=14)
axes[0].set_xlabel('Number of Bedrooms', fontsize=12)
axes[0].set_ylabel('Number of Listings', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', linestyle='--')

# Plotting number of bathrooms
sns.barplot(ax=axes[1], x=sorted_bathroom_counts.index, y=sorted_bathroom_counts, palette='coolwarm')
axes[1].set_title('Number of Listings by Number of Bathrooms', fontsize=14)
axes[1].set_xlabel('Number of Bathrooms', fontsize=12)
axes[1].set_ylabel('Number of Listings', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', linestyle='--')

# Displaying the plots
plt.tight_layout()
plt.show()

cleaned_data['bedrooms_numeric'] = pd.to_numeric(cleaned_data['bedrooms'], errors='coerce')
cleaned_data['bathrooms_numeric'] = pd.to_numeric(cleaned_data['bathrooms'], errors='coerce')

# Creating a figure with two subplots
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(28, 8))

# Scatter plot for bathrooms
sns.scatterplot(ax=ax[0], x='bathrooms_numeric', y='price_numeric', data=cleaned_data, color='dodgerblue', alpha=0.6)
ax[0].set_title('Impact of Bathrooms on Rental Prices', fontsize=14)
ax[0].set_xlabel('Number of Bathrooms', fontsize=12)
ax[0].set_ylabel('Rental Price ($)', fontsize=12)
ax[0].grid(True)

# Scatter plot for bedrooms
sns.scatterplot(ax=ax[1], x='bedrooms_numeric', y='price_numeric', data=cleaned_data, color='mediumseagreen', alpha=0.6)
ax[1].set_title('Impact of Bedrooms on Rental Prices', fontsize=14)
ax[1].set_xlabel('Number of Bedrooms', fontsize=12)
ax[1].set_ylabel('Rental Price ($)', fontsize=12)
ax[1].grid(True)

# Adjust layout for better fit and display the plots
plt.tight_layout()
plt.show()

filtered_data = cleaned_data[(cleaned_data['square_feet'] > 0) & (cleaned_data['price'] > 0)]

plt.figure(figsize=(12, 6))

# Plot for square_feet
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
sns.distplot(filtered_data['square_feet'], bins=30, kde=True, rug=True, color='green')
plt.title('Log-scaled Distribution of Property Sizes (Square Feet)')
plt.xlabel('Square Feet (log scale)')
plt.ylabel('Density')
plt.xscale('log')  # Apply logarithmic scale
plt.grid(True)

# Plot for price
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
sns.distplot(filtered_data['price'], bins=30, kde=True, rug=True, color='blue')
plt.title('Log-scaled Distribution of Prices')
plt.xlabel('Price (log scale)')
plt.ylabel('Density')
plt.xscale('log')  # Apply logarithmic scale
plt.grid(True)

plt.tight_layout()
plt.show()

############# PLEASE WAIT, IT TAKES TIME #####################
filtered_data = cleaned_data[(cleaned_data['bathrooms'] > 0) &
                             (cleaned_data['bedrooms'] > 0) & 
                             (cleaned_data['price'] > 0) & 
                             (cleaned_data['square_feet'] > 0)]

# Multivariate KDE for bathrooms and bedrooms
plt.figure(figsize=(8, 6))
sns.kdeplot(data=filtered_data, x='bathrooms', y='bedrooms', cmap="Reds", fill=True, bw_adjust=0.5)
plt.title('Multivariate KDE of Bathrooms and Bedrooms')
plt.show()

############# PLEASE WAIT, IT TAKES TIME #####################
# Multivariate KDE for price and square_feet
plt.figure(figsize=(8, 6))
sns.kdeplot(data=filtered_data, x='price', y='square_feet', cmap="Blues", fill=True, bw_adjust=0.5)
plt.title('Multivariate KDE of Price and Square Feet')
plt.show()
