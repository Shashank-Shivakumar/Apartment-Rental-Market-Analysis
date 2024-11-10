# 🏠 US Rental Market Analysis and Visualization Dashboard

A comprehensive data analysis project focused on exploring the dynamics of the US apartment rental market using Python and Dash. The project analyzes rental listings to uncover insights into pricing trends, property features, and geographical influences.

## 🚀 Project Overview
This project utilizes a dataset of approximately 100,000 apartment listings across the United States, sourced from the University of California, Irvine (UCI). The goal is to understand the factors influencing rental prices, explore geographical variations, and analyze the impact of amenities on rental costs.

## 📊 Key Highlights

### 1. 🛠️ Data Preprocessing & Feature Engineering
- Cleaned and processed raw data, handling missing values and detecting outliers.
- Performed feature engineering, including calculating price per square foot and categorizing listings into luxury vs. budget segments.

### 2. 🔍 Exploratory Data Analysis
- Created static visualizations such as heatmaps, correlation matrices, regression plots, and KDEs.
- Conducted Principal Component Analysis (PCA) to reduce dimensionality while retaining significant variance.

### 3. 🌐 Interactive Dashboard
- Developed using Python’s Dash framework to provide real-time, interactive visualizations.
- Key features:
  - **Property Map**: Visualize rental listings by location.
  - **Price-to-Space Analysis**: Analyze the relationship between property size and rental price.
  - **Average Price Trends**: Explore rental price trends over time.
  - **State-wise Dashboard**: Drill down into property listings by state.
  - **Advanced Filters**: Filter by state, price range, pet policies, and more.
- **Live Dashboard**: [Access the dashboard here](https://dashapp-5jocwpusma-ue.a.run.app/)

### 4. 📦 Deployment
- The application is containerized using Docker and deployed on Google Cloud Platform (GCP) for scalability and global accessibility.

## 🗃️ Dataset Details
- **Source**: University of California, Irvine (UCI)
- **Size**: ~100,000 listings with 24 attributes (e.g., price, amenities, bedrooms, bathrooms, square footage, and location).
- **Dependent Variable**: `price_display` (Rental price)
- **Independent Variables**: Property features such as `bedrooms`, `bathrooms`, `square_feet`, and `amenities`.

## 🛠️ Technologies Used
- **Python**: Data preprocessing, analysis, and dashboard development.
- **Dash & Plotly**: Interactive data visualizations.
- **Docker**: Containerization for deployment.
- **Google Cloud Platform**: Application hosting.

## 📈 Insights and Findings
- Rental prices are significantly influenced by geographical location, with coastal cities having higher rental rates.
- The number of bedrooms and bathrooms correlates with both the size and cost of apartments.
- Affordable rental listings are more common, with a sharp decrease in availability as prices increase.

## 📝 Future Enhancements
- Integrate machine learning models to predict rental prices based on historical data.
- Expand the dashboard to include neighborhood-level analysis.
- Add additional filtering options for more granular insights.

Feel free to clone, fork, and contribute to this project. Your feedback and suggestions are welcome!
