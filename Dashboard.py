import os
import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import requests
import dash_bootstrap_components as dbc
from datetime import date
import base64
from plotly.subplots import make_subplots
from dash.exceptions import PreventUpdate

# Load data and clean it up
file_path = os.path.join(os.getcwd(), "apartments_data.csv")

image1 = '../assets/rental_image.png'
image_path = base64.b64encode(open(image1, 'rb').read()).decode('ascii')

image2 = '../assets/flag_img.png'
image_path2 = base64.b64encode(open(image2, 'rb').read()).decode('ascii')

try:
    df = pd.read_csv(file_path, sep=';', encoding='ISO-8859-1')
except UnicodeDecodeError:
    df = pd.read_csv(file_path, sep=';', encoding='unicode_escape')

if 'longitude' not in df.columns or 'latitude' not in df.columns:
    raise ValueError("The dataframe does not contain 'longitude' and 'latitude' columns")

df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df = df.dropna(subset=['longitude', 'latitude'])
df = df[(df['longitude'].between(-180, 180)) & (df['latitude'].between(-90, 90))]
df['datetime'] = pd.to_datetime(df['time'], unit='s')

url = "https://gist.githubusercontent.com/mshafrir/2646763/raw/8b0dbb93521f5d6889502305335104218454c2bf/states_hash.json"
response = requests.get(url)
states_json = response.json()

df['state_name'] = df['state'].map(states_json)
state_counts = df.groupby('state_name').size().reset_index(name='count')

geojson_url = 'https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json'
us_states = requests.get(geojson_url).json()

df = df.dropna(subset=['state_name'])

df['state_name'] = df['state_name'].astype(str)

unique_states = sorted(df['state_name'].unique())

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP,'https://use.fontawesome.com/releases/v5.8.1/css/all.css'])

filters_card = dbc.Card(
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Label("State:", htmlFor="states-dropdown"),
                dcc.Dropdown(
                    id='states-dropdown',
                    options=[{'label': state, 'value': state} for state in unique_states],
                    placeholder="Select a state",
                    clearable=True
                )
            ], width=4),
            dbc.Col([
                html.Label("Pets Allowed:", htmlFor="pets-radio"),
                dcc.RadioItems(
                    id='pets-radio',
                    options=[
                        {'label': 'Yes', 'value': 'Yes'},
                        {'label': 'No', 'value': 'None'}
                    ],
                    value='Yes',
                    labelStyle={'display': 'inline-block'}
                )
            ], width=4),
            dbc.Col([
                html.Label("Records:", htmlFor="num-records-input"),
                dbc.Input(
                    type='number',
                    id='num-records-input',
                    value=20,
                    min=1,
                    placeholder="Number of records",
                )
            ], width=4),
        ]),
        dbc.Row([
            dbc.Col([
                html.Label("Price Range:", htmlFor="price-range-slider"),
                dcc.RangeSlider(
                    id='price-range-slider',
                    min=df['price'].min(),
                    max=df['price'].max(),
                    step=100,
                    value=[df['price'].min(), df['price'].max()],
                    marks={int(x): {'label': str(int(x))} for x in
                           np.linspace(df['price'].min(), df['price'].max(), num=5)}
                )
            ], width=12)
        ]),
        dbc.Row([
            dbc.Col([
                html.Label("Date Range:", htmlFor="date-range-picker"),
                dcc.DatePickerRange(
                    id='date-range-picker',
                    min_date_allowed=date(2019, 1, 1),
                    max_date_allowed=date(2020, 1, 1),
                    start_date=date(2019, 1, 1),
                    end_date=date(2020, 1, 1)
                )
            ], width=12)
        ]),
        dbc.Row([
            dbc.Col(
                html.Button('Submit', id='submit-button', n_clicks=0, className='me-1 btn btn-primary'),
                width={'size': 2, 'offset': 5}
            ),
            dbc.Col(
                html.Button('Download', id='download-button', n_clicks=0, className='btn btn-secondary'),
                width=2
            )
        ])
    ]),
    className="mb-4"
)

table_card = dbc.Card(
    dbc.CardBody([
        dcc.Loading(
            id="loading-1",
            type="default",
            children=html.Div(id='table-container')
        )
    ]),
    className="mb-4"
)

display_columns = ['id', 'title', 'bedrooms', 'bathrooms', 'price_display','square_feet', 'state_name','cityname', 'datetime']

feature_count = len(df.columns)

# Function to detect outliers
def detect_outliers_iqr(data):
    outliers = {}
    for column in data.select_dtypes(include=[np.number]).columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_data = data[(data[column] < Q1 - outlier_step) | (data[column] > Q3 + outlier_step)]
        if len(outlier_data) > 1:  # Change here to check for more than one outlier
            outliers[column] = len(outlier_data)
    return outliers

# Apply the function to the DataFrame
outliers = detect_outliers_iqr(df)

# Filter columns to only those with more than one outlier
columns_with_outliers = [col for col in df.columns if col in outliers]

# Outlier Summary Table
outlier_summary_table = dbc.Table.from_dataframe(
    pd.DataFrame(list(outliers.items()), columns=["Feature", "Outliers Count"]),
    striped=True,
    bordered=True,
    hover=True
)

# Create box plots for the numerical features with more than one outlier
num_plots = len(columns_with_outliers)
num_rows = (num_plots // 3) + (num_plots % 3 > 0)  # Determining the number of rows needed for subplots

fig = make_subplots(rows=num_rows, cols=3, subplot_titles=columns_with_outliers)

for index, column in enumerate(columns_with_outliers, start=1):
    fig.add_trace(
        go.Box(y=df[column], name=column),
        row=(index-1)//3+1,
        col=(index-1)%3+1
    )

# Adjust the layout based on the number of actual plots
if num_plots % 3:
    for i in range(3 - (num_plots % 3)):
        fig.update_yaxes(visible=False, row=num_rows, col=3-i)
        fig.update_xaxes(visible=False, row=num_rows, col=3-i)

fig.update_layout(height=400*num_rows, title_text="Outliers in Numerical Features", showlegend=False)
fig.update_layout(grid=dict(columns=3, rows=num_rows))

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
cleaned_df = df.copy()
for column in columns_to_clean:
    cleaned_df = remove_outliers_iqr(cleaned_df, column)

def generate_table_rows(dataframe):
    rows = []
    for column, dtype in dataframe.dtypes.items():
        row = html.Tr([html.Td(column), html.Td(str(dtype))])
        rows.append(row)
    return rows

tab_style={'fontSize': '20px','fontWeight': 'bold', 'color': 'black', 'borderBottom': '4px solid white'}
tab_selected_style={'fontSize': '20px', 'fontWeight': 'bold', 'color': 'blue', 'borderBottom': '4px solid blue'}

app.layout = html.Div([
    html.H1("US Rental Market Analysis Dashboard",
            style={
                'textAlign': 'center',
                'fontSize':'80px',
                'fontWeight':'900',
                'marginTop':'10px',
                'marginBottom':'20px',
                'backgroundImage': f'url("data:image/png;base64,{format(image_path2)}")',
                'backgroundSize': 'cover',  # Ensure it covers the entire space
                'backgroundRepeat': 'repeat',  # No repeating of the image
                'backgroundPosition': 'bottom',  # Center the image
                'color': 'white',  # Text color to ensure legibility
                'padding': '20px',  # Add some padding to space the text away from the edges
                '-webkit-background-clip': 'text',
                '-webkit-text-fill-color': 'transparent'
            }),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Introduction to the Project', value='tab-1',style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Statistics of the Project', value='tab-2',style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Property Map', value='tab-3',style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Price-to-Space Analysis', value='tab-4',style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Average Price Trends', value='tab-5',style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='State-wise Dashboard ', value='tab-6',style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Advanced Filters', value='tab-7',style=tab_style, selected_style=tab_selected_style),

    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.Img(src='data:image/png;base64,{}'.format(image_path), style={'width': '100%', 'height': '500px', 'object-fit': 'cover'}),
            dbc.Row([
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            html.H1(f"{len(df)}"),
                            html.P("The dataset observations"),
                        ]),
                        className="mb-4",
                        style={"textAlign": "center", 'borderRadius': '20px'}
                    )
                ], width=2),
                # New Card for number of features
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            html.H1(f"{feature_count}"),
                            html.P("Number of features"),
                        ]),
                        className="mb-4",
                        style={"textAlign": "center", 'borderRadius': '20px'}
                    )
                ], width=2),
                # About the Project Card
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            html.H2("About the Project", style={'textAlign': 'center'}),
                            html.P("""This project provides an in-depth analysis of the U.S. apartment rental market using a dataset of about 100,000 listings from the UCI Database. 
                            We explore a variety of factors including pricing, amenities, location, and market trends, aiming to uncover the intricate dynamics that influence rental prices and availability across the country. 
                            By employing statistical methods and data visualization techniques, the project offers valuable insights for renters, real estate professionals, and policymakers, highlighting how different variables affect the rental landscape.
                            This interactive dashboard allows users to explore data further, enhancing understanding through dynamic filters and real-time analysis.""",
                                   style={'margin': '20px'}),
                        ]),
                        className="mb-4",
                        style={'borderRadius': '20px'}
                    )
                ], width=8),
                # Features and Data Types Card
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            html.H2("Features and Data Types", style={'textAlign': 'center'}),
                            html.Div(
                                dbc.Table(
                                    [
                                        html.Thead(html.Tr([html.Th("Feature"), html.Th("Data Type")])),
                                        html.Tbody(generate_table_rows(df))
                                    ],
                                    bordered=True,  # Add borders around the cells
                                    hover=True,  # Highlight rows on mouse-over
                                    responsive=False,  # Set to False to manage scroll inside the div instead
                                    striped=True,  # Zebra-striping on the table rows
                                    className="table"
                                ),
                                style={'overflowY': 'auto', 'height': '400px'}
                            )
                        ]),
                        className="mb-4",
                        style={'borderRadius': '20px'}
                    )
                ], width=12)
            ], style={
                'marginTop': '20px',
                'marginBottom': '5px',
                'marginLeft': '3%',
                'marginRight': '3%',
                'background': '#d3d3d3',
                'padding': '20px',
                'boxShadow': '0px 0px 15px 2px black'
            }),
        ])

    elif tab == 'tab-2':
        return html.Div([
            html.H3('Statistics of the Project'),
            dbc.Row([
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("Categorical Variables", className="card-title"),
                            html.H5(f"{df.select_dtypes(include=['object']).shape[1]}"),
                        ]),
                        className="mb-3",
                    ),
                    width={"size": 3, "offset": 0},
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("Numerical Variables", className="card-title"),
                            html.H5(f"{df.select_dtypes(include=[np.number]).shape[1]}"),
                        ]),
                        className="mb-3",
                    ),
                    width={"size": 3, "offset": 0},
                ),
            ]),
            dbc.Row([
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("Descriptive Statistics", className="card-title"),
                            dbc.Table.from_dataframe(
                                df.describe(),  # Assuming df is your DataFrame
                                striped=True,
                                bordered=True,
                                hover=True,
                                index=True,  # Ensure that row labels are included
                                responsive=True  # Allow the table to be scrollable on smaller screens
                            ),
                        ]),
                        className="mb-3",
                    ),
                    width=12,
                ),
            ]),
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.H4("Outlier Summary", className="card-title"),
                        outlier_summary_table
                    ]),
                    width=12,
                ),
            ]),
            dbc.Row([
                dbc.Col(
                    dcc.Graph(figure=fig),
                    width=12,
                ),
            ]),
        ])

    elif tab == 'tab-3':
        return html.Div([
            html.H3("Category"),
            dcc.Dropdown(
                id='category-filter',
                options=[
                    {'label': 'Apartments', 'value': 'housing/rent/apartment'},
                    {'label': 'Homes', 'value': 'housing/rent/home'},
                    {'label': 'Short-term Rentals', 'value': 'housing/rent/short_term'},
                    {'label': 'Condos', 'value': 'housing/rent/condo'},
                    {'label': 'Other Housing', 'value': 'housing/rent/other'},
                    {'label': 'Commercial/Retail', 'value': 'housing/rent/commercial/retail'}
                ],
                value='housing/rent/home',
                clearable=False,
                searchable=True,
                multi=True,
                placeholder="Select a category",
                style={'margin': '10px', 'marginLeft': '0px', 'marginTop': '5px'}
            ),
            dcc.Graph(id='map-plot'),
        ])

    elif tab == 'tab-4':
        return html.Div([
            html.H3("Interactive Price vs Square Feet Chart"),
            dbc.Row([
                dbc.Col([
                    html.Label("Select number of bedrooms:"),
                    dcc.Dropdown(
                        id='bedroom-dropdown',
                        options=[],  # Initially empty; options will be populated by the callback
                        placeholder="Select number of bedrooms"
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Select number of bathrooms:"),
                    dcc.Dropdown(
                        id='bathroom-dropdown',
                        options=[],  # Initially empty; options will be populated by the callback
                        placeholder="Select number of bathrooms"
                    )
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col(
                    dcc.Graph(id='price-vs-square-feet-chart'),
                    width=12
                )
            ])
        ])

    elif tab == 'tab-5':
        return html.Div([
            html.H1("Interactive Average Price Trends"),

            # Dropdown to select the category to display
            html.Div([
                html.Label("Select Category:"),
                dcc.Dropdown(
                    id='category-selector',
                    options=[
                        {'label': 'Apartments', 'value': 'housing/rent/apartment'},
                        {'label': 'Homes', 'value': 'housing/rent/home'},
                        {'label': 'Short-term Rentals', 'value': 'housing/rent/short_term'},
                        {'label': 'Condos', 'value': 'housing/rent/condo'},
                        {'label': 'Other Housing', 'value': 'housing/rent/other'},
                        {'label': 'Commercial/Retail', 'value': 'housing/rent/commercial/retail'}
                    ],
                    value='housing/rent/home',
                    multi=True  # Allow selecting multiple categories
                ),
            ], style={'width': '48%', 'display': 'inline-block'}),

            # Dropdown to select the type of average
            html.Div([
                html.Label("Select Type of Average:"),
                dcc.Dropdown(
                    id='average-type-selector',
                    options=[
                        {'label': 'Mean', 'value': 'mean'},
                        {'label': 'Median', 'value': 'median'},
                        {'label': 'Rolling Average', 'value': 'rolling'}
                    ],
                    value='mean',
                    clearable=False
                ),
            ], style={'width': '48%', 'display': 'inline-block'}),

            dcc.Graph(id='price-trend-graph'),
        ])

    elif tab == 'tab-6':
        return html.Div([
            html.H3("Interactive Map"),
            dcc.Graph(
                id='state-choropleth',
                config={'scrollZoom': False, 'displayModeBar': False}
            ),
            html.Div(id='selected-data', style={'paddingTop': 35}),
            html.Div(id='stored-state', style={'display': 'none'}),  # Hidden div to store the state value
            html.Div(id='city-data', style={'paddingTop': 35}),  # Div to display city data
        ])

    elif tab == 'tab-7':
        return html.Div([
            html.H3("Advanced Search Filters"),
            filters_card,  # Your filter components arranged neatly
            table_card,  # The table to display results
            dcc.Download(id='download-data'),
            dcc.Store(id='stored-data')
        ])

# elif tab == 'tab-2':
    #     return html.Div([
    #         html.H3('Content for Tab 2'),
    #         # Content for Tab 2 goes here...
    #     ])

@app.callback(
    [Output('table-container', 'children'),
     Output('stored-data', 'data')],
    [Input('submit-button', 'n_clicks')],
    [State('states-dropdown', 'value'),
     State('pets-radio', 'value'),
     State('price-range-slider', 'value'),
     State('num-records-input', 'value'),
     State('date-range-picker', 'start_date'),
     State('date-range-picker', 'end_date')]
)
def update_table(n_clicks, selected_state, pets_allowed, price_range, num_records, start_date, end_date):
    if n_clicks > 0:
        print(cleaned_df.describe)
        filtered_df = cleaned_df[(cleaned_df['price'] >= price_range[0]) & (cleaned_df['price'] <= price_range[1])]
        filtered_df = filtered_df[(filtered_df['datetime'] >= start_date) & (filtered_df['datetime'] <= end_date)]

        if selected_state:
            filtered_df = filtered_df[cleaned_df['state_name'] == selected_state]

        if pets_allowed == 'Yes':
            filtered_df = filtered_df[filtered_df['pets_allowed'].isin(['Cats', 'Dogs', 'Cats,Dogs'])]
        elif pets_allowed == 'None':
            filtered_df = filtered_df[filtered_df['pets_allowed'] == 'None']

        filtered_df = filtered_df[display_columns]  # Select only specified columns
        filtered_df = filtered_df.head(num_records)

        # Generate table using Dash Bootstrap Components
        table = dbc.Table.from_dataframe(filtered_df, striped=True, bordered=True, hover=True)
        return table, filtered_df.to_dict('records')
    return None, None

@app.callback(
    Output('download-data', 'data'),
    Input('download-button', 'n_clicks'),
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def generate_csv(n_clicks, stored_data):
    if stored_data is not None and n_clicks:
        # Convert the stored data back into a DataFrame
        df_to_download = pd.DataFrame(stored_data)
        # Select only the columns specified
        df_to_download = df_to_download[display_columns]
        return dcc.send_data_frame(df_to_download.to_csv, filename="filtered_data.csv")
    return None

@app.callback(
    [Output('state-choropleth', 'figure'),
     Output('selected-data', 'children'),
     Output('stored-state', 'children'),
     Output('city-data', 'children')],
    [Input('state-choropleth', 'clickData')],
    [State('stored-state', 'children')]
)
def update_choropleth_and_display_click_data(clickData, stored_state):
    if clickData:
        state_full_name = clickData['points'][0]['location']
        prop_count = state_counts[state_counts['state_name'] == state_full_name]['count'].iloc[0]
        selected_data_text = f"{state_full_name}: {prop_count} properties"

        # Filter dataframe for selected state and count properties by city
        state_df = cleaned_df[cleaned_df['state_name'] == state_full_name]
        city_counts = state_df.groupby('cityname').size().reset_index(name='count').sort_values(by='count',ascending=False).head(5)

        median_rental_price = state_df['price'].median()

        state_df['price_per_sqft'] = state_df['price'] / state_df['square_feet']
        avg_price_per_sqft = state_df['price_per_sqft'].mean()

        city_avg_values = state_df.groupby('cityname')['price'].mean().reset_index()
        highest_valued_city = city_avg_values.loc[city_avg_values['price'].idxmax(), 'cityname']

        #Scatter plot
        fig_scatter = px.scatter(
            state_df,
            x='square_feet',
            y='price',
            size='square_feet',  # Use size of the property for bubble size; change if needed
            title='Price Range by Property Size',
            hover_name=state_df.index  # Assuming index holds unique identifiers for properties
        )

        # Update layout if needed
        fig_scatter.update_layout(
            xaxis_title='Property Size (Square Feet)',
            yaxis_title='Price',
            margin=dict(l=40, r=40, t=40, b=40),
            height=280
        )

        # donut chart
        pets_count = state_df['pets_allowed'].value_counts().reset_index()
        pets_count.columns = ['Pets', 'Count']

        row_height = 30  # Estimated height per row in pixels
        header_height = 30  # Estimated height of the header in pixels
        num_rows = 8  # The number of rows in your table, adjust as needed
        estimated_table_height = (num_rows * row_height) + header_height

        fig_donut = px.pie(pets_count, values='Count', names='Pets', hole=0.3)
        fig_donut.update_traces(textinfo='percent+label',pull=[0, 0, 0, 0.2])
        fig_donut.update_layout(
            showlegend=False,
            height=estimated_table_height,
            margin=dict(l=0, r=0, t=0, b=0)  # Reduce the margin as much as possible
        )

        # Bar-graph bedroom
        bedroom_counts = state_df['bedrooms'].value_counts().sort_index().reset_index()
        bedroom_counts.columns = ['Number of Bedrooms', 'Count']

        # Create a horizontal bar chart
        fig_bar = px.bar(
            bedroom_counts,
            y='Number of Bedrooms',  # Correct column name for y-axis
            x='Count',
            text='Count',
            title="Distribution of Number of Bedroom Properties",
            orientation='h'  # This makes the bar chart horizontal
        )

        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(
            yaxis_title="Number of Bedrooms",
            xaxis_title="Number of Properties",
            margin=dict(l=40, r=40, t=40, b=40),
            height=300  # Adjust the height to fit all bars
        )

        # Bar-graph bathroom
        bathroom_counts = state_df['bathrooms'].value_counts().sort_index().reset_index()
        bathroom_counts.columns = ['Number of Bathrooms', 'Count']

        # Create a horizontal bar chart
        fig_bar_new = px.bar(
            bathroom_counts,
            y='Number of Bathrooms',  # Correct column name for y-axis
            x='Count',
            text='Count',
            title="Distribution of Number of Bathroom Properties",
            orientation='h'  # This makes the bar chart horizontal
        )

        fig_bar_new.update_traces(textposition='outside',marker_color='indianred')
        fig_bar_new.update_layout(
            yaxis_title="Number of Bathrooms",
            xaxis_title="Number of Properties",
            margin=dict(l=40, r=40, t=40, b=40),
            height=300  # Adjust the height to fit all bars
        )

        #Line chart
        state_df['datetime'] = pd.to_datetime(state_df['time'], unit='s')
        price_trends = state_df.groupby(state_df['datetime'].dt.date)['price'].mean().reset_index()
        price_trends['datetime'] = pd.to_datetime(price_trends['datetime'])  # Ensure datetime format for plotting

        # Create a line chart with Plotly Express
        fig_line = px.line(
            price_trends,
            x='datetime',
            y='price',
            title='Average Price Trend Over Time'
        )

        # Update layout if needed
        fig_line.update_layout(
            xaxis_title='Date',
            yaxis_title='Average Price',
            margin=dict(l=40, r=40, t=40, b=20),
            height=350
        )

        # Create a neat table for city data
        table_header = [html.Thead(html.Tr([html.Th("City"), html.Th("No. of Properties")]))]
        table_body = [html.Tbody([html.Tr([html.Td(city), html.Td(count)]) for city, count in city_counts.values])]
        city_table = dbc.Table(table_header + table_body, bordered=True, dark=False, hover=True, responsive=True,striped=True)

        date_range_card = dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Price Range by Property Size", className="h5"),
                            dbc.CardBody([dcc.Graph(figure=fig_scatter)]),
                        ],
                        className="mb-4 shadow-sm",
                        style={"width": "100%", 'borderRadius': '20px', 'height':"95%"}
                    ),
                    width=6
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Top Cities by Number of Properties", className="h5"),
                            dbc.CardBody([city_table]),
                        ],
                        className="mb-4 shadow-sm",
                        style={"width": "100%", 'borderRadius': '20px'}
                    ),
                    width=4
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Pets Allowed", className="h5"),
                            dbc.CardBody(dcc.Graph(figure=fig_donut,
                                                   # style={'height': '100%'}
                                                   )),
                        ],
                        className="mb-4 shadow-sm",
                        # Remove fixed height to allow content to dictate the card size
                        style={"width": "100%", 'borderRadius': '20px'}
                    ),
                    width=2  # Adjust width if necessary to fit your layout
                ),
            ],
            style={'marginTop': '20px'}
        )

        # Cards for the Key Metrics
        key_metric_cards = dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader([
                                html.I(className="fa fa-home", style={'marginRight': '10px'}),  # Add the FontAwesome home icon
                                f"{state_full_name} Real Estate Metrics"
                            ],
                            className="h1",
                            style={'color': 'white','fontSize':'50px'}),
                        ],
                        className="mb-4 shadow-sm",
                        style={"background": "#4e4eff",'borderRadius': '20px','padding':'25px','textAlign':'center'}
                    ),
                    width=4
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardBody([html.H1(f"{prop_count}")],
                                         style={'fontSize': '50px', 'fontWeight': 'bold', 'textAlign': 'center'}),
                            dbc.CardBody([html.P("Properties in State")],
                                         style={'textAlign': 'center', 'marginTop': '-10px'})
                        ],
                        className="mb-4 shadow-sm",
                        style={'borderRadius': '20px', 'paddingLeft': '3%', 'paddingRight': '3%'}
                    ),
                    width=2
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardBody([html.H1(f"${median_rental_price}")],style={'fontSize':'50px','fontWeight':'bold','textAlign':'center'}),
                            dbc.CardBody([html.P("Median Rental Price")],style={'textAlign':'center','marginTop':'-10px'})
                        ],
                        className="mb-4 shadow-sm",
                        style={'borderRadius': '20px','paddingLeft':'3%','paddingRight':'3%'}
                    ),
                    width=2
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardBody([html.H1(f"${avg_price_per_sqft:.2f}")],
                                         style={'fontSize': '50px', 'fontWeight': 'bold', 'textAlign': 'center'}),
                            dbc.CardBody([html.P("Average Price Per Sqft")],
                                         style={'textAlign': 'center', 'marginTop': '-10px'})
                        ],
                        className="mb-4 shadow-sm",
                        style={'borderRadius': '20px', 'paddingLeft': '3%', 'paddingRight': '3%'}
                    ),
                    width=2
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardBody([html.H1(f"{highest_valued_city}")],style={'fontSize':'30px','fontWeight':'bold','textAlign':'center'}),
                            dbc.CardBody([html.P("Highest Valued City")],style={'textAlign':'center','marginTop':'-5px'})
                        ],
                        className="mb-4 shadow-sm",
                        style={'borderRadius': '20px','paddingLeft':'3%','paddingRight':'3%'}
                    ),
                    width=2
                )
                # ... (Two more cards for the other key metrics)
            ],
            style={"marginTop": "15px"}
        )

        # Card for the Pie Chart
        bar_chart_card1 = dbc.Card(
    [
                dbc.CardHeader("Distribution of Number of Bedroom Properties", className="h5"),
                dbc.CardBody([dcc.Graph(figure=fig_bar)]),
            ],
            className="mb-4 shadow-sm",
            style={"width": "100%", 'borderRadius': '20px'}
        )

        bar_chart_card2 = dbc.Card(
            [
                dbc.CardHeader("Distribution of Number of Bathroom Properties", className="h5"),
                dbc.CardBody([dcc.Graph(figure=fig_bar_new)]),
            ],
            className="mb-4 shadow-sm",
            style={"width": "100%", 'borderRadius': '20px'}
        )

        line_chart_card = dbc.Card(
            [
                dbc.CardHeader("Average Price Trend Over Time", className="h5"),
                dbc.CardBody([dcc.Graph(figure=fig_line)]),
            ],
            className="mb-4 shadow-sm",
            style={"width": "100%", 'borderRadius': '20px'}
        )


        # Define the card div to hold both cards side by side
        dashboard_div = dbc.Row(
            [
                dbc.Col(key_metric_cards, width=12),
                dbc.Col(date_range_card, width=12),
                dbc.Col(bar_chart_card1, width=6),
                dbc.Col(bar_chart_card2, width=6),
                dbc.Col(line_chart_card, width=12),
                # ... (Other cards positioned accordingly)
            ],
            style={
                'marginTop': '5px',
                'marginBottom': '5px',
                'marginLeft': '3%',
                'marginRight': '3%',
                'background': '#d3d3d3',
                # 'borderRadius': '25px',
                'boxShadow': 'black 0px 0px 15px 2px',
                'padding': '10px'
            },
            className='mt-3'
        )
    else:
        selected_data_text = "Click on a state"
        dashboard_div = html.Div()

    # Define the choropleth figure
    fig = px.choropleth(
        state_counts,
        geojson=us_states,
        locations='state_name',
        color='count',
        color_continuous_scale='Bluered_r',
        featureidkey="properties.name",
        projection="mercator"
    )

    # Adjust the map's geospatial layout
    fig.update_geos(
        visible=False,
        lonaxis_range=[-140, -66],  # These ranges are approximate for the continental US
        lataxis_range=[24, 50],  # Expanded latitude range
        center={"lat": 37.0902, "lon": -95.7129}  # Center of the US
    )

    # Update the layout to be more responsive and include Alaska
    fig.update_layout(
        autosize=True,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        geo=dict(
            bgcolor='rgba(255,255,255,0)',
            lakecolor='white',
            landcolor='#f5f5f5',
            subunitcolor='grey',
            countrycolor='grey',
            showlakes=True,
        )
    )

    # Update traces for hover info
    fig.update_traces(
        hoverinfo='location+z',
        marker_line_width=0,
    )

    # Customize the legend
    fig.update_layout(coloraxis_colorbar=dict(
        title="Property Count",
        thicknessmode="pixels", thickness=15,
        lenmode="pixels", len=300,
        yanchor="top", y=1,
        ticks="outside", ticksuffix=" properties",
    ))

    # To ensure that the map resizes correctly on different screens, set the width and height to None
    fig.update_layout(width=None, height=None)

    return fig, selected_data_text, state_full_name if clickData else stored_state, dashboard_div

@app.callback(
    Output('price-trend-graph', 'figure'),
    [Input('category-selector', 'value'),
     Input('average-type-selector', 'value')]
)
def update_graph(selected_categories, average_type):
    fig = go.Figure()

    for category in selected_categories:
        category_df = cleaned_df[cleaned_df['category'] == category]

        if average_type == 'mean':
            price_trend = category_df.groupby(category_df['datetime'].dt.date)['price'].mean()
        elif average_type == 'median':
            price_trend = category_df.groupby(category_df['datetime'].dt.date)['price'].median()
        elif average_type == 'rolling':
            price_trend = category_df.groupby(category_df['datetime'].dt.date)['price'].mean().rolling(window=7).mean()

        fig.add_trace(go.Scatter(
            x=price_trend.index,
            y=price_trend,
            mode='lines',
            name=f'{category} - {average_type}'
        ))

    fig.update_layout(
        title='Average Price Trends by Category',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='closest',
        height=600
    )

    return fig

@app.callback(
    [Output('bedroom-dropdown', 'options'),
     Output('bathroom-dropdown', 'options')],
    [Input('tabs', 'value')])
def populate_bed_bath_dropdowns(tab):
    if tab == 'tab-4':
        # Static options from 0 to 9 for both dropdowns
        options = [{'label': i, 'value': i} for i in range(10)]  # Generates options 0 through 9
        return options, options
    return [], []


@app.callback(
    Output('price-vs-square-feet-chart', 'figure'),
    [Input('bedroom-dropdown', 'value'),
     Input('bathroom-dropdown', 'value')],
    prevent_initial_call=True)
def update_bar_chart(selected_bedrooms, selected_bathrooms):
    if selected_bedrooms is None or selected_bathrooms is None:
        raise PreventUpdate
    filtered_df = cleaned_df[(cleaned_df['bedrooms'] == selected_bedrooms) & (cleaned_df['bathrooms'] == selected_bathrooms)]

    # Use histogram to manage bins more effectively
    fig = px.histogram(
        filtered_df,
        x='price',
        y='square_feet',
        title=f'Price vs Square Feet for {selected_bedrooms} Bedrooms and {selected_bathrooms} Bathrooms',
        labels={'price': 'Price', 'square_feet': 'Square Feet'},
        histfunc='avg',  # Aggregate by average square feet in each price bin
        barmode='group'
    )

    # Update layout to improve visual appeal
    fig.update_layout(
        xaxis_title='Price',
        yaxis_title='Square Feet',
        bargap=0.05,  # Smaller gap between bars
        # bargroupgap=0.1  # Space between groups of bars, if applicable
        height=600
    )

    return fig

@app.callback(
    Output('map-plot', 'figure'),
    [Input('category-filter', 'value')]
)
def update_map(selected_categories):
    if not isinstance(selected_categories, list):
        selected_categories = [selected_categories]

    # Filter the DataFrame based on the selected categories
    filtered_df = cleaned_df[cleaned_df['category'].isin(selected_categories)] if selected_categories else cleaned_df

    # Assign colors based on the unique categories in the filtered DataFrame
    colors = {category: px.colors.qualitative.Plotly[i] for i, category in enumerate(cleaned_df['category'].unique())}

    # Create the figure and add a scattermapbox trace for each category
    fig = go.Figure()

    for category in selected_categories:
        category_df = filtered_df[filtered_df['category'] == category]
        fig.add_trace(go.Scattermapbox(
            lon=category_df['longitude'],
            lat=category_df['latitude'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=5,
                color=colors[category],
            ),
            name=category,  # This will create a legend entry for each category
        ))

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=3,
        mapbox_center={
            "lat": filtered_df['latitude'].mean(),
            "lon": filtered_df['longitude'].mean()
        } if not filtered_df.empty else {"lat": 0, "lon": 0},
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        width=3200,
        height=1500,
        title='Property Locations'
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=False, port=8093, host='0.0.0.0')