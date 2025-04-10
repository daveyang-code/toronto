import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import json
import geopandas as gpd
from plotly.subplots import make_subplots


# Initialize the Dash app with a modern Bootstrap theme
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

app.title = "Toronto Neighbourhood Explorer"
app.config.suppress_callback_exceptions = True


# Load neighbourhoods from GeoJSON
def get_neighbourhoods_from_geojson():
    gdf = gpd.read_file("neighbourhoods.geojson")
    return sorted(gdf["AREA_NAME"].unique())


# Load data from Excel file
def load_toronto_data():
    # Load the Excel file
    df = pd.read_excel("toronto.xlsx", engine="openpyxl")

    # Rename columns to match our desired format
    column_mapping = {
        "Neighbourhood Name": "neighbourhood",
        "Population": "population",
        "Average Age": "avg_age",
        "Median Age": "median_age",
        "Median Income": "median_income",
        "Average Income": "avg_income",
        "Average household size": "avg_household_size",
        "Participation rate": "participation_rate",
        "Employment rate": "employment_rate",
        "Unemployment rate": "unemployment_rate",
        "Assault": "assault",
        "Auto Theft": "auto_theft",
        "Break and Enter": "break_enter",
        "Robbery": "robbery",
        "Theft Over": "theft_over",
        "Average Property Price": "housing_price",
    }

    # Rename columns
    df = df.rename(columns=column_mapping)

    # Calculate violent crime rate per 1000 people
    df["assault"] = (df["assault"] / df["population"]) * 1000
    df["auto_theft"] = (df["auto_theft"] / df["population"]) * 1000
    df["break_enter"] = (df["break_enter"] / df["population"]) * 1000
    df["robbery"] = (df["robbery"] / df["population"]) * 1000
    df["theft_over"] = (df["theft_over"] / df["population"]) * 1000
    df["total_crime"] = (
        df["assault"]
        + df["auto_theft"]
        + df["break_enter"]
        + df["robbery"]
        + df["theft_over"]
    )

    return df


# Get neighbourhood boundaries
neighbourhoods = get_neighbourhoods_from_geojson()

# Load real Toronto data
df = load_toronto_data()

# Create metric labels dictionary for consistent use across components
metric_labels = {
    "population": "Population",
    "avg_age": "Average Age",
    "median_age": "Median Age",
    "median_income": "Median Income ($)",
    "avg_income": "Average Income ($)",
    "avg_household_size": "Average Household Size",
    "participation_rate": "Participation Rate (%)",
    "employment_rate": "Employment Rate (%)",
    "unemployment_rate": "Unemployment Rate (%)",
    "assault": "Assault (per 1000)",
    "auto_theft": "Auto Theft (per 1000)",
    "break_enter": "Break & Enter (per 1000)",
    "robbery": "Robbery (per 1000)",
    "theft_over": "Theft Over $5000",
    "total_crime": "Total Crime (per 1000)",
    "housing_price": "Housing Price ($)",
}

# Define metric categories
metric_categories = {
    "Demographics": [
        "population",
        "median_age",
        "avg_age",
        "avg_household_size",
    ],
    "Economics": [
        "avg_income",
        "median_income",
        "housing_price",
        "unemployment_rate",
        "participation_rate",
        "employment_rate",
    ],
    "Crime": [
        "total_crime",
        "assault",
        "auto_theft",
        "break_enter",
        "robbery",
        "theft_over",
    ],
}

# Create metric dropdown options with categories
metric_options = []
for category, metrics in metric_categories.items():
    metric_options.append(
        {
            "label": category,
            "disabled": True,
            "value": f"category_{category}",
        }
    )
    for metric in metrics:
        if metric in metric_labels:
            metric_options.append({"label": metric_labels[metric], "value": metric})

app.layout = dbc.Container(
    [
        # Header section
        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        html.H1(
                            "Toronto Neighbourhood Explorer",
                            className="display-4 text-center my-4 text-primary",
                        ),
                        html.P(
                            "Interactive analysis of Toronto's neighbourhoods",
                            className="lead text-center mb-4",
                        ),
                    ],
                    className="p-4 bg-light rounded shadow-sm",
                )
            ),
            className="mb-4",
        ),
        # Main content section
        dbc.Row(
            [
                # Sidebar with filters
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                html.H4("Filters", className="text-primary mb-0"),
                                className="bg-light",
                            ),
                            dbc.CardBody(
                                [
                                    html.Label(
                                        "Analysis Metric:", className="fw-bold mb-2"
                                    ),
                                    dcc.Dropdown(
                                        id="metric-dropdown",
                                        options=metric_options,
                                        value="population",
                                        multi=False,
                                        className="mb-3",
                                    ),
                                    html.Label(
                                        "Neighbourhoods to Compare:",
                                        className="fw-bold mb-2",
                                    ),
                                    dcc.Dropdown(
                                        id="neighbourhood-dropdown",
                                        options=[
                                            {"label": "Select All", "value": "ALL"},
                                            *[
                                                {"label": n, "value": n}
                                                for n in sorted(
                                                    df["neighbourhood"].unique()
                                                )
                                            ],
                                        ],
                                        value=[],
                                        multi=True,
                                        className="mb-3",
                                    ),
                                ]
                            ),
                        ],
                        className="shadow-sm h-100",
                    ),
                    width=12,
                    lg=3,
                    className="mb-4",
                ),
                # Main visualization area
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    dbc.Tabs(
                                        [
                                            dbc.Tab(label="Map View", tab_id="map-tab"),
                                            dbc.Tab(
                                                label="Comparison", tab_id="bar-tab"
                                            ),
                                            dbc.Tab(
                                                label="Correlations",
                                                tab_id="scatter-tab",
                                            ),
                                            dbc.Tab(
                                                label="Crime Analysis",
                                                tab_id="crime-tab",
                                            ),
                                            dbc.Tab(
                                                label="Scatterplot Matrix",
                                                tab_id="matrix-tab",
                                            ),
                                        ],
                                        id="viz-tabs",
                                        active_tab="map-tab",
                                        className="card-header-tabs",
                                    ),
                                    className="bg-light overflow-hidden",
                                ),
                                dbc.CardBody(
                                    [
                                        html.Div(
                                            dcc.Graph(
                                                id="choropleth-map",
                                                config={"displayModeBar": False},
                                                style={"height": "800px"},
                                            ),
                                            id="map-container",
                                            className="vis-container",
                                        ),
                                        html.Div(
                                            dcc.Graph(
                                                id="bar-chart",
                                                config={"displayModeBar": False},
                                                style={"height": "800px"},
                                            ),
                                            id="bar-container",
                                            className="vis-container d-none",
                                        ),
                                        html.Div(
                                            dcc.Graph(
                                                id="scatter-chart",
                                                config={"displayModeBar": False},
                                                style={"height": "800px"},
                                            ),
                                            id="scatter-container",
                                            className="vis-container d-none",
                                        ),
                                        html.Div(
                                            dcc.Graph(
                                                id="crime-chart",
                                                config={"displayModeBar": False},
                                                style={"height": "800px"},
                                            ),
                                            id="crime-container",
                                            className="vis-container d-none",
                                        ),
                                        html.Div(
                                            dcc.Graph(
                                                id="scatterplot-matrix",
                                                config={"displayModeBar": False},
                                                style={
                                                    "height": "800px",
                                                    "width": "100%",
                                                },
                                            ),
                                            id="matrix-container",
                                            className="vis-container d-none",
                                        ),
                                    ]
                                ),
                            ],
                            className="shadow-sm h-100",
                        ),
                    ],
                    width=12,
                    lg=6,
                    className="mb-4",
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                html.H4(
                                    "Neighbourhood Details",
                                    className="text-primary mb-0",
                                ),
                                className="bg-light",
                            ),
                            dbc.CardBody(
                                html.Div(
                                    id="neighbourhood-stats",
                                    style={"maxHeight": "70vh", "overflowY": "auto"},
                                )
                            ),
                        ],
                        className="shadow-sm h-100",
                    ),
                    width=12,
                    lg=3,
                    className="mb-4",
                ),
            ],
        ),
        # Hidden div for sharing data between callbacks
        html.Div(id="selected-data", style={"display": "none"}),
    ],
    fluid=True,
    className="px-4 py-3",
)


# Tab switching callback
@callback(
    [
        Output("map-container", "className"),
        Output("bar-container", "className"),
        Output("scatter-container", "className"),
        Output("crime-container", "className"),
        Output("matrix-container", "className"),
    ],
    [Input("viz-tabs", "active_tab")],
)
def switch_tab(active_tab):
    map_class = "vis-container" if active_tab == "map-tab" else "vis-container d-none"
    bar_class = "vis-container" if active_tab == "bar-tab" else "vis-container d-none"
    scatter_class = (
        "vis-container" if active_tab == "scatter-tab" else "vis-container d-none"
    )
    crime_class = (
        "vis-container" if active_tab == "crime-tab" else "vis-container d-none"
    )
    matrix_class = (
        "vis-container" if active_tab == "matrix-tab" else "vis-container d-none"
    )
    return (map_class, bar_class, scatter_class, crime_class, matrix_class)


@app.callback(
    Output("neighbourhood-dropdown", "value"),
    [Input("neighbourhood-dropdown", "value")],
    [State("neighbourhood-dropdown", "options")],
)
def update_neighbourhoods(selected_values, available_options):
    all_neighbourhoods = [
        opt["value"] for opt in available_options if opt["value"] != "ALL"
    ]

    if "ALL" in selected_values:
        return all_neighbourhoods  # Return all when "ALL" is selected
    elif set(selected_values) == set(all_neighbourhoods):
        return all_neighbourhoods  # Already all selected
    return selected_values  # Normal selection


# Map callback
@callback(
    Output("choropleth-map", "figure"),
    [Input("metric-dropdown", "value")],
)
def update_choropleth(metric):
    # Load GeoJSON data
    toronto_geojson = load_geojson()

    # Create a lookup dictionary to easily match GeoJSON properties with dataframe
    neighbourhood_metrics = {}
    for _, row in df.iterrows():
        neighbourhood_metrics[row["neighbourhood"]] = row[metric]

    # Create a color scale based on the metric
    if metric in [
        "assault",
        "property_crime",
        "drug_crime",
        "public_disorder",
        "total_crime",
        "auto_theft",
        "break_enter",
        "robbery",
        "theft_over",
        "unemployment_rate",
    ]:
        color_scale = "RdYlGn_r"  # Red-Yellow-Green reversed (red=high, green=low)
    else:
        color_scale = "Blues"  # Higher values are generally better for other metrics

    # Add the metric data to a copy of the GeoJSON
    geojson_with_data = json.loads(json.dumps(toronto_geojson))
    for feature in geojson_with_data["features"]:
        neighbourhood_name = feature["properties"]["AREA_NAME"]
        if neighbourhood_name in neighbourhood_metrics:
            feature["properties"][metric] = neighbourhood_metrics[neighbourhood_name]

    # Create the choropleth map
    fig = px.choropleth_mapbox(
        df,
        geojson=geojson_with_data,
        locations="neighbourhood",
        featureidkey="properties.AREA_NAME",
        color=metric,
        color_continuous_scale=color_scale,
        mapbox_style="carto-positron",
        zoom=10,
        center={"lat": 43.7, "lon": -79.4},  # Toronto coordinates
        opacity=0.8,
        labels={metric: metric_labels[metric]},
        hover_data=["neighbourhood", metric],
    )

    # Update layout
    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        height=800,
        title={
            "text": f"{metric_labels[metric]} by Toronto Neighbourhood",
            "x": 0.5,
            "xanchor": "center",
        },
        title_font_size=18,
        coloraxis_colorbar={
            "title": metric_labels[metric],
            "thicknessmode": "pixels",
            "thickness": 20,
            "lenmode": "pixels",
            "len": 300,
        },
    )

    return fig


# Bar chart callback
@callback(
    Output("bar-chart", "figure"),
    [
        Input("neighbourhood-dropdown", "value"),
        Input("metric-dropdown", "value"),
    ],
)
def update_bar_chart(neighbourhoods, metric):
    filtered_df = df[df["neighbourhood"].isin(neighbourhoods)]

    # For crime and unemployment metrics, sort ascending (lower is better)
    ascending = metric in [
        "assault",
        "property_crime",
        "drug_crime",
        "public_disorder",
        "total_crime",
        "auto_theft",
        "break_enter",
        "robbery",
        "theft_over",
        "unemployment_rate",
    ]
    filtered_df = filtered_df.sort_values(by=metric, ascending=ascending)

    fig = px.bar(
        filtered_df,
        x="neighbourhood",
        y=metric,
        color=(
            metric
            if metric
            in [
                "assault",
                "property_crime",
                "drug_crime",
                "public_disorder",
                "total_crime",
                "auto_theft",
                "break_enter",
                "robbery",
                "theft_over",
                "unemployment_rate",
            ]
            else "neighbourhood"
        ),
        color_continuous_scale=(
            "RdYlGn_r"
            if metric
            in [
                "assault",
                "property_crime",
                "drug_crime",
                "public_disorder",
                "total_crime",
                "auto_theft",
                "break_enter",
                "robbery",
                "theft_over",
                "unemployment_rate",
            ]
            else None
        ),
        color_discrete_sequence=px.colors.qualitative.Plotly,
        height=800,
        text_auto=True,
        hover_data=["neighbourhood", metric, "population"],
    )

    fig.update_traces(
        texttemplate="%{value:,.0f}",
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>"
        + f"{metric_labels[metric]}: "
        + "%{y:,.0f}<br>",
    )

    fig.update_layout(
        title={
            "text": f"{metric_labels[metric]} by Neighbourhood",
            "x": 0.5,
            "xanchor": "center",
        },
        title_font_size=18,
        xaxis_title="",
        yaxis_title=metric_labels[metric],
        xaxis_tickangle=-45,
        showlegend=False,
        margin={"r": 20, "t": 60, "l": 20, "b": 60},
    )

    # Format y-axis based on metric type
    if metric in ["avg_income", "median_income", "housing_price"]:
        fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",")

    return fig


# Scatter chart callback
@callback(
    Output("scatter-chart", "figure"),
    [Input("neighbourhood-dropdown", "value")],
)
def update_scatter_chart(neighbourhoods):
    filtered_df = df[df["neighbourhood"].isin(neighbourhoods)]

    # Create 3D scatter plot with total_crime as size and population as z-axis
    fig = px.scatter_3d(
        filtered_df,
        x="avg_income",
        y="housing_price",
        z="population",  # Population as z-axis
        size="total_crime",  # Total crime as bubble size
        size_max=30,  # Reduced maximum size
        color="neighbourhood",
        color_discrete_sequence=px.colors.qualitative.Plotly,
        height=800,
        hover_name="neighbourhood",
        hover_data={
            "neighbourhood": False,
            "avg_income": True,
            "housing_price": True,
            "population": True,
            "total_crime": True,
        },
        labels={
            "avg_income": "Average Income",
            "housing_price": "Housing Price",
            "population": "Population",
            "total_crime": "Total Crime",
        },
    )

    # Calculate a dynamic size reference based on the data range
    crime_range = filtered_df["total_crime"].max() - filtered_df["total_crime"].min()
    sizeref = (
        8 * crime_range / (15**2) if crime_range > 0 else 1
    )  # Avoid division by zero

    # Improve hover template and marker appearance
    fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "Average Income: $%{x:,.0f}<br>"
            "Housing Price: $%{y:,.0f}<br>"
            "Population: %{z:,.0f}<br>"
            "Total Crime: %{marker.size:,.0f}<extra></extra>"
        ),
        marker=dict(
            sizemode="diameter",
            sizeref=sizeref,  # Dynamic size reference
            line=dict(width=0.5, color="DarkSlateGrey"),
            opacity=0.8,
        ),
    )

    # Update layout for 3D
    fig.update_layout(
        title={
            "text": "Income vs. Housing Price vs. Population<br>Bubble Size = Total Crime Rate",
            "x": 0.5,
            "xanchor": "center",
        },
        title_font_size=16,
        scene=dict(
            xaxis_title="Average Income ($)",
            yaxis_title="Housing Price ($)",
            zaxis_title="Population",
            xaxis=dict(tickprefix="$", tickformat=","),
            yaxis=dict(tickprefix="$", tickformat=","),
            zaxis=dict(tickformat=","),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
        legend_title="Neighbourhood",
        margin={"r": 20, "t": 80, "l": 20, "b": 40},
    )

    return fig


# Crime chart callback
@callback(
    Output("crime-chart", "figure"),
    [Input("neighbourhood-dropdown", "value")],
)
def update_crime_chart(neighbourhoods):
    if not neighbourhoods:
        # Return empty figure with message if no neighbourhoods selected
        return go.Figure().update_layout(
            title="Select neighborhoods to view crime data",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )

    filtered_df = df[df["neighbourhood"].isin(neighbourhoods)].sort_values(
        by="total_crime", ascending=False
    )

    # Define crime metrics
    crime_metrics = [
        "assault",
        "auto_theft",
        "break_enter",
        "robbery",
        "theft_over",
    ]

    # Create a stacked bar chart
    fig = go.Figure()

    # Add a trace for each crime type
    for metric in crime_metrics:
        fig.add_trace(
            go.Bar(
                x=filtered_df["neighbourhood"],
                y=filtered_df[metric],
                name=metric_labels.get(metric, metric),
                hovertemplate="<b>%{x}</b><br>"
                + "%{fullData.name}: %{y:.2f} per 1000<extra></extra>",
            )
        )

    # Update layout
    fig.update_layout(
        title={
            "text": "Crime Rate Comparison by Neighbourhood (per 1,000 people)",
            "x": 0.5,
            "xanchor": "center",
        },
        title_font_size=18,
        xaxis_title="",
        yaxis_title="Crime Rate (per 1,000 people)",
        xaxis_tickangle=-45,
        barmode="stack",
        legend_title="Crime Type",
        height=800,
        margin={
            "r": 20,
            "t": 60,
            "l": 20,
            "b": 120,
        },  # Increased bottom margin for rotated labels
    )

    return fig


# Scatterplot matrix callback
@callback(
    Output("scatterplot-matrix", "figure"),
    [Input("neighbourhood-dropdown", "value")],
)
def update_scatterplot_matrix(neighbourhoods):
    if not neighbourhoods:
        # Return empty figure with message if no neighbourhoods selected
        return go.Figure().update_layout(
            title="Select neighborhoods to view scatterplot matrix",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )

    # Filter data by selected neighbourhoods
    filtered_df = df[df["neighbourhood"].isin(neighbourhoods)]

    # Select metrics for the scatterplot matrix
    selected_metrics = [
        "avg_income",
        "median_age",
        "housing_price",
        "population",
        "unemployment_rate",
        "total_crime",
        "avg_household_size",
    ]

    # Create labels for the axes
    axis_labels = {
        "avg_income": "Income",
        "housing_price": "Housing",
        "population": "Population",
        "median_age": "Median Age",
        "unemployment_rate": "Unemployment",
        "total_crime": "Crime",
        "avg_household_size": "Household",
    }

    # Calculate the number of rows and columns for the grid
    n_metrics = len(selected_metrics)

    # Create the subplot grid
    fig = make_subplots(
        rows=n_metrics,
        cols=n_metrics,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
    )

    # Create a colorscale mapping for the neighbourhoods
    color_scale = px.colors.qualitative.Plotly
    color_map = {
        neighbourhood: color_scale[i % len(color_scale)]
        for i, neighbourhood in enumerate(filtered_df["neighbourhood"])
    }

    # Add traces for each combination of variables
    for i, y_metric in enumerate(selected_metrics):
        for j, x_metric in enumerate(selected_metrics):
            row, col = i + 1, j + 1

            # If it's the diagonal, show a histogram of the metric
            if i == j:
                for neighborhood in filtered_df["neighbourhood"].unique():
                    nbhd_data = filtered_df[
                        filtered_df["neighbourhood"] == neighborhood
                    ]
                    fig.add_trace(
                        go.Histogram(
                            x=nbhd_data[x_metric],
                            name=neighborhood,
                            marker_color=color_map[neighborhood],
                            showlegend=False,
                            opacity=0.7,
                        ),
                        row=row,
                        col=col,
                    )
            else:  # Add scatterplot for off-diagonal
                for neighborhood in filtered_df["neighbourhood"].unique():
                    nbhd_data = filtered_df[
                        filtered_df["neighbourhood"] == neighborhood
                    ]
                    fig.add_trace(
                        go.Scatter(
                            x=nbhd_data[x_metric],
                            y=nbhd_data[y_metric],
                            mode="markers",
                            name=neighborhood,
                            marker=dict(
                                color=color_map[neighborhood],
                                size=3,
                                opacity=0.5,
                                line=dict(width=1, color="DarkSlateGrey"),
                            ),
                            text=nbhd_data["neighbourhood"],
                            hovertemplate=(
                                "<b>%{text}</b><br>"
                                + f"{axis_labels[x_metric]}: "
                                + "%{x:,.0f}<br>"
                                + f"{axis_labels[y_metric]}: "
                                + "%{y:,.0f}<br>"
                            ),
                        ),
                        row=row,
                        col=col,
                    )

    # Update axis labels for the matrix border
    for i, metric in enumerate(selected_metrics):
        row, col = i + 1, i + 1
        # Add axis titles to the border cells
        fig.update_xaxes(title_text=axis_labels[metric], row=n_metrics, col=col)
        fig.update_yaxes(title_text=axis_labels[metric], row=row, col=1)

    # Update layout and formatting
    fig.update_layout(
        title="Scatterplot Matrix of Key Toronto Neighbourhood Metrics",
        height=700,
        autosize=True,
        showlegend=False,
        legend=dict(title="Neighbourhoods", orientation="h", y=-0.1),
        margin=dict(l=60, r=20, t=60, b=100),
    )

    # Format axes for specific types of data
    for i in range(n_metrics):
        for j in range(n_metrics):
            row, col = i + 1, j + 1
            x_metric = selected_metrics[j]
            y_metric = selected_metrics[i]

            # Apply currency formatting for income and housing price
            if x_metric in ["avg_income", "housing_price"]:
                fig.update_xaxes(
                    tickprefix="$",
                    tickformat=",.2s",
                    row=row,
                    col=col,
                )
            if y_metric in ["avg_income", "housing_price"]:
                fig.update_yaxes(
                    tickprefix="$",
                    tickformat=",.2s",
                    row=row,
                    col=col,
                )

            # Remove tick labels from interior cells to reduce clutter
            if 1 < row < n_metrics and 1 < col < n_metrics:
                fig.update_xaxes(showticklabels=False, row=row, col=col)
                fig.update_yaxes(showticklabels=False, row=row, col=col)

            # Add gridlines for better readability
            fig.update_xaxes(
                showgrid=True, gridwidth=0.5, gridcolor="lightgray", row=row, col=col
            )
            fig.update_yaxes(
                showgrid=True, gridwidth=0.5, gridcolor="lightgray", row=row, col=col
            )

            # Set axis ranges for scatterplots
            if i != j:  # Only for scatterplots, not histograms
                x_padding = (
                    filtered_df[x_metric].max() - filtered_df[x_metric].min()
                ) * 0.05
                y_padding = (
                    filtered_df[y_metric].max() - filtered_df[y_metric].min()
                ) * 0.05
                fig.update_xaxes(
                    range=[
                        filtered_df[x_metric].min() - x_padding,
                        filtered_df[x_metric].max() + x_padding,
                    ],
                    row=row,
                    col=col,
                )
                fig.update_yaxes(
                    range=[
                        filtered_df[y_metric].min() - y_padding,
                        filtered_df[y_metric].max() + y_padding,
                    ],
                    row=row,
                    col=col,
                )

    return fig


# Neighbourhood stats callback
@callback(
    Output("neighbourhood-stats", "children"),
    [
        Input("neighbourhood-dropdown", "value"),
        Input("metric-dropdown", "value"),  # Add metric dropdown as input
        Input("viz-tabs", "active_tab"),
    ],
)
def update_neighbourhood_stats(neighbourhoods, selected_metric, active_tab):
    if not neighbourhoods:
        return html.P("Select neighbourhoods to view statistics.")

    filtered_df = df[df["neighbourhood"].isin(neighbourhoods)]

    # Sort by the selected metric (default to population if not specified)
    sort_by = selected_metric if selected_metric in df.columns else "population"
    filtered_df = filtered_df.sort_values(by=sort_by, ascending=False)

    stat_cards = []
    for _, row in filtered_df.iterrows():
        # Format values for display
        population = f"{row['population']:,}"
        avg_income = f"${row['avg_income']:,.0f}"
        housing_price = f"${row['housing_price']:,.0f}"
        crime_rate = f"{row['total_crime']:.1f} per 1,000"
        crime_comparison = (
            "lower" if row["total_crime"] < df["total_crime"].mean() else "higher"
        )
        income_comparison = (
            "higher" if row["avg_income"] > df["avg_income"].mean() else "lower"
        )

        # Highlight the selected metric in the card
        def format_metric(value, metric_name, row):
            is_selected = metric_name == selected_metric
            return html.Span(
                [
                    html.Strong(f"{metric_labels.get(metric_name, metric_name)}: "),
                    html.Span(
                        value,
                        style={
                            "fontWeight": "bold",
                            "color": "#0d6efd" if is_selected else "inherit",
                        },
                    ),
                    (
                        html.Span(
                            " (sorted)" if is_selected else "",
                            className="text-primary small ms-1",
                        )
                        if is_selected
                        else None
                    ),
                ],
                className="mb-2 d-block",
            )

        stat_cards.append(
            dbc.Card(
                [
                    dbc.CardHeader(
                        html.H5(
                            row["neighbourhood"],
                            className="text-primary mb-0",
                        )
                    ),
                    dbc.CardBody(
                        [
                            html.Div(
                                [
                                    html.Strong("Population: "),
                                    html.Span(population),
                                ],
                                className="mb-2",
                            ),
                            html.Div(
                                [
                                    html.Strong("Average Income: "),
                                    html.Span(avg_income),
                                    html.Span(
                                        f" ({income_comparison} than city average)",
                                        className="text-muted ms-1 small",
                                    ),
                                ],
                                className="mb-2",
                            ),
                            html.Div(
                                [
                                    html.Strong("Average Housing Price: "),
                                    html.Span(housing_price),
                                ],
                                className="mb-2",
                            ),
                            html.Div(
                                [
                                    html.Strong("Crime Rate: "),
                                    html.Span(crime_rate),
                                    html.Span(
                                        f" ({crime_comparison} than city average)",
                                        className="text-muted ms-1 small",
                                    ),
                                ],
                                className="mb-2",
                            ),
                            html.Div(
                                [
                                    html.Strong("Median Age: "),
                                    html.Span(f"{row['median_age']:.1f} years"),
                                ],
                                className="mb-2",
                            ),
                            html.Div(
                                [
                                    html.Strong("Household Size: "),
                                    html.Span(
                                        f"{row['avg_household_size']:.2f} people"
                                    ),
                                ],
                                className="mb-2",
                            ),
                        ]
                    ),
                ],
                className="mb-3 shadow-sm",
            )
        )

    return html.Div(stat_cards)


# Helper function to load GeoJSON data
def load_geojson():
    with open("neighbourhoods.geojson", "r") as f:
        return json.load(f)


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
