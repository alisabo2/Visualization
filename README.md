# Environmental Waste Dashboard 🌍🗑️

This project is an interactive dashboard developed using Streamlit, which visualizes environmental waste data for cities in Israel. The dashboard fetches data from multiple public APIs and provides insights into waste types, infrastructure conditions, and public behaviors regarding waste management. It is equipped with various visualization tools, including Plotly and Folium, to present the data interactively and in real time.

## Features 🚀

- **City-Level Analysis**: Users can select a city to explore waste data, infrastructure conditions, and public behavior patterns.
- **Interactive Visualizations**:
  - Bar charts showing the top waste types by city.
  - Stacked bar plots for infrastructure conditions like sidewalks, fences, and vegetation.
  - Maps with circle markers indicating waste levels in specific locations.
  - Pie charts highlighting behavioral patterns across various cities.
- **Dynamic Maps**: Uses Folium to create maps with markers based on waste data and city infrastructure.
- **Right-to-Left Text Support**: For better readability of Hebrew text.

## Technologies 🛠️

- **Streamlit** for building the web interface.
- **Folium** and **Streamlit-Folium** for interactive maps.
- **Plotly** for creating beautiful and customizable graphs.
- **Pandas** for data manipulation and analysis.
- **Matplotlib** for color scaling and normalization of values.
- APIs from **Data.Gov.IL** to fetch real-time data on waste management in Israeli cities.

## Data Sources 📊

The data for this dashboard is fetched from the Data.Gov.IL platform through public APIs:

- **Bin Storage Data**: Information on waste bins in various cities.
- **Behaviors Data**: Public behaviors regarding waste management.
- **Infrastructure Conditions**: City infrastructure related to waste management (e.g., sidewalks, fences).
- **Dirt Information**: Data on waste types observed in public spaces.

## Setup Instructions 🔧

To run this project locally, follow these steps:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/your-username/environmental-waste-dashboard.git
    ```

2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit app**:

    ```bash
    streamlit run app.py
    ```

## How It Works ⚙️

The dashboard fetches data from APIs, processes it using Pandas, and displays various visualizations in real time. Here's a breakdown of the core functionality:

- `fetch_data_and_create_df(api_dict)`: Fetches data from multiple APIs and creates a dictionary of pandas DataFrames.
- `plot_top5_waste_types(city, measurer_type, k)`: Generates a bar chart showing the top 5 waste types in a selected city.
- `update_map(city, measurer_type_list)`: Updates a Folium map based on the selected city and waste type.
- `plot_infrastructure_condition(df, city)`: Plots the infrastructure conditions (sidewalks, fences, etc.) for a selected city using stacked bar charts.
- `plot_waste_levels_by_city(city)`: Shows the waste levels in bins for a selected city.

## Example Visualizations 📈

- **Top Waste Types by City**
  
- **Infrastructure Conditions**

- **Waste Levels Map**

## Future Work 🚧

- Add more city-level filters for detailed analysis.
- Improve user experience with additional interactivity.
- Enhance the performance of data fetching and visualization.

## Contributing 🤝

Feel free to open issues and submit pull requests if you'd like to contribute to the project.
