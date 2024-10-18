import streamlit as st
import pandas as pd
import re
import numpy as np
import requests
import branca.colormap as bcm
import folium
from matplotlib.colors import Normalize
from streamlit_folium import folium_static
import matplotlib
from matplotlib import cm
import plotly.graph_objects as go
import plotly.express as px

def fetch_data_and_create_df(api_dict):
    """
    Fetches data from APIs and creates pandas DataFrames.

    Parameters:
    - api_dict (dict): A dictionary where keys are table names and values are resource IDs.

    Returns:
    - df_dict (dict): A dictionary where keys are table names and values are pandas DataFrames.
    """
    df_dict = {}

    for table_name, resource_id in api_dict.items():
        url = 'https://data.gov.il/api/3/action/datastore_search'
        params = {'resource_id': resource_id}  # You can adjust the limit as needed
        response = requests.get(url, params=params)

        if response.status_code == 200:
            json_data = response.json()
            records = json_data['result']['records']
            df_dict[table_name] = pd.DataFrame(records)
        else:
            print(f'Failed to retrieve data from {table_name}: {response.status_code}')

    return df_dict

def plot_top_k_waste_types(df, waste_types, city, measurer_type, k, prefix):
    """
    :param df: The input DataFrame
    :param waste_types: The waste types to plot
    :param city: The selected city
    :param measurer_type: The type of measurement points
    :param k: Number of top waste types to plot
    :param prefix: Prefix for the plot title
    """
    # Filter data for the selected measurer type
    israel_filtered = df[df['סוג נקודת המדידה'].isin(measurer_type)]
    israel_avg_values = israel_filtered[waste_types].mean()

    if city != 'כל הארץ':
        # Filter data for the selected city and measurer type
        df_filtered = df[(df['יישוב'] == city) & (df['סוג נקודת המדידה'].isin(measurer_type))]
        # Calculate the average for each column
        avg_values = df_filtered[waste_types].mean()
    else:
        # Use the national averages for 'כל הארץ'
        avg_values = israel_avg_values

    # Get the top k waste types with the largest difference
    topk = avg_values.nlargest(k)

    # Sort the top k waste types and their corresponding labels
    sorted_topk = topk.sort_values(ascending=False)
    labels = sorted_topk.index.tolist()  # Get the labels in the sorted order

    # Prepare the average values for the sorted top k waste types
    sorted_avg_values = avg_values[sorted_topk.index]

    # Define colors for bars
    colors = {3: ['#D55E00',  # Red-Orange
                  '#F2A900',  # Bright Orange
                  '#5DADE2'   # Light Blue
                  ], 5: [
        '#D55E00',  # Red-Orange
        '#F2A900',  # Golden Yellow
        '#5DADE2',  # Light Blue
        '#0072B2',  # Dark Blue
        '#182139'   # Light Green
    ], 10: [
        '#D55E00',  # Red-Orange
        '#F2A900',  # Golden Yellow
        '#FFCC00',  # Bright Yellow
        '#A7D600',  # Bright Green
        '#5DADE2',  # Light Blue
        '#0072B2',  # Dark Blue
        '#4B0082',  # Indigo
        '#9933CC',  # Purple
        '#FF66B2',  # Light Pink
        '#FF3333'   # Coral Red
    ]}

    # Create the figure with Plotly graph_objects
    fig = go.Figure()

    # Add the bar chart for the city or national data, starting from zero
    fig.add_trace(go.Bar(
        x=labels,
        y=sorted_avg_values,  # Use the sorted average values
        marker_color=colors[k],
        showlegend=False,
        hovertemplate=(
                '<b>%{x}</b><br>' +  # Column label (waste type)
                'Average in ' + city + ': %{y:.2f}<br>' +  # City average
                'National Average: %{customdata[0]:.2f}<br>' +  # National average
                'Difference: %{customdata[1]:.2f}<extra></extra>'  # Difference between city and national
        ),
        customdata=np.stack((israel_avg_values[sorted_topk.index], sorted_avg_values - israel_avg_values[sorted_topk.index]), axis=-1)  # National average and difference
    ))

    # Add horizontal lines for national average for each waste type
    fig.add_trace(go.Scatter(
        x=labels,  # The x-position for each waste type
        y=israel_avg_values[sorted_topk.index],  # The y-values are the national averages
        mode='markers',
        name='ממוצע ארצי',
        showlegend=True,  # Show a legend for the national average markers
        marker=dict(
            color='black',  # Black color for the marker
            symbol='x',  # 'x' marker symbol
            size=10  # You can adjust the size as needed
        )
    ))

    # Update layout for Hebrew labels and set zero-based y-axis
    fig.update_layout(
        title={
            'text': f'{prefix} סוגי הפסולת המובילים',
            'x': 1.0,  # Align to the right
            'xanchor': 'right'
        },
        xaxis_title='סוג פסולת',
        yaxis_title='כמות ממוצעת',
        font=dict(family='Arial, sans-serif', size=14),
        xaxis_tickangle=-45,
        bargap=0.2,
        height=400,
        yaxis=dict(
            zeroline=True,  # Ensure y-axis starts from zero
            zerolinecolor='gray',
            rangemode='tozero'  # Ensure the y-axis includes zero
        ),
        legend_title='מקרא'
    )
    # Show the interactive Plotly plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def calculate_avg_coords(df):
    """
    Function to calculate average coordinates of all city_df coordinates
    :param df:
    :return:
    """

    latitudes = []
    longitudes = []

    for coords in df['נ.צ כתובת']:
        try:
            lat, lon = map(float, coords.split(','))
            latitudes.append(lat)
            longitudes.append(lon)
        except ValueError:
            continue
    return [sum(latitudes) / len(latitudes), sum(longitudes) / len(longitudes)] if latitudes else [0, 0]


def update_map(filtered_df, city, measurer_type_list, measurer_type, bin_storage_df):
    # Filter data for the selected city
    if city != 'כל הארץ':
        # Filter the data for the selected city
        city_df = filtered_df[(filtered_df['יישוב'] == city) & (filtered_df['סוג נקודת המדידה'].isin(measurer_type))]
    else:
        city_df = filtered_df[filtered_df['סוג נקודת המדידה'].isin(measurer_type)]

    # Recalculate average coordinates for the selected city
    if not city_df.empty and city != 'כל הארץ':
        avg_coords = calculate_avg_coords(city_df)
        zoom_start = 12
    else:
        avg_coords = [31.813, 35.163]  # Default coordinates if no data
        zoom_start = 10

    # Create a new map centered at the average coordinates of the selected city
    mymap = folium.Map(location=avg_coords, zoom_start=zoom_start)

    # Calculate min and max values across selected columns
    min_value = city_df[list(measurer_type_list)].min().min()
    max_value = city_df[list(measurer_type_list)].max().max()

    # Create a colormap using the existing color scale based on min-max values
    colormap = bcm.LinearColormap(
        colors=['darkblue', 'lightblue', 'orange', 'red'],  # Replace with your color scale
        vmin=min_value,
        vmax=max_value,
        caption='Average Value Legend'  # Caption for the color legend
    )
    # Add the colormap (legend) to the map
    colormap.add_to(mymap)

    # Add CircleMarkers to the map with color based on the average of the selected columns
    for index, row in city_df.iterrows():
        try:
            lat, lon = map(float, row['נ.צ כתובת'].split(','))

            # Calculate the average of the selected columns
            avg_value = row[list(measurer_type_list)].mean()

            # Get the color for the CircleMarker based on the average value
            color = get_color_scale(avg_value, min_value, max_value)

            # Create the popup content
            popup_content = f"""
                            <strong>כתובת/תיאור נקודת המדידה:</strong> {row['כתובת תיאור מיקום נקודת המדידה']}<br>
                            <strong>סוג נקודת המדידה:</strong> {row['סוג נקודת המדידה']}<br>
                        """
            # Look up the number of bins in bin_storage_df
            bin_row = bin_storage_df[bin_storage_df['נ.צ כתובת'] == row['נ.צ כתובת']]
            if not bin_row.empty:
                num_bins = bin_row['כמה פחים יש בנקודת המדידה'].sum()
                if type(num_bins)==np.float64:
                    popup_content += f"<strong>כמות פחים בנקודת המדידה:</strong> {num_bins:.0f}<br>"
                else:
                    popup_content += f"<strong>כמות פחים לא ידועה</strong><br>"
            else:
                popup_content += f"<strong>כמות פחים לא ידועה</strong><br>"

            # Add the selected measurement columns and their values to the popup
            amounts_list = []
            for measurer_type in measurer_type_list:
                if measurer_type != 'הכל':
                    amount = row[measurer_type]
                    amounts_list.append((measurer_type, amount))

            # Sort the list by amount in descending order and take the top 5
            top_amounts = sorted(amounts_list, key=lambda x: x[1], reverse=True)[:5]
            # Add the top 5 measurer_type and their amounts to the popup
            for measurer_type, amount in top_amounts:
                popup_content += f"<strong>כמות {measurer_type}:</strong> {amount:.0f}<br>"



            # Add a CircleMarker with the calculated color
            folium.CircleMarker(
                location=[lat, lon],
                radius=7,  # Size of the marker
                color=colormap(avg_value),  # Use the colormap
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_content, max_width=300)
            ).add_to(mymap)
        except ValueError:
            continue

    return mymap


def plot_infrastructure_condition(df, city, measurer_type):
    if city != 'כל הארץ':
        # Filter the data for the selected city
        df_filtered = df[(df['יישוב'] == city) & (df['point_type'].isin(measurer_type))]
    else:
        df_filtered = df[df['point_type'].isin(measurer_type)]

    # Define the columns to plot
    columns_to_plot = ['מדרכה', 'אבנישפה', 'גדרות', 'צמחייה']

    # Define the condition categories and their corresponding colors
    condition_categories = ['לא תקין (מוזנח)', 'סביר', 'תקין (מטופח)', 'לא רלוונטי']
    color_map = {
        'לא רלוונטי': '#b6cfc5',  # Light gray
        'לא תקין (מוזנח)': '#D55E00',  # Orange
        'סביר': '#FF8C00',  # Bright orange
        'תקין (מטופח)': '#0072B2'  # Dark blue
    }

    # Initialize a dictionary to hold counts for each condition category
    condition_counts_norm = {condition: [] for condition in condition_categories}
    condition_counts = {condition: [] for condition in condition_categories}
    total_counts = []  # To store the total counts for each column

    # Count the occurrences of each condition for each column
    for col in columns_to_plot:
        total = df_filtered[col].notnull().sum()  # Get the total count of non-null entries for the column
        total_counts.append(total)

        for condition in condition_categories:
            count = df_filtered[df_filtered[col] == condition].shape[0]
            condition_counts[condition].append(count / total if total > 0 else 0)  # Normalize the count
            condition_counts_norm[condition].append(count)  # Normalize the count

    # Create a figure for the horizontal diverging stacked bar chart
    fig = go.Figure()

    # Plot the 'positive' conditions on the right side of the diverging bar
    for condition in ['תקין (מטופח)', 'לא רלוונטי']:
        fig.add_trace(go.Bar(
            y=columns_to_plot,
            x=condition_counts[condition],  # Positive values
            name=condition,
            orientation='h',
            marker_color=color_map[condition],
             hovertemplate='<b>כמות נטו: </b>%{customdata:.0f}<br>',
            customdata=condition_counts_norm[condition] # Values to display on hover
            # hovertemplate = '<b>%{y}:</b><br>' +  # Column label (waste type)
            #                 '<b></b> %{customdata:.2f}<extra></extra>',
            # customdata = condition_counts[condition]
        ))

    # Plot the 'negative' conditions on the left side of the diverging bar
    for condition in ['סביר', 'לא תקין (מוזנח)']:
        fig.add_trace(go.Bar(
            y=columns_to_plot,
            x=[-count for count in condition_counts[condition]],  # Negative values for diverging bar
            name=condition,
            orientation='h',
            marker_color=color_map[condition],
            hovertemplate='<b>כמות נטו: </b>%{customdata:.0f}<br>', # Column label (infrastructure category)
            customdata=condition_counts_norm[condition]  # Values to display on hover
        ))

    # Update the layout
    fig.update_layout(
        barmode='relative',
        title={
            'text': 'תקינות התשתיות',
            'x': 1.0,  # Align title to the right
            'xanchor': 'right'
        },
        xaxis_title='כמות',
        yaxis_title='קטגוריות',
        font=dict(family='Arial, sans-serif', size=14),
        height=400,
        legend_title='מצב',
        xaxis=dict(
            tickvals=[-1, 0, 1],  # Set ticks at -1, 0, and 1
            ticktext=['Negative', '0', 'Positive'],  # Custom tick labels
            range=[-1, 1],  # Limit x-axis range to -1 to 1 (100% each side)
            tickformat=".0%"  # Display values as percentages
        )
    )

    # Show the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def sort_age_bins(age_bin):
    new_age_bin = age_bin.split('-')
    if len(new_age_bin) == 1:
        lower_bound = int(age_bin.split(' ')[0])
    else:
        lower_bound = int(new_age_bin[0])
    return lower_bound

# def plot_behaviors_teorshlilihiyuvi(df, city, measurer_type):
#     # Filter the data for the selected city
#     st.markdown("""
#         <style>
#         .stRadio > label {
#             float: right;  /* Align the label text to the right */
#         }
#         .stRadio [role=radiogroup] {
#             justify-content: flex-end;  /* Align the radio buttons to the right */
#         }
#         </style>
#     """, unsafe_allow_html=True)
#
#     x_axis_val = st.radio(
#         'בחר/י פילוח לפי',  # Label for the radio buttons in Hebrew
#         ('מגדר', 'גיל'),
#         index=1,
#         horizontal=True
#     )
#     x_axis_dict = {'מגדר': 'gender', 'גיל': 'age'}
#     x_axis_col = x_axis_dict[x_axis_val]
#
#     if city != 'כל הארץ':
#         # Filter the data for the selected city
#         df_filtered = df[(df['יישוב'] == city) & (df['סוגנקודתהמדידהתשובה'].isin(measurer_type))]
#     else:
#         df_filtered = df[df['סוגנקודתהמדידהתשובה'].isin(measurer_type)]
#
#     # Group by the chosen x-axis column (either 'gender' or 'age') and count occurrences of 'teorshlilihiyuvi'
#     grouped_data = df_filtered.groupby([x_axis_col, 'teorshlilihiyuvi']).size().unstack().fillna(0)
#     # Sort the data by age bins if the selected x-axis is 'age'
#     # Custom sorting function to extract the lower bound of age bins
#
#     # Sort the data by age bins using the custom function if the selected x-axis is 'age'
#     if x_axis_col == 'age':
#         grouped_data = grouped_data.reindex(sorted(grouped_data.index, key=sort_age_bins))
#
#     # Calculate total occurrences for percentage calculations
#     total_counts = grouped_data.sum().sum()  # Total counts across all groups
#
#     # Create the figure with Plotly graph_objects
#     fig = go.Figure()
#
#     color_map = ['#1f77b4', '#ff7f0e']  # Blue and Orange
#
#     # Add bars for each 'teorshlilihiyuvi' category
#     for i, value in enumerate(grouped_data.columns):
#         # Calculate absolute counts for each bar
#         absolute_counts = grouped_data[value]
#
#         # Calculate percentages relative to the entire plot
#         overall_percentage = (absolute_counts / total_counts) * 100  # Percent relative to total
#
#         # Calculate group-specific percentages
#         group_total = grouped_data.sum(axis=1)  # Total for the current group
#         group_percentage = (absolute_counts / group_total) * 100  # Percent relative to group
#
#         fig.add_trace(go.Bar(
#             x=grouped_data.index,
#             y=absolute_counts,
#             name=f' התנהגות {value}',
#             marker=dict(color=color_map[i]),
#             text=absolute_counts,  # Add data labels
#             textposition='auto',
#             hovertemplate='<b>אחוז מההתנהגויות בקבוצה זו::</b> %{customdata[0]:.2f}%<br>' +
#                           '<b>אחוז מכלל ההתנהגויות:</b> %{customdata[1]:.2f}%<extra></extra>',
#             customdata=np.column_stack((group_percentage, overall_percentage))  # Combine both percentages
#         ))
#
#     title_text = ' פיזור התנהגות שלילית וחיובית לפי ' + x_axis_val
#     # Update layout for Hebrew labels and other visual elements
#     fig.update_layout(
#         title={
#             'text': f"{title_text} <br>",
#             'x': 1.0,  # Align to the right
#             'xanchor': 'right'
#         },
#         xaxis_title=x_axis_val,
#         yaxis_title='ספירת מופעים',
#         font=dict(family='Arial, sans-serif', size=14),
#         xaxis_tickangle=-45,
#         bargap=0.2,
#         height=500,
#         legend=dict(
#             x=1.2,  # Position legend on the right
#             y=1,  # Position legend at the top
#             xanchor='right',  # Align text to the right
#             orientation="v",  # Horizontal layout of legend
#             font=dict(family="Arial", size=12, color="black"),
#             itemclick="toggleothers",  # Allow toggling between legend items
#             itemdoubleclick="toggle"  # Single item toggle on double click
#         )
#     )
#
#     # Show the interactive Plotly plot in Streamlit
#     st.plotly_chart(fig, use_container_width=True)

def plot_behaviors_teorshlilihiyuvi(df, city, measurer_type):
    # Filter the data for the selected city
    st.markdown("""
        <style>
        .stRadio > label {
            float: right;  /* Align the label text to the right */
        }
        .stRadio [role=radiogroup] {
            justify-content: flex-end;  /* Align the radio buttons to the right */
        }
        </style>
    """, unsafe_allow_html=True)

    x_axis_val = st.radio(
        'בחר/י פילוח לפי',  # Label for the radio buttons in Hebrew
        ('מגדר', 'גיל'),
        index=1,
        horizontal=True
    )

    # Mapping Hebrew categories to DataFrame columns
    x_axis_dict = {'מגדר': 'gender', 'גיל': 'age'}
    x_axis_col = x_axis_dict[x_axis_val]

    # Filter the DataFrame based on the selected city and measurement type
    if city != 'כל הארץ':
        df_filtered = df[(df['יישוב'] == city) & (df['סוגנקודתהמדידהתשובה'].isin(measurer_type))]
    else:
        df_filtered = df[df['סוגנקודתהמדידהתשובה'].isin(measurer_type)]

    # Count occurrences of 'teorshlilihiyuvi' grouped by selected column
    grouped_data = df_filtered.groupby([x_axis_col, 'teorshlilihiyuvi']).size().reset_index(name='count')

    # Create a Sunburst chart
    fig = px.sunburst(
        grouped_data,
        path=[x_axis_col, 'teorshlilihiyuvi'],  # Hierarchical structure
        values='count',  # Count of occurrences
        color='teorshlilihiyuvi',  # Coloring by behavior type
        color_discrete_sequence=['#1f77b4', '#ff7f0e'],  # Blue and Orange
        title='פיזור התנהגות שלילית וחיובית לפי ' + x_axis_val,
        labels={'teorshlilihiyuvi': 'סוג התנהגות', x_axis_col: x_axis_val}  # Custom labels
    )

    # Show the interactive Plotly plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def plot_combined_behaviors(df, city, measurer_type):
    # Filter the data for the selected city
    st.markdown("""
        <style>
        .stRadio > label {
            float: right;  /* Align the label text to the right */
        }
        .stRadio [role=radiogroup] {
            justify-content: flex-end;  /* Align the radio buttons to the right */
        }
        </style>
    """, unsafe_allow_html=True)

    x_axis_val = st.radio(
        'בחר/י פילוח לפי',  # Label for the radio buttons in Hebrew
        ('משולב','מגדר', 'גיל'),  # Added 'משולב' option
        index=1,
        horizontal=True
    )
    x_axis_dict = {'מגדר': 'gender', 'גיל': 'age'}
    x_axis_col = x_axis_dict.get(x_axis_val, None)  # Get column based on selection

    if city != 'כל הארץ':
        # Filter the data for the selected city
        df_filtered = df[(df['יישוב'] == city) & (df['סוגנקודתהמדידהתשובה'].isin(measurer_type))]
    else:
        df_filtered = df[df['סוגנקודתהמדידהתשובה'].isin(measurer_type)]

    if x_axis_val == 'משולב':
        plot_combined_stacked(df_filtered)
    else:
        # Group by the chosen x-axis column (either 'gender' or 'age') and count occurrences of 'teorshlilihiyuvi'
        grouped_data = df_filtered.groupby([x_axis_col, 'teorshlilihiyuvi']).size().unstack().fillna(0)

        # Sort the data by age bins if the selected x-axis is 'age'
        if x_axis_col == 'age':
            grouped_data = grouped_data.reindex(sorted(grouped_data.index, key=sort_age_bins))

        # Calculate total occurrences for percentage calculations
        total_counts = grouped_data.sum().sum()  # Total counts across all groups

        # Create the figure with Plotly graph_objects
        fig = go.Figure()

        color_map = ['#1f77b4', '#ff7f0e']  # Blue and Orange

        # Add bars for each 'teorshlilihiyuvi' category
        for i, value in enumerate(grouped_data.columns):
            absolute_counts = grouped_data[value]

            # Calculate percentages
            overall_percentage = (absolute_counts / total_counts) * 100
            group_total = grouped_data.sum(axis=1)
            group_percentage = (absolute_counts / group_total) * 100

            fig.add_trace(go.Bar(
                x=grouped_data.index,
                y=absolute_counts,
                name=f' התנהגות {value}',
                marker=dict(color=color_map[i]),
                text=absolute_counts,
                textposition='auto',
                hovertemplate='<b>אחוז מההתנהגויות בקבוצה זו::</b> %{customdata[0]:.2f}%<br>' +
                              '<b>אחוז מכלל ההתנהגויות:</b> %{customdata[1]:.2f}%<extra></extra>',
                customdata=np.column_stack((group_percentage, overall_percentage))
            ))

        title_text = ' פיזור התנהגות שלילית וחיובית לפי ' + x_axis_val
        # Update layout for Hebrew labels and other visual elements
        fig.update_layout(
            title={
                'text': f"{title_text} <br>",
                'x': 1.0,
                'xanchor': 'right'
            },
            xaxis_title=x_axis_val,
            yaxis_title='ספירת מופעים',
            font=dict(family='Arial, sans-serif', size=14),
            xaxis_tickangle=-45,
            bargap=0.2,
            height=500,
            legend=dict(
                x=1.2,
                y=1,
                xanchor='right',
                orientation="v",
                font=dict(family="Arial", size=12, color="black"),
                itemclick="toggleothers",
                itemdoubleclick="toggle"
            )
        )

        # Show the interactive Plotly plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

def plot_combined_stacked(df):
    # Group by age and gender and count occurrences of 'teorshlilihiyuvi'
    grouped_data = df.groupby(['age', 'gender', 'teorshlilihiyuvi']).size().unstack(fill_value=0)

    # Create the figure with Plotly graph_objects
    fig = go.Figure()

    # Color map for behavior types and genders
    color_map = {
        ('זכר', 'חיובית'): '#D55E00',  # Blue for Male - Positive
        ('זכר', 'שלילית'): '#F2A900',  # Orange for Male - Negative
        ('נקבה', 'חיובית'): '#5DADE2',  # Green for Female - Positive
        ('נקבה', 'שלילית'): '#0072B2'   # Red for Female - Negative
    }
    # Add bars for each gender and behavior type
    for (gender, behavior) in color_map.keys():
        # Get counts for each behavior type for the specific gender
        behavior_counts = grouped_data.loc[(slice(None), gender), :][behavior]

        # Add a bar for each gender-behavior pair
        fig.add_trace(go.Bar(
            x=behavior_counts.index.get_level_values(0),  # Age groups
            y=behavior_counts,
            name=f'{gender} - {behavior}',
            marker=dict(color=color_map[(gender, behavior)]),
            text=behavior_counts,  # Add data labels
            textposition='auto',
            hovertemplate='אחוז מההתנהגויות: %{y}<extra></extra>',
        ))

    title_text = 'התנהגות לפי גיל ומגדר'
    # Update layout for Hebrew labels and other visual elements
    fig.update_layout(
        title={
            'text': f"{title_text} <br>",
            'x': 0.5,  # Center title
            'xanchor': 'center'
        },
        xaxis_title='גיל',
        yaxis_title='ספירת מופעים',
        font=dict(family='Arial, sans-serif', size=14),
        xaxis_tickangle=-45,
        barmode='group',  # Set bars to be grouped
        height=500,
        legend=dict(
            x=1.1,  # Position legend on the right
            y=1,  # Position legend at the top
            xanchor='right',  # Align text to the right
            orientation="v",  # Vertical layout of legend
            font=dict(family="Arial", size=12, color="black"),
            itemclick="toggleothers",  # Allow toggling between legend items
            itemdoubleclick="toggle"  # Single item toggle on double click
        )
    )

    # Show the interactive Plotly plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# Example usage:
# plot_behaviors_age_gender_stacked(data_frames['behaviors'])


# Example usage:
# plot_behaviors_age_gender_stacked(data_frames['behaviors'])


def get_color_scale(avg_value, min_value, max_value):
    """
    Function to normalize selected values and map them to a color scale
    :param avg_value:
    :param min_value:
    :param max_value:
    :return:
    """
    norm = Normalize(vmin=min_value, vmax=max_value)
    cmap = matplotlib.colormaps.get_cmap('RdBu_r')
    return cm.colors.rgb2hex(cmap(norm(avg_value)))


def plot_top_k_behaviors(df, city, k, prefix, measurer_type):
    # Filter the dataframe for the chosen city
    if city != 'כל הארץ':
        df_filtered = df[(df['יישוב'] == city) & (df['סוגנקודתהמדידהתשובה'].isin(measurer_type))]
    else:
        df_filtered = df[df['סוגנקודתהמדידהתשובה'].isin(measurer_type)]

    # Add gender and age filters
    gender_options = ['הכל'] + df_filtered['gender'].dropna().unique().tolist()
    selected_gender = st.selectbox("סינון לפי מגדר", gender_options, index=0)

    age_options = ['הכל'] + df_filtered['age'].dropna().unique().tolist()
    selected_age = st.selectbox("סינון לפי טווח גילאים", age_options, index=0)

    # Apply gender filter if selected
    if selected_gender != 'הכל':
        df_filtered = df_filtered[df_filtered['gender'] == selected_gender]

    # Apply age group filter if selected
    if selected_age != 'הכל':
        df_filtered = df_filtered[df_filtered['age'] == selected_age]

    # List of the 'heged' columns
    heged_columns = [f'heged{i}' for i in range(1, 14)]

    # Melt the dataframe to turn all 'heged' columns into rows (ignoring null values)
    df_melted = df_filtered.melt(id_vars=['יישוב', 'point_type'], value_vars=heged_columns,
                                 var_name='heged_type', value_name='phrase').dropna(subset=['phrase'])

    df_melted = df_melted[df_melted['phrase'].notnull() & (df_melted['phrase'] != '') & (
                df_melted['phrase'] != 'אחר (יש לפרט את התיאור המתאים בהערות)')]

    # Count the occurrences of each phrase
    phrase_counts = df_melted['phrase'].value_counts()

    # Get the top k phrases
    top_k_phrases = phrase_counts.nlargest(k)
    others_count = phrase_counts.sum() - top_k_phrases.sum()

    # Combine the top k phrases with the "Others" slice
    all_phrases = pd.concat([top_k_phrases, pd.Series({'אחר': others_count})])

    # Calculate the percentage of each phrase, including "Others"
    total_phrases_count = phrase_counts.sum()
    phrases_percentage = (all_phrases / total_phrases_count) * 100

    # Define the color palette based on the number of phrases
    colors = {
        3: ['#D55E00', '#F2A900', '#5DADE2', '#b6cfc5'],  # Light gray for "Others"
        5: ['#D55E00', '#F2A900', '#5DADE2', '#0072B2', '#182139', '#b6cfc5'],
        10: ['#D55E00', '#F2A900', '#FFCC00', '#A7D600', '#5DADE2', '#0072B2', '#4B0082', '#9933CC', '#FF66B2',
             '#FF3333', '#b6cfc5']
    }

    # Create a donut chart using Plotly
    fig = px.pie(
        names=phrases_percentage.index,
        values=phrases_percentage.values,
        hole=0.4,  # Creates the donut hole
        title=f'Top {k} Phrases and Others in {city}',  # This title will be overridden
        labels={'phrase': 'Phrase', 'count': 'Count'},
        color_discrete_sequence=colors[k],
    )

    # Update the layout to customize the title
    fig.update_layout(
        title={
            'text': f' {prefix} ההתנהגויות הסביבתיות המובילות',
            'x': 1.0,  # Align to the right
            'xanchor': 'right'
        },
    )

    fig.update_traces(sort=False, direction="clockwise")
    fig.update_traces(textinfo='percent', texttemplate='%{percent:.1%}')

    # Show the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def plot_top_k_stacked_bar_chart(df, city, k, prefix, measurer_type):
    # Filter the dataframe for the chosen city
    if city != 'כל הארץ':
        df_filtered = df[(df['יישוב'] == city) & (df['סוגנקודתהמדידהתשובה'].isin(measurer_type))]
    else:
        df_filtered = df[df['סוגנקודתהמדידהתשובה'].isin(measurer_type)]

    # Add gender and age filters
    gender_options = ['הכל'] + df_filtered['gender'].dropna().unique().tolist()
    selected_gender = st.selectbox("סינון לפי מגדר", gender_options, index=0)

    age_options = ['הכל'] + df_filtered['age'].dropna().unique().tolist()
    selected_age = st.selectbox("סינון לפי טווח גילאים", age_options, index=0)

    # Apply gender filter if selected
    if selected_gender != 'הכל':
        df_filtered = df_filtered[df_filtered['gender'] == selected_gender]

    # Apply age group filter if selected
    if selected_age != 'הכל':
        df_filtered = df_filtered[df_filtered['age'] == selected_age]

    # List of the 'heged' columns
    heged_columns = [f'heged{i}' for i in range(1, 14)]

    # Melt the dataframe to turn all 'heged' columns into rows (ignoring null values)
    df_melted = df_filtered.melt(id_vars=['יישוב', 'point_type'], value_vars=heged_columns,
                                 var_name='heged_type', value_name='phrase').dropna(subset=['phrase'])

    df_melted = df_melted[df_melted['phrase'].notnull() & (df_melted['phrase'] != '') & (
                df_melted['phrase'] != 'אחר (יש לפרט את התיאור המתאים בהערות)')]

    # Count the occurrences of each phrase
    phrase_counts = df_melted['phrase'].value_counts()

    # Get the top k phrases
    top_k_phrases = phrase_counts.nlargest(k).index.tolist()

    # Filter the melted dataframe to include only the top k phrases
    df_top_k = df_melted[df_melted['phrase'].isin(top_k_phrases)]

    # Count the occurrences of each phrase grouped by area type (or any other category you want)
    phrase_counts_grouped = df_top_k.groupby(['phrase', 'point_type']).size().reset_index(name='count')

    # Create a Stacked Bar Chart using Plotly
    fig = px.bar(
        phrase_counts_grouped,
        x='phrase',
        y='count',
        color='point_type',  # Coloring by area type (or you can choose gender or age group)
        title=f'{prefix} ההתנהגויות הסביבתיות המובילות לפי סוג נקודת מדידה',
        labels={'phrase': 'Phrase', 'count': 'Count', 'point_type': 'Measurement Type'},
        barmode='stack'  # Stacked bar chart
    )

    # Update the layout to customize the title and orientation
    fig.update_layout(
        title={
            'text': f'{prefix} {k} ההתנהגויות השליליות המובילות לפי סוג אזור',
            'x': 0.5,  # Center the title
            'xanchor': 'center'
        },
        xaxis_title='התנהגות',
        yaxis_title='כמות',
        legend_title='סוג אזור',
    )

    # Show the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def plot_waste_levels_by_city(df, city, measurer_type):
    # Define the fixed waste levels in the correct order
    waste_levels_order = ['ריק', '1/4', '1/2', '3/4', 'מלא']

    # Filter the data where 'האם יש פחים בנקודת המדידה' is 'כן' and the selected city
    if city != 'כל הארץ':
        # Filter the data for the selected city
        df_filtered = df[
            (df['האם יש פחים בנקודת המדידה'] == 'כן') &
            (df['יישוב'] == city) & (df['סוגנ קודת המדידה'].isin(measurer_type))]
    else:
        df_filtered = df[(df['האם יש פחים בנקודת המדידה'] == 'כן') & (df['סוגנ קודת המדידה'].isin(measurer_type))]

    # Count occurrences of each 'מפלס הפסולת בפח' level
    waste_level_counts = df_filtered['מפלס הפסולת בפח'].value_counts()

    # Reindex the series to ensure all levels are present, even if their count is zero
    waste_level_counts = waste_level_counts.reindex(waste_levels_order, fill_value=0)

    # Convert Hebrew text to right-to-left for the labels (this ensures the order is correct in Hebrew)
    labels = waste_levels_order

    # Optional: Define custom colors if needed (replace color_map with your color choices)
    colors = [
        '#0072B2',  # Dark Blue
        '#5DADE2',  # Light Blue
        '#b6cfc5',  # Light Orange
        '#FF8C00',  # Bright Orange
        '#D55E00'   # Red-Orange
    ][:len(waste_levels_order)]

    total_count = waste_level_counts.sum()

    # Create the figure with Plotly graph_objects
    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=waste_level_counts.values,
        marker_color=colors,
        text=waste_level_counts.values,  # Add data labels
        textposition='auto',
        hovertemplate='<b>אחוז מכלל הפחים:</b> %{customdata:.2f}%<extra></extra>',
        customdata= (waste_level_counts / total_count) * 100 if total_count > 0 else [0] * len(labels)
        # Automatically position the text
    )])

    # Update layout for Hebrew labels
    fig.update_layout(
        title={
            'text': f'מפלסי הפסולת בפחים',
            'x': 1.0,  # Align to the right
            'xanchor': 'right'
        },
        xaxis_title='מפלס הפסולת בפח',
        yaxis_title='כמות',
        font=dict(family='Arial, sans-serif', size=14),
        bargap=0.2,
        height=400
    )

    # Show the interactive Plotly plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def prepare_data(data_frames):
    """
    Function to prepare the data
    :param data_frames:
    :return:
    """
    bin_storage_df = data_frames['bin_storage']
    behaviors_df = data_frames['behaviors']
    infrastructures_df = data_frames['infrastructures']
    dirt_information_df = data_frames['dirt_information']
    # prepare dirt_information
    columns_to_convert = [
        'בדלי סיגריות', 'קופסאות סיגריות', 'מסכות כירורגיות',
        'מכלי משקה למיניהם', 'פקקים של מכלי משקה', 'אריזות מזון Take Away נייר',
        'אריזות מזון Take Away פלסטיק', 'צלחות חדפ', 'סכום חדפ',
        'כוסות שתייה קרה חדפ', 'כוסות שתייה חמה חדפ', 'אריזות של חטיפים',
        'זכוכית לא מכלי משקה או לא ניתן לזיה', 'נייר אחר לא אריזות מזון',
        'פלסטיק אחר שקיות פלסטיק ורכיבי פלס', 'פסולת אורגנית', 'פסולת בלתי חוקית שקית אשפה מלאה שה',
        'פסולת אחרת למשל בגדים סוללות חומרי', 'צואת כלבים', 'כתמי מסטיק',
        'פריט פסולת גדול', 'אריזות קרטון', 'גרפיטי', 'אחר1', 'אחר2'
    ]

    dirt_information_df.dropna(subset=['נ.צ כתובת', 'יישוב'], inplace=True)
    dirt_information_df[columns_to_convert] = dirt_information_df[columns_to_convert].apply(pd.to_numeric, errors='coerce')
    # Custom function to process each cell

    def process_cell(cell):
        # Step 1: Check if cell says "לא" or "אין"
        if cell in ['לא', 'אין', 'None']:
            return 0
        # Step 3: Check if the cell contains numbers (comma-separated)
        numbers = re.findall(r'\d+', cell)
        if len(numbers) > 0:
            return sum(map(int, numbers))
        # Step 4: If none of the above, return 1
        return 1

    dirt_information_df['אחר1'] = dirt_information_df['אחר1'].astype(str)
    dirt_information_df['אחר2'] = dirt_information_df['אחר2'].astype(str)
    # Apply the function to both columns and sum the results
    dirt_information_df['שונות'] = dirt_information_df['אחר1'].apply(process_cell) + dirt_information_df['אחר2'].apply(process_cell)
    dirt_information_df.drop(columns=['אחר1', 'אחר2'], inplace=True)
    dirt_information_df.rename(columns={
        'זכוכית לא מכלי משקה או לא ניתן לזיה': 'זכוכית לא מכלי משקה או לא ניתן לזיהוי',
        'פלסטיק אחר שקיות פלסטיק ורכיבי פלס': 'פלסטיק אחר שקיות פלסטיק ורכיבי פלסטיק',
        'פסולת בלתי חוקית שקית אשפה מלאה שה': 'פסולת בלתי חוקית/ שקית אשפה מלאה',
        'פסולת אחרת למשל בגדים סוללות חומרי': 'פסולת אחרת למשל בגדים, סוללות'
    }, inplace=True)

    # prepare behaviors_df
    behaviors_df['teorshlilihiyuvi'] = behaviors_df['teorshlilihiyuvi'].replace(r'^\s*$', np.nan, regex=True)
    behaviors_df['age'] = behaviors_df['age'].replace(r'^\s*$', np.nan, regex=True)

    return bin_storage_df, behaviors_df, infrastructures_df, dirt_information_df

if __name__ == '__main__':
    # Dictionary containing table names and resource IDs
    api_resource_ids = {
        'bin_storage': '3436bb7f-8b67-49be-94a4-3c34c9cc1e7a',
        'behaviors': 'ece44fa9-f47c-4116-a16e-9477e0d4d2dc',
        'infrastructures': '7b32b590-d130-4f41-bb63-1ca462d91f3a',
        'dirt_information': '94400680-6a74-4c4b-be55-704e20ca4e76'
    }
    # Fetch data and create DataFrames

    data_frames = fetch_data_and_create_df(api_resource_ids)
    bin_storage_df, behaviors_df, infrastructures_df, dirt_information_df = prepare_data(data_frames)
    waste_types = [
        'בדלי סיגריות', 'קופסאות סיגריות', 'מסכות כירורגיות',
        'מכלי משקה למיניהם', 'פקקים של מכלי משקה', 'אריזות מזון Take Away נייר',
        'אריזות מזון Take Away פלסטיק', 'צלחות חדפ', 'סכום חדפ',
        'כוסות שתייה קרה חדפ', 'כוסות שתייה חמה חדפ', 'אריזות של חטיפים',
        'זכוכית לא מכלי משקה או לא ניתן לזיה', 'נייר אחר לא אריזות מזון',
        'פלסטיק אחר שקיות פלסטיק ורכיבי פלס', 'פסולת אורגנית', 'פסולת בלתי חוקית שקית אשפה מלאה שה',
        'פסולת אחרת למשל בגדים סוללות חומרי', 'צואת כלבים', 'כתמי מסטיק',
        'פריט פסולת גדול', 'אריזות קרטון', 'גרפיטי', 'שונות'
    ]
    city_impressions = infrastructures_df.groupby('יישוב').agg(
        avg_impression=('התרשמותכלליתמנקודתהמדידה', 'mean')
    ).reset_index()
    city_impressions = city_impressions[city_impressions['יישוב'].isin(dirt_information_df['יישוב'].unique())]

    # Sort the DataFrame by 'יישוב'
    city_impressions = city_impressions.sort_values(by='יישוב')

    cities_sort = ['כל הארץ'] + [
        f"{row['יישוב']} ({row['avg_impression']:.2f})" for _, row in city_impressions.iterrows()
    ]
    measurer_type_list = dirt_information_df['סוג נקודת המדידה'].unique()

    # Set up the page configuration
    st.set_page_config(
        page_title="המשרד להגנת הסביבה - ניקיון במרחב הציבורי",
        page_icon="🚮",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # Add the header to the page
    st.markdown(
        """
        <style>
        .container {
            display: flex;
            justify-content: flex-end;
        }
        </style>
        <div class="container">
            <h1>🚮 המשרד להגנת הסביבה - ניקיון במרחב הציבורי</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align: right;">דאשבורד זה פותח עבורכם, המשרד להגנת הסביבה, במטרה לסייע לכם בגיבוש תוכניות התערבות מבוססות נתונים לשיפור רמת הניקיון במרחב הציבורי. המערכת מציגה את מצב הפסולת והתשתיות בנקודות מדידה שונות ברחבי הארץ, לצד מידע על התנהגויות הציבור המשפיעות על הניקיון. באמצעות הוויזואליזציה, תוכלו לזהות תחומים מרכזיים בהם יש למקד את המשאבים לשם יצירת שינוי משמעותי, כגון שיפור תשתיות, הגברת פינוי הפסולת, או קמפיינים ממוקדים לשינוי התנהגותי. בכל שלב, תוכלו להיעזר בפאנל הנפתח מצד שמאל כדי להגדיר סינונים רלוונטיים. ראשית, ביכולתכם לבחור את היישוב שבו תרצו להתמקד. המספרים המופיעים בסוגריים ליד השם של כל יישוב ברשימה הנ״ל מייצגים את מדד הניקיון הכללי של היישוב, כך שתוכלו לבחור להתמקד קודם ביישובים שמצבם פחות טוב מאחרים. בנוסף, תוכלו לבחור אילו סוגים של נקודות מדידה מעניינים אתכם, וכן כמה פריטים תרצו שיוצגו בגרפים המותאמים לכך </p>',
        unsafe_allow_html=True
    )
    # Sidebar for user inputs
    with st.sidebar:
        st.markdown(
            """
            <style>
                /* Custom styles for the sidebar */
                section[data-testid="stSidebar"] {
                    width: 100px !important; /* Set the width to your desired value */
                }
                .stSelectbox > label,
                .stMultiselect > label {
                    float: right;  /* Align the label text to the right */
                    text-align: right;  /* Ensure the text aligns to the right */
                    direction: rtl;  /* Set direction to right-to-left */
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        city_full = st.selectbox('בחר/י עיר', cities_sort)
        if city_full != 'כל הארץ':
            city=city_full[:-7]
        else:
            city = city_full
        k = st.selectbox('בחר/י כמות להצגה', [3, 5, 10], index=1)

        # Determine prefix based on value of k
        if k == 3:
            prefix = 'שלושת'
        elif k == 5:
            prefix = 'חמשת'
        elif k == 10:
            prefix = 'עשרת'
        else:
            prefix = ''  # Default case if K doesn't match any known values

        measurer_type = st.multiselect('בחר/י סוגי נקודות מדידה', measurer_type_list, measurer_type_list)

    # Align the header to the right and dynamically insert the city
    st.markdown(f'<h2 style="text-align: right;">מיפוי נקודות מדידת הפסולת ב{city}</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: right;">במפה שלפניכם מוצגות נקודות הציון המדויקות של המקומות בהם הסוקרים מדדו את כמות הפסולת בכל יישוב. באפשרותכם להתרשם מסקירה כללית של נקודות המדידה בכל הארץ, או להתמקד ביישוב מסוים מתוך הרשימה הנמצאת בצד שמאל. בנוסף, באפשרותכם לבחור באילו מסוגי הפסולת תרצו להתמקד באמצעות בחירה מהרשימה הנגללת המופיעה מטה. צבע הנקודות במפה מסמן את חומרת רמת הפסולת בנקודה – ככל שהנקודה אדומה יותר משמע יש בה יותר פריטי פסולת מהסוג הנבחר, וככל שהיא כחולה יותר כך המקום נקי יותר מסוג פסולת זה. בלחיצה על נקודה במפה תוכלו לראות את כתובת הנקודה או התיאור שניתן לה על ידי הסוקר, את סוג נקודת המדידה (רחוב מגורים, רחוב מסחרי, מרכז מסחרי, פארק – פנאי ונופש, אזור תעשייה, מבני ציבור או חוף ים), וכן את כמות פריטי הפסולת מהסוג הנבחר. במידה ונבחרו יותר מ-5 סוגי פסולת להצגה, יוצגו רק חמשת הסוגים הבולטים ביותר, כאשר הם ממוינים בסדר יורד לפי כמות פריטי הפסולת שנמצאו בנקודה</p>',
        unsafe_allow_html=True)

    st.markdown("""
        <style>
        .stMultiSelect > label {
            float: right;  /* Align the label text to the right */
            text-align: right;  /* Ensure the text aligns to the right */
        }
        .stMultiSelect [role=group] {
            direction: rtl;  /* Set direction to right-to-left */
        }
        </style>
    """, unsafe_allow_html=True)

    # Define the waste types
    waste_types = [
        'בדלי סיגריות', 'קופסאות סיגריות', 'מסכות כירורגיות',
        'מכלי משקה למיניהם', 'פקקים של מכלי משקה', 'אריזות מזון Take Away נייר',
        'אריזות מזון Take Away פלסטיק', 'צלחות חדפ', 'סכום חדפ',
        'כוסות שתייה קרה חדפ', 'כוסות שתייה חמה חדפ', 'אריזות של חטיפים',
        'זכוכית לא מכלי משקה או לא ניתן לזיהוי', 'נייר אחר לא אריזות מזון',
        'פלסטיק אחר שקיות פלסטיק ורכיבי פלסטיק', 'פסולת אורגנית', 'פסולת בלתי חוקית/ שקית אשפה מלאה',
        'פסולת אחרת למשל בגדים, סוללות', 'צואת כלבים', 'כתמי מסטיק',
        'פריט פסולת גדול', 'אריזות קרטון', 'גרפיטי', 'שונות'
    ]

    # Add 'All' to the list
    waste_types_with_all = waste_types + ['הכל']

    # Create the multiselect widget
    selected_columns = st.multiselect(
        'בחר/י את סוג הפסולת',
        waste_types_with_all,
        default=['בדלי סיגריות']  # Default selection
    )

    # Logic to handle 'All' selection
    if 'הכל' in selected_columns:
        selected_columns = waste_types  # Include all options if 'All' is selected

    # Render the map
    folium_map = update_map(dirt_information_df, city, selected_columns, measurer_type, bin_storage_df)
    #folium_static(folium_map, width=1100, height=250)
    # Embed the map in an HTML iframe for responsive width
    map_html = folium_map._repr_html_()
    st.components.v1.html(map_html, width=None, height=500)

    # Align the header to the right and dynamically insert the city
    st.markdown(f'<h2 style="text-align: right;">פילוח הפסולת הסביבתית ב{city}</h2>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="text-align: right;">הגרף שלפניכם מציג את {prefix} סוגי הפסולת הכי נפוצים ביישוב (באפשרותכם לשנות את כמות סוגי הפסולת המוצגת בעזרת תיבת הבחירה המופיעה משמאל). לכל סוג פסולת יש עמודה שמייצגת את כמות הפריטים הממוצעת שנמדדה על פני כל נקודות המדידה ביישוב. בנוסף, בכל עמודה מופיע סימון של איקס שחור, המסמל את הממוצע הארצי של פריטי פסולת מסוג זה. בעזרת גרף זה, ובשילוב עם הגרפים הנוספים המופיעים בדאשבורד, תוכלו למפות את הבעיות הדחופות ביותר לטיפול ביישוב ולהעריך את חומרתן ביחס למצב הארצי</p>',
        unsafe_allow_html=True
    )


    plot_top_k_waste_types(dirt_information_df, waste_types, city, measurer_type, k, prefix)
    st.markdown('</div>', unsafe_allow_html=True)

    # Create containers for different sections
    st.markdown(
        """
        <div style="background-color: #f0f0f0; padding: 10px; margin-top: 20px; text-align: right;">
        </div>
        """,
        unsafe_allow_html=True
    )

    # Align the header to the right and dynamically insert the city
    st.markdown(f'<h2 style="text-align: right;">מצב התשתיות ב{city}</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: right;">בחלק זה מוצג מצב התשתיות ביישוב, באמצעות שני גרפים. בגרף הימני תוכלו ללמוד על מצב הפחים ביישוב ולהסיק מכך האם תדירות הפינוי תואמת את כמות הפסולת המצטברת ביישוב. הגרף השמאלי משלים את תמונת המצב ומציג את תקינות התשתיות, כך שכל עמודה אופקית מייצגת סוג תשתית שונה. מצד ימין (בכחול) מוצגות התשתיות שדורגו כתקינות (או לא רלוונטיות), בעוד מצד שמאל (בכתום) מוצגות התשתיות שדורגו במצב סביר או לא תקין. במעבר על כל אחד מהחלקים תוכלו לראות בדיוק כמה פעמים קיבלה כל תשתית את הדירוג הרלוונטי</p>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)  # Create two columns for the infrastructure plots
    with col1:
        plot_infrastructure_condition(infrastructures_df, city, measurer_type)
    with col2:
        plot_waste_levels_by_city(bin_storage_df, city, measurer_type)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div style="background-color: #f0f0f0; padding: 10px; margin-top: 20px; text-align: right;">
        </div>
        """,
        unsafe_allow_html=True
    )

    # Align the header to the right and dynamically insert the city
    st.markdown(f'<h2 style="text-align: right;">התנהגויות הציבור ב{city}</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: right;">בחלק זה מוצגים גרפים המתארים את התנהגויות הציבור במרחב העירוני. הגרף הימני מציג פילוח של התנהגויות חיוביות ושליליות לפי טווחי גילאים או לפי מגדר, ומסייע לנו לזהות את אוכלוסיות היעד שכדאי לפנות אליהן כדי למקסם את שיפור הניקיון במרחב הציבורי. זיהוי נכון של אוכלוסיית היעד, בין אם בטווח גילאים ובין אם במגדר, מאפשר לנו להכווין את המשאבים בצורה מיטבית - בין אם מדובר בפעילות חינוכית בבתי הספר ובגנים, הגברת פעילות אכיפה, תליית כרזות, ועוד. במעבר על העמודות תוכלו לראות איזה אחוז מכלל ההתנהגויות מהווה עמודה זו, וכן איזה אחוז היא מהווה מההתנהגויות בקבוצת הגיל / במגדר זה. הגרף השמאלי מאפשר לנו לצלול לעומק ההתנהגויות ולזהות את ההתנהגויות הבולטות ביותר ביישוב. הבנת הגורמים לפערים בהתנהגויות תסייע לנו לטפל בבעיות באופן ישיר ויעיל, ותומכת בגיבוש אסטרטגיות לשיפור הניקיון במרחב הציבורי</p>',
        unsafe_allow_html=True
    )

    col3, col4 = st.columns(2)  # Create two columns for public behavior plots
    with col3:
        plot_top_k_behaviors(behaviors_df, city, k, prefix, measurer_type)
        #plot_top_k_stacked_bar_chart(behaviors_df, city, k, prefix, measurer_type)

    with col4:
        #plot_behaviors_teorshlilihiyuvi(behaviors_df, city, measurer_type)
        plot_combined_behaviors(behaviors_df, city, measurer_type)

    st.markdown('</div>', unsafe_allow_html=True)
