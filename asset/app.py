import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from dashscope import Application
import dashscope
from env import API

# Part 1: Config
st.set_page_config(
    page_title="Asset Management Optimization",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded")
alt.themes.enable("dark")
color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo',
                        'viridis']
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

# Part 2: Ingest Data
df_reshaped = pd.read_csv('data/final-data.csv', sep=";")

# Part 2.1: Function that we need
# Dashboard
def make_heatmap(input_df, input_y, input_x, input_color, input_color_theme):
    heatmap = alt.Chart(input_df).mark_rect().encode(
            y=alt.Y(f'{input_y}:O', axis=alt.Axis(title="Year", titleFontSize=18, titlePadding=15, titleFontWeight=900, labelAngle=0)),
            x=alt.X(f'{input_x}:O', axis=alt.Axis(title="", titleFontSize=18, titlePadding=15, titleFontWeight=900)),
            color=alt.Color(f'max({input_color}):Q',
                             legend=None,
                             scale=alt.Scale(scheme=input_color_theme)),
            stroke=alt.value('black'),
            strokeWidth=alt.value(0.25),
        ).properties(width=900
        ).configure_axis(
        labelFontSize=12,
        titleFontSize=12
        )
    return heatmap

def make_choropleth(input_df, latitude_col, longitude_col, input_id, input_column, input_color_theme):
    # Membuat scatter mapbox berdasarkan longitude dan latitude
    choropleth = px.scatter_mapbox(
        input_df,
        lat=latitude_col,
        lon=longitude_col,
        color=input_column,
        color_continuous_scale=input_color_theme,
        size_max=15,
        zoom=4,  # Atur zoom untuk fokus pada Indonesia
        mapbox_style="carto-positron",  # Gaya peta, bisa disesuaikan
        text = input_df[input_id], #+ ': ' + input_df[input_column].astype(str),  # Menampilkan label
        labels={'value':'Valuation'}
    )

    # Memperbarui layout
    choropleth.update_traces(textposition='top center')
    choropleth.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return choropleth

def make_donut(input_response, input_text, input_color):
    if input_color == 'blue':
        chart_color = ['#29b5e8', '#155F7A']
    if input_color == 'green':
        chart_color = ['#27AE60', '#12783D']
    if input_color == 'orange':
        chart_color = ['#F39C12', '#875A12']
    if input_color == 'red':
        chart_color = ['#E74C3C', '#781F16']

    source = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100 - input_response, input_response]
    })
    source_bg = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100, 0]
    })

    plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
        theta="% value",
        color=alt.Color("Topic:N",
                        scale=alt.Scale(
                            # domain=['A', 'B'],
                            domain=[input_text, ''],
                            # range=['#29b5e8', '#155F7A']),  # 31333F
                            range=chart_color),
                        legend=None),
    ).properties(width=130, height=130)

    text = plot.mark_text(align='center', color="#29b5e8", font="Lato", fontSize=32, fontWeight=700,
                          fontStyle="italic").encode(text=alt.value(f'{input_response} %'))
    plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=45, cornerRadius=20).encode(
        theta="% value",
        color=alt.Color("Topic:N",
                        scale=alt.Scale(
                            # domain=['A', 'B'],
                            domain=[input_text, ''],
                            range=chart_color),  # 31333F
                        legend=None),
    ).properties(width=130, height=130)
    return plot_bg + plot + text

def format_number(num):
    if num >= 1000000000000:
        return f'Rp {num / 1000000000000:.1f} T'  # Triliun
    elif num >= 1000_000_000:
        return f'Rp {num / 1000000000:.1f} M'   # Miliar
    elif num >= 1000000:
        return f'Rp {num / 1000000:.1f} JT'       # Juta
    elif num >= 000:
        return f'Rp {num / 1000:.1f}'            # Ribuan
    else:
        return f'Rp {num}'

# Gen AI
def call_agent_app(prompt):
    response = Application.call(app_id='90063298ddd4471497cb9e1781653bda',
                                prompt=prompt,
                                api_key=API,)
    return response

def recommend_utilization(value, condition):
    """
    Function to recommend asset utilization based on value and condition.
    """
    if condition == 'Well Maintained' and value > 1000000:
        return "The asset is in good condition and high value. Consider leasing or optimizing it for better returns."
    elif condition == 'Needs Repair' and value > 500000:
        return "The asset may still be valuable. Consider minor renovations and leasing it."
    elif condition == 'Slightly Damaged':
        return "The asset is in poor condition. Consider selling or redeveloping the area."
    else:
        return "Maintain the asset as-is and monitor for future investment opportunities."

def calculate_asset_optimization(input_df):
    if selected_location != 'All' and selected_condition == 'All':
        selected_condition_data = df_reshaped[df_reshaped['provinsi'] == selected_location].reset_index()
    elif selected_location == 'All' and selected_condition != 'All':
        selected_condition_data = df_reshaped[df_reshaped['condition'] == selected_condition].reset_index()
    elif selected_location != 'All' and selected_condition != 'All':
        selected_condition_data = df_reshaped[(df_reshaped['provinsi'] == selected_location) &
                                  (df_reshaped['condition'] == selected_condition)]
    else:
        selected_condition_data = df_reshaped

    # Tobe drop below
    optimize_value = selected_condition_data[selected_condition_data['potential'] == 0].shape[0]
    total_value = selected_condition_data.shape[0]
    selected_condition_data['optimization_difference'] = total_value - optimize_value

    return pd.concat([selected_condition_data.provinsi, selected_condition_data.name, selected_condition_data.potential, selected_condition_data.value], axis=1).sort_values(by="potential", ascending=False)

# Part 3: Sidebar
with st.sidebar:
    st.title('Filter Data & Ask Our AI')
    # Select color theme
    color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo',
                            'viridis']
    selected_color_theme = st.selectbox('Select a color theme', color_theme_list)

    st.write("Here, you can easily explore and filter through location & condition! If you want to see everything, simply select 'All'")
    # List location & condition
    location_list = list(df_reshaped.provinsi.unique())[::-1]
    condition_list = list(df_reshaped.condition.unique())[::-1]

    # Filtering based on location and condition
    selected_location = st.selectbox('Select location', ['All'] + location_list)
    selected_condition = st.selectbox('Select condition', ['All'] + condition_list)

    # Filter the data based on selected filters
    if selected_location != 'All' and selected_condition == 'All':
        df_filtered = df_reshaped[df_reshaped['provinsi'] == selected_location]
    elif selected_location == 'All' and selected_condition != 'All':
        df_filtered = df_reshaped[df_reshaped['condition'] == selected_condition]
    elif selected_location != 'All' and selected_condition != 'All':
        df_filtered = df_reshaped[(df_reshaped['provinsi'] == selected_location) &
                                  (df_reshaped['condition'] == selected_condition)]
    else:
        df_filtered = df_reshaped

    # Opt color theme
    # Sort filtered data
    df_filtered_sorted = df_filtered.sort_values(by="value", ascending=False)
    df_filtered_sorted['longitude'] = pd.to_numeric(df_filtered_sorted['longitude'], errors='coerce')
    df_filtered_sorted['latitude'] = pd.to_numeric(df_filtered_sorted['latitude'], errors='coerce')

    st.write(
        "Here, our gen-AI, ask below!")
    # Chatbot AI
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.sidebar.container():
            st.markdown(f"**{message['role']}**: {message['content']}")

    if prompting := st.sidebar.text_input("Ask about a location or asset utilization here"):
        # Build the initial prompt for the chatbot
        text_awal = f"Provide recommendations on how to optimize asset utilization in {prompting}. " \
                    "Consider whether to sell, lease, or optimize the asset based on its condition and value. "
        prompt = text_awal + prompting

        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompting})
        st.sidebar.markdown(f"**User**: {prompting}")

        # AI response simulation (you can replace this with actual AI model integration)
        with st.sidebar.container():
            # Simulate response generation
            location_data = df_reshaped[df_reshaped['provinsi'].str.contains(prompting, case=False, na=False)]
            if not location_data.empty:
                recommendations = []
                for _, row in location_data.iterrows():
                    rec = recommend_utilization(row['value'], row['condition'])
                    recommendations.append(f"- **{row['name']}** in **{row['provinsi']}**: {rec}")
                full_response = "\n".join(recommendations)
            else:
                full_response = f"No data available for the location: {prompting}. Please try another location."

            # Display assistant response
            st.sidebar.markdown(f"**Assistant**: {full_response}")
            st.session_state.messages.append({"role": "assistant", "content": full_response})

col = st.columns((1.5, 4.5, 2), gap='medium')
with col[0]:
    st.markdown('#### Potential/Non')

    df_asset_difference_sorted = calculate_asset_optimization(df_reshaped)

    # Gain Losses (Teratas)
    first_asset_name = df_asset_difference_sorted.name.iloc[0]
    first_asset_value = format_number(df_asset_difference_sorted.value.iloc[0])
    first_asset_delta = format_number(df_asset_difference_sorted.potential.iloc[0])
    st.metric(label=first_asset_name, value=first_asset_value, delta=first_asset_delta)

    # Gain Losses (Terbawah)
    first_asset_name = df_asset_difference_sorted.name.iloc[-1]
    first_asset_value = format_number(df_asset_difference_sorted.value.iloc[-1])
    first_asset_delta = format_number(df_asset_difference_sorted.potential.iloc[-1])
    st.metric(label=first_asset_name, value=first_asset_value, delta=first_asset_delta)

    st.markdown('#### Percentage Asset Optimization')
    df_optimized = df_asset_difference_sorted[df_asset_difference_sorted.potential == 0]
    df_not_optimized = df_asset_difference_sorted[df_asset_difference_sorted.potential > 0]
    percentage_optimized = round((len(df_optimized) / len(df_asset_difference_sorted)) * 100)
    percentage_not_optimized = round((len(df_not_optimized) / len(df_asset_difference_sorted)) * 100)
    donut_chart_optimized = make_donut(percentage_optimized, 'Optimized', 'green')
    donut_chart_not_optimized = make_donut(percentage_not_optimized, 'Not OK', 'red')

    migrations_col = st.columns((0.2, 1, 0.2))
    with migrations_col[1]:
        st.write('Optimized')
        st.altair_chart(donut_chart_optimized)
        st.write('Not OK')
        st.altair_chart(donut_chart_not_optimized)

with col[1]:
    st.markdown('#### Potential Asset Distribution')

    choropleth = make_choropleth(df_filtered_sorted, 'latitude', 'longitude','kabupaten/kota', 'potential', selected_color_theme)
    st.plotly_chart(choropleth, use_container_width=True)

    heatmap = make_heatmap(df_filtered_sorted, 'year', 'provinsi', 'potential', selected_color_theme)
    st.altair_chart(heatmap, use_container_width=True)

with col[2]:
    st.markdown('#### Top Potential Utilization')
    df_filtered_sorted_new = df_filtered_sorted.rename(columns={'value': 'Asset Value'})
    df_filtered_sorted_new = df_filtered_sorted_new.rename(columns={'potential': 'Potential'})
    df_filtered_sorted_new = df_filtered_sorted_new.sort_values(by="Potential", ascending=False)
    st.dataframe(df_filtered_sorted_new,
                 column_order=("name", "Asset Value", "Potential"),
                 hide_index=True,
                 column_config={
                     "name": st.column_config.TextColumn(
                         "Asset Name",
                     ),
                     "value": st.column_config.ProgressColumn(
                         "Asset Value",
                         format="%f",
                         min_value=0,
                         max_value=max(df_filtered_sorted_new["Asset Value"]),
                     )}
                 )

    with st.expander('About', expanded=True):
        st.write('''
            - Created with Love - Serenade Data.
            - :orange[**Potential/Non**]: Valuation potential asset and non potential
            - :orange[**Percentage Asset Optimization**]: Percentage of optimized vs. non-optimized assets.
            - :orange[**Potential Asset Distribution**]: Distribution asset potential.
            - :orange[**Top Potential Utilization**]: Assessment asset potential to be utilized.
            ''')
