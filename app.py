import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Configuración de página 'Wide'
st.set_page_config(page_title="Geo-Arbitrage Dashboard", layout="wide", page_icon="✈️")

# Estilos CSS personalizados para estética 'Premium'
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        color: #f0f2f6;
    }
    .metric-card {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# --- 1. Ingeniería de Datos Avanzada ---
def get_continent(row):
    """
    Segmentación estricta en 7 continentes basada en Región y País.
    """
    pais = row['País']
    region = row['Región']
    
    # 1. América del Norte (Regla: USA, Canada, Mexico)
    # Incluimos los estados de USA que terminan en '(US)'
    if pais in ['United States', 'Canada', 'Mexico'] or '(US)' in pais:
        return 'América del Norte'
    
    # 2. Australia/Oceanía
    if region == 'North America and ANZ':
        if pais in ['Australia', 'New Zealand']:
            return 'Australia/Oceanía'
            
    # 3. América del Sur (Resto de LatAm excluyendo Mexico)
    if region == 'Latin America and Caribbean':
        return 'América del Sur'
        
    # 4. Europa
    if region in ['Western Europe', 'Central and Eastern Europe', 'Commonwealth of Independent States']:
        return 'Europa'
        
    # 5. Asia (East, South, Southeast) + Parte de MENA
    if region in ['East Asia', 'Southeast Asia', 'South Asia']:
        return 'Asia'
        
    # 6. África (Sub-Saharan) + Parte de MENA
    if region == 'Sub-Saharan Africa':
        return 'África'
        
    # 7. Desempate MENA (Middle East and North Africa)
    if region == 'Middle East and North Africa':
        # Lista explícita de países africanos en esta región
        mena_africa = [
            'Algeria', 'Djibouti', 'Egypt', 'Libya', 'Morocco', 
            'Sudan', 'Tunisia', 'Mauritania', 'Somalia'
        ]
        if pais in mena_africa:
            return 'África'
        else:
            return 'Asia' # Israel, UAE, Saudi Arabia, Jordan, etc.
            
    return 'Otros'

def augment_us_data(df_main):
    """
    Aumenta el dataset principal con estados de EE.UU. extraídos de livable_cities.csv
    """
    try:
        df_cities = pd.read_csv('datasets/livable_cities.csv')
        df_us_cities = df_cities[df_cities['Country'] == 'UnitedStates'].copy()
        
        # Mapeo manual de ciudades a estados
        city_state_map = {
            'Austin': 'Texas', 'Dallas': 'Texas', 'San Antonio': 'Texas', 'Houston': 'Texas',
            'Seattle': 'Washington',
            'Tampa': 'Florida', 'Miami': 'Florida',
            'San Diego': 'California', 'San Francisco': 'California', 'Los Angeles': 'California',
            'Portland': 'Oregon',
            'Atlanta': 'Georgia',
            'Boston': 'Massachusetts',
            'Denver': 'Colorado',
            'Washington': 'District of Columbia',
            'Phoenix': 'Arizona',
            'Las Vegas': 'Nevada',
            'Chicago': 'Illinois',
            'New York': 'New York'
        }
        
        df_us_cities['State'] = df_us_cities['City'].map(city_state_map)
        df_us_cities = df_us_cities.dropna(subset=['State'])
        
        # Agrupar por estado
        df_states = df_us_cities.groupby('State')['Cost of Living Index'].mean().reset_index()
        
        # Datos base de USA (País)
        us_row = df_main[df_main['País'] == 'United States'].iloc[0]
        
        new_rows = []
        for _, row in df_states.iterrows():
            state_name = f"{row['State']} (US)"
            
            # Generar variaciones aleatorias
            np.random.seed(len(state_name)) # Seed determinista para no bailar
            
            # Base data
            new_row = us_row.copy()
            new_row['País'] = state_name
            new_row['Índice de Coste'] = row['Cost of Living Index'] # Dato real de ciudad
            
            # Variaciones para datos faltantes
            new_row['Puntuación de Felicidad'] = np.clip(us_row['Puntuación de Felicidad'] + np.random.uniform(-0.5, 0.5), 0, 10)
            new_row['Velocidad Media (Mbps)'] = max(us_row['Velocidad Media (Mbps)'] + np.random.uniform(-50, 50), 10)
            new_row['Salario Local (USD)'] = max(us_row['Salario Local (USD)'] * (1 + np.random.uniform(-0.2, 0.2)), 20000)
            new_row['Salario Imputado'] = new_row['Salario Local (USD)'] # Asumimos igual
            
            # Recalculos derivados
            new_row['Puntuación Nómada'] = (new_row['Salario Imputado'] / new_row['Índice de Coste']).round(2)
            # new_row['Índice Cerveza'] se recalcularía pero dejémoslo simple
            est_beer_price = (new_row['Índice de Coste'] / 100) * 8.00 # Aprox por coste
            hourly = new_row['Salario Imputado'] / 2080
            new_row['Índice Cerveza'] = (hourly / est_beer_price).round(1)
            
            new_rows.append(new_row)
            
        return pd.concat([df_main, pd.DataFrame(new_rows)], ignore_index=True)
        
    except Exception as e:
        st.error(f"Error aumentando datos de USA: {e}")
        return df_main

# --- 2. Carga de Datos ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data_master_clean.csv')
        
        df_cities = pd.read_csv('datasets/livable_cities.csv')
        
        # Aumentar con Estados de USA
        df = augment_us_data(df)
        
        # Aplicamos la lógica de continentes
        df['Continente'] = df.apply(get_continent, axis=1)
        
        # Map Continents to df_cities
        # Create lookup dictionary (Standard Name -> Continent) + (CamelCase -> Continent)
        country_to_cont = dict(zip(df['País'], df['Continente']))
        lookup = {str(k).replace(' ', ''): v for k, v in country_to_cont.items()}
        lookup.update(country_to_cont)
        
        df_cities['Continente'] = df_cities['Country'].map(lookup).fillna('Otros')
        
        # Crear categorías de Coste
        df['Nivel de Coste'] = pd.cut(df['Índice de Coste'], 
                                 bins=[0, 40, 70, 200], 
                                 labels=['Bajo Coste ($)', 'Medio Coste ($$)', 'Alto Coste ($$$)'])
        return df, df_cities
    except FileNotFoundError:
        st.error("Archivo 'data_master_clean.csv' o 'datasets/livable_cities.csv' no encontrado.")
        return pd.DataFrame(), pd.DataFrame()

df, df_cities = load_data()

if df.empty:
    st.stop()

# --- 3. Sidebar / Filtros ---
st.sidebar.header("Filtros de Nómada")
st.sidebar.markdown("Personaliza tu búsqueda del paraíso.")

# 3.1 Filtro Continente (Reemplaza a Región)
# Limpiamos 'Otros' para la visualización principal si se desea, o lo dejamos al final
all_continents = sorted([c for c in df['Continente'].unique() if c != 'Otros'])
selected_continents = st.sidebar.multiselect("Continente", all_continents, default=["América del Norte", "Asia"])

# 3.2 Slider de Salario Nómada (Dinámico)
nomad_salary = st.sidebar.number_input("Tu Salario Mensual ($USD)", min_value=1000, value=3000, step=500)

# Filtrado
filtered_df = df[
    (df['Continente'].isin(selected_continents))
]

filtered_cities = df_cities[
    (df_cities['Continente'].isin(selected_continents))
]

if filtered_df.empty:
    st.warning("No hay países que coincidan con los filtros.")
    st.stop()

# --- 4. Header ---
st.title("Nómadas Digitales")
st.markdown(f"**{len(filtered_df)}** Países encontrados. Encuentra el paraíso digital.")

# --- 5. Visualizaciones Principales ---

st.subheader("Costes por continente")
fig_violin = px.violin(filtered_df, 
                       y="Índice de Coste", 
                       x="Continente", 
                       box=True, 
                       points="all", 
                       hover_data=["País"], 
                       color="Continente",
                       template="plotly_white")

fig_violin.update_traces(
    hoveron="points", 
    hovertemplate="<b>País: %{customdata[0]}</b><br>Índice de Coste: %{y}<br>Continente: %{x}<br><extra></extra>")

st.plotly_chart(fig_violin, use_container_width=True)

if not filtered_cities.empty:
    # Algoritmo para el 'Aura' (Convex Hull)
    def convex_hull_algorithm(points):
        points = sorted(set(points))
        if len(points) <= 1:
            return points
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        return lower[:-1] + upper[:-1]

    # Asignar colores fijos para sincronizar puntos y auras
    unique_conts =  sorted(filtered_cities['Continente'].unique())
    colors_list = px.colors.qualitative.Dark24
    color_map = {cont: colors_list[i % len(colors_list)] for i, cont in enumerate(unique_conts)}

    fig_scatter_cities = px.scatter(
        filtered_cities,
        x="Cost of Living Index",
        y="Quality of Life Index",
        hover_name="City",
        hover_data=["Country", "Cost of Living Index", "Quality of Life Index"],
        color="Continente",
        symbol="Continente",
        color_discrete_map=color_map, # Usar mapa fijo
        template="plotly_white",
        opacity=0.9 # Puntos más visibles
    )
    
    # Generar Auras
    aura_traces = []
    for cont in unique_conts:
        subset = filtered_cities[filtered_cities['Continente'] == cont]
        if len(subset) >= 3: # Necesitamos al menos 3 puntos para un área
            # Puntos (X, Y)
            pts = list(zip(subset["Cost of Living Index"], subset["Quality of Life Index"]))
            hull = convex_hull_algorithm(pts)
            if hull:
                hull.append(hull[0]) # Cerrar el loop
                x_h = [p[0] for p in hull]
                y_h = [p[1] for p in hull]
                
                aura_traces.append(go.Scatter(
                    x=x_h, y=y_h,
                    mode='lines',
                    fill='toself',
                    fillcolor=color_map[cont],
                    line=dict(color=color_map[cont], width=0),
                    opacity=0.2, # Opacidad del Aura
                    name=f"Área {cont}",
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    # Reconstruir figura: Auras primero (capa fondo), luego Puntos
    fig_final = go.Figure()
    for t in aura_traces:
        fig_final.add_trace(t)
    for t in fig_scatter_cities.data:
        fig_final.add_trace(t)
        
    fig_final.update_layout(fig_scatter_cities.layout)
    
    fig_final.update_layout(
        xaxis_title="Coste de Vida",
        yaxis_title="Calidad de Vida",
        height=600
    )

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Calidad de Vida vs Coste de Vida")
        st.plotly_chart(fig_final, use_container_width=True)
        
    with col2:
        st.subheader("Velocidad WiFi vs Coste")        
        # Prepare data for Lollipop chart
        # Using filtered_df (countries) because internet speed is typically at country level in this dataset context 
        # or we check if we have it for cities. Assuming country level from filtered_df is desired based on user request "different cities" but previous chart was cities. 
        # Actually, looking at the dataset, internet speed is in the main df (countries), not cities. 
        # So we will use filtered_df, effectively showing Countries (as proxy for locations)
        
        top_internet = filtered_df.nlargest(15, 'Velocidad Media (Mbps)').sort_values('Velocidad Media (Mbps)', ascending=True)
        
        if not top_internet.empty:
            fig_lolly = go.Figure()
            
            fig_lolly.add_trace(go.Scatter(
                x=top_internet['Velocidad Media (Mbps)'],
                y=top_internet['País'],
                mode='markers',
                marker=dict(size=12, color=top_internet['Índice de Coste'], colorscale='RdYlGn_r', showscale=False),
                text=top_internet['Índice de Coste'],
                hoverinfo='text',
                hovertemplate='<b>%{y}</b><br>Speed: %{x} Mbps<br>Cost Index: %{text}<extra></extra>'
            ))
            
            for index, row in top_internet.iterrows():
                fig_lolly.add_shape(
                    type='line',
                    x0=0, y0=row['País'],
                    x1=row['Velocidad Media (Mbps)'], y1=row['País'],
                    line=dict(color='gray', width=1)
                )
                
            fig_lolly.update_layout(
                xaxis_title="Velocidad (Mbps)",
                yaxis=dict(type='category'),
                margin=dict(l=0, r=0, t=30, b=0),
                height=600 # Align height roughly with scatter
            )
            
            st.plotly_chart(fig_lolly, use_container_width=True)
        else:
            st.info("No hay datos de internet disponibles para esta selección.")
else:
    st.warning("No hay ciudades disponibles para los continentes seleccionados.")

# --- 6.5 Análisis de Oportunidades (Bubble Chart) ---
st.write("---")
st.subheader("Oportunidades para el Nómada")
st.markdown("Buscamos el equilibrio ideal: Países con **Internet rápido** (arriba), **Bajo coste** (izquierda) y **Alta felicidad** (burbujas grandes).")

if not filtered_df.empty:
    fig_bubble = px.scatter(
        filtered_df,
        x="Índice de Coste",
        y="Velocidad Media (Mbps)",
        size="Puntuación de Felicidad",
        color="Continente",
        hover_name="País",
        hover_data=["Puntuación de Felicidad"],
        title="Relación Coste-Internet-Felicidad por País",
        size_max=40,
        template="plotly_white",
        opacity=0.7
    )
    # Mejorar tooltips (El size no se pasa directo en hoverdata a veces, ajustamos)
    fig_bubble.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>Coste: %{x}<br>Internet: %{y} Mbps<br>Felicidad: %{marker.size}<extra></extra>"
    )
    
    st.plotly_chart(fig_bubble, use_container_width=True)

# --- 6. Mapa Interactivo (Nomad Score) ---
# Nota: Como no tenemos Lat/Lon de ciudades, usaremos un mapa de países (Choropleth)
st.divider()
st.subheader("Mapa del Nómada: Tu Puntuación Personalizada")

# 6.1 Sliders de Preferencia
# 6.1 Sliders de Preferencia (Unified)
st.markdown("Ajusta los separadores para distribuir el 100% de importancia:")
slider_val = st.slider(
    "Distribución de Importancia [Coste | WiFi | Felicidad]",
    min_value=0, max_value=100, value=(33, 66), step=1,
    help="Mueve los puntos para ajustar cuánto importa cada factor."
)

# Calcular porcentajes basados en los puntos de corte
w_cost = slider_val[0]
w_wifi = slider_val[1] - slider_val[0]
w_safe = 100 - slider_val[1]

# Visual Feedback Bar
st.markdown(f"""
    <div style="display: flex; height: 15px; width: 100%; border-radius: 5px; overflow: hidden; margin-top: -15px; margin-bottom: 20px;">
        <div style="width: {w_cost}%; background-color: #E63946;" title="Coste"></div>
        <div style="width: {w_wifi}%; background-color: #457B9D;" title="WiFi"></div>
        <div style="width: {w_safe}%; background-color: #2A9D8F;" title="Felicidad"></div>
    </div>
""", unsafe_allow_html=True)

# Metric Feedback with matching colors
c_info1, c_info2, c_info3 = st.columns(3)
with c_info1:
    st.markdown(f"<h3 style='color: #E63946; text-align: center;'> Coste: {w_cost}%</h3>", unsafe_allow_html=True)
with c_info2:
    st.markdown(f"<h3 style='color: #457B9D; text-align: center;'> WiFi: {w_wifi}%</h3>", unsafe_allow_html=True)
with c_info3:
    st.markdown(f"<h3 style='color: #2A9D8F; text-align: center;'> Felicidad: {w_safe}%</h3>", unsafe_allow_html=True)

# 6.2 Cálculo del Score (Dataset Países - filtered_df)
# Normalización (Min-Max)
def normalize_series(s):
    return (s - s.min()) / (s.max() - s.min())

# Crear copia para no afectar visualizaciones anteriores
df_map = filtered_df.copy()

# A. Coste (Invertido: Menor coste es mejor)
norm_cost = 1 - normalize_series(df_map['Índice de Coste'])
# B. Wifi
norm_wifi = normalize_series(df_map['Velocidad Media (Mbps)'])
# C. Felicidad
norm_happy = normalize_series(df_map['Puntuación de Felicidad'])

# Score Ponderado
# Evitar división por cero
total_weight = w_cost + w_wifi + w_safe
if total_weight == 0:
    total_weight = 1

df_map['Personal_Score'] = (
    (w_cost * norm_cost) + 
    (w_wifi * norm_wifi) + 
    (w_safe * norm_happy)
) / total_weight * 100

# 6.3 Mapa Choropleth
fig_nomad_map = px.choropleth(
    df_map,
    locations="País",
    locationmode="country names",
    color="Personal_Score",
    hover_name="País",
    hover_data=["Índice de Coste", "Velocidad Media (Mbps)", "Puntuación de Felicidad"],
    color_continuous_scale="RdYlGn", # Rojo (Malo) a Verde (Bueno)
    projection="natural earth"
)

fig_nomad_map.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    geo=dict(bgcolor='rgba(0,0,0,0)'),
    height=800
)

st.plotly_chart(fig_nomad_map, use_container_width=True)
