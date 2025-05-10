import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración inicial
st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('spain_menorca_procesado_outliers (2) (1).csv')
    
    # Limpiar y convertir la columna 'price' a float
    df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
    
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    numeric_cols = numeric_df.columns.tolist()
    
    text_df = df.select_dtypes(include=['object'])
    text_cols = text_df.columns.tolist()
    
    # Extraemos categorías únicas para algunas columnas relevantes
    host_response_time_cats = df['host_response_time'].unique()
    host_is_superhost_cats = df['host_is_superhost'].unique()
    room_type_cats = df['room_type'].unique()
    
    return df, numeric_cols, text_cols, host_response_time_cats, host_is_superhost_cats, room_type_cats, numeric_df

df, numeric_cols, text_cols, host_response_time_cats, host_is_superhost_cats, room_type_cats, numeric_df = load_data()

# Selector de vista en la barra lateral
view = st.sidebar.selectbox(label='Seleccionar vista', options=['Vista General', 'Análisis de Precios', 'Regresión Lineal', 'Regresión Logística'])

if view == 'Vista General':
    st.title('Airbnb Menorca España - Dashboard Analítico')
    st.header('Vista General de los Datos')
    
    # Barra lateral
    st.sidebar.title('Opciones de Filtrado')
    st.sidebar.header('Filtros de Datos')
    
    # Checkbox para mostrar dataset
    check_box = st.sidebar.checkbox(label='Mostrar dataset completo')
    
    if check_box:
        # Seleccionar solo las columnas especificadas
        selected_columns = [
            'last_scraped', 'host_name', 'host_since', 'host_location', 'host_response_time', 
            'host_response_rate', 'host_acceptance_rate', 'host_is_superhost', 
            'host_has_profile_pic', 'host_identity_verified', 'property_type', 'room_type', 
            'bathrooms_text', 'price', 'has_availability', 'calendar_last_scraped', 
            'first_review', 'last_review', 'license', 'instant_bookable', 'id', 'host_id', 
            'host_listings_count', 'host_total_listings_count', 'accommodates', 'bathrooms', 
            'bedrooms', 'beds', 'minimum_nights', 'maximum_nights', 'minimum_minimum_nights', 
            'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights', 
            'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'availability_30', 
            'availability_90', 'number_of_reviews', 'number_of_reviews_ltm', 
            'number_of_reviews_l30d', 'review_scores_rating', 'review_scores_accuracy', 
            'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_location', 
            'calculated_host_listings_count_entire_homes', 
            'calculated_host_listings_count_private_rooms', 
            'calculated_host_listings_count_shared_rooms', 'reviews_per_month'
        ]
        st.write(df[selected_columns])
        st.write("Estadísticas descriptivas:", df[selected_columns].describe())
    
    # Filtros interactivos y otros elementos de la vista general
    st.sidebar.subheader('Filtros Interactivos')
    price_range = st.sidebar.slider(
        'Rango de Precio (€)',
        float(df['price'].min()),
        float(df['price'].max()),
        (float(df['price'].min()), float(df['price'].max()))
    )
    
    superhost_filter = st.sidebar.selectbox(
        '¿Es Superhost?',
        options=['Todos', 'Sí', 'No']
    )
    
    room_type_filter = st.sidebar.multiselect(
        'Tipo de Alojamiento',
        options=room_type_cats,
        default=room_type_cats
    )
    
    # Aplicar filtros
    filtered_df = df.copy()
    filtered_df = filtered_df[(filtered_df['price'] >= price_range[0]) & (filtered_df['price'] <= price_range[1])]

    if superhost_filter != 'Todos':
        # Mapear 'Sí' a 't' y 'No' a 'f'
        superhost_value = 't' if superhost_filter == 'Sí' else 'f'
        filtered_df = filtered_df[filtered_df['host_is_superhost'] == superhost_value]

    if room_type_filter:
        filtered_df = filtered_df[filtered_df['room_type'].isin(room_type_filter)]
    
    # Mostrar métricas clave
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Propiedades", len(filtered_df))
    col2.metric("Precio Promedio", f"€{filtered_df['price'].mean():.2f}")
    col3.metric("Rating Promedio", f"{filtered_df['review_scores_rating'].mean():.2f}")
    
    # Gráfico: Distribución de precios
    st.subheader('Distribución de Precios')
    fig1 = px.histogram(filtered_df, x='price', nbins=50, title='Distribución de Precios de Alojamientos')
    st.plotly_chart(fig1, use_container_width=True)

elif view == 'Análisis de Precios':
    st.title('Airbnb Menorca - Análisis de Precios')
    
    # Widgets para análisis de precios
    st.sidebar.header('Opciones de Análisis de Precios')
    x_axis = st.sidebar.selectbox(
        'Variable para eje X',
        options=['room_type', 'host_is_superhost', 'host_response_time', 'accommodates', 'bathrooms'],
        index=0
    )
    
    color_by = st.sidebar.selectbox(
        'Categoría de agrupación',
        options=['room_type', 'host_is_superhost', 'host_response_time', 'beds'],
        index=0
    )

    relevant_features = ['accommodates', 'bathrooms', 'beds', 'review_scores_rating']

    property_feature = st.sidebar.selectbox(
        'Seleccionar característica de propiedad',
        options=relevant_features,
        index=0
    )
    
    # Gráfico de caja: Precio vs variable seleccionada
    st.subheader('Relación entre Precio y Otras Variables')
    fig = px.box(df, x=x_axis, y='price', color=color_by, title=f'Distribución de Precios por {x_axis}')
    st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap de correlación
    st.subheader('Correlación entre Variables Numéricas')
    selected_columns = ['accommodates', 'bathrooms', 'beds', 'review_scores_rating']
    numeric_df_corr = df[selected_columns]
    corr_matrix = numeric_df_corr.corr()
    fig2 = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Heatmap de Correlación (Variables Seleccionadas)")
    st.plotly_chart(fig2, use_container_width=True)
    
    # Gráfico de dispersión: Precio vs característica seleccionada
    st.subheader(f'Relación entre {property_feature} y Precio')
    fig3 = px.scatter(df, x=property_feature, y='price', color='room_type',
                     trendline="lowess", 
                     title=f'Precio vs {property_feature} por tipo de alojamiento')
    st.plotly_chart(fig3, use_container_width=True)
    
# ... (código anterior igual hasta la sección de regresión lineal)

elif view == 'Regresión Lineal':
    st.title('Modelo de Regresión Lineal')
    st.write('Selecciona el tipo de regresión lineal que deseas realizar.')

    # Preprocesamiento básico para eliminar valores nulos
    df_clean = df.dropna(subset=['price'])
    
    regression_type = st.radio(
        'Tipo de Regresión Lineal:',
        options=['Simple', 'Múltiple']
    )

    if regression_type == 'Simple':
        st.sidebar.header('Regresión Lineal Simple')
        
        predefined_features = [
            'accommodates', 'bathrooms', 'beds', 'review_scores_rating', 
            'minimum_nights', 'number_of_reviews', 'availability_30'
        ]
        
        selected_feature = st.sidebar.selectbox(
            'Variable independiente:',
            predefined_features
        )

        if selected_feature:
            # Eliminar filas con valores nulos en las columnas seleccionadas
            temp_df = df_clean[[selected_feature, 'price']].dropna()
            
            X = temp_df[[selected_feature]]
            y = temp_df['price']

            # Solo codificar si es categórico
            if X[selected_feature].dtype == 'object':
                X = pd.get_dummies(X, drop_first=True)
            
            # Verificar que haya datos suficientes
            if len(X) > 0:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                # Escalar solo si es numérico
                if X[selected_feature].dtype in ['int64', 'float64']:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Mostrar resultados
                st.subheader('Resultados del Modelo')
                col1, col2 = st.columns(2)
                col1.metric("R²", f"{r2_score(y_test, y_pred):.4f}")
                col2.metric("MSE", f"{mean_squared_error(y_test, y_pred):.2f}")
                
                # Gráfico de dispersión con línea de regresión
                fig = px.scatter(
                    x=y_test, y=y_pred, 
                    labels={'x': 'Valor Real', 'y': 'Predicción'},
                    title='Valores Reales vs Predicciones'
                )
                #fig.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(), 
                 #            line=dict(color="red", dash="dash"))
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar coeficientes
                if hasattr(model, 'coef_'):
                    st.write("Coeficiente:", model.coef_[0])
                st.write("Intercepto:", model.intercept_)
            else:
                st.error("No hay suficientes datos válidos para este análisis.")

    elif regression_type == 'Múltiple':
        st.sidebar.header('Regresión Lineal Múltiple')
        
        predefined_features = [
            'accommodates', 'bathrooms', 'beds', 'review_scores_rating',
            'minimum_nights', 'number_of_reviews', 'availability_30',
            'room_type', 'host_is_superhost'
        ]
        
        selected_features = st.sidebar.multiselect(
            'Variables independientes:',
            predefined_features,
            default=['accommodates', 'bathrooms']
        )

        if len(selected_features) > 0:
            # Eliminar filas con valores nulos
            temp_df = df_clean[selected_features + ['price']].dropna()
            
            X = temp_df[selected_features]
            y = temp_df['price']

            # Codificar variables categóricas
            X_encoded = pd.get_dummies(X, drop_first=True)
            
            if len(X_encoded.columns) > 0:
                # Escalar las características numéricas
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_encoded)
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.3, random_state=42
                )
                
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Resultados
                st.subheader('Resultados del Modelo')
                col1, col2 = st.columns(2)
                col1.metric("R²", f"{r2_score(y_test, y_pred):.4f}")
                col2.metric("MSE", f"{mean_squared_error(y_test, y_pred):.2f}")
                
                # Gráfico
                fig = px.scatter(
                    x=y_test, y=y_pred, 
                    labels={'x': 'Valor Real', 'y': 'Predicción'},
                    title='Valores Reales vs Predicciones'
                )
                #fig.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(),
                            # line=dict(color="red", dash="dash"))
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar importancia de características
                st.subheader('Importancia de las Variables')
                if hasattr(model, 'coef_'):
                    coef_df = pd.DataFrame({
                        'Variable': X_encoded.columns,
                        'Coeficiente': model.coef_
                    }).sort_values('Coeficiente', ascending=False)
                    st.dataframe(coef_df)
            else:
                st.error("No se pudo codificar las variables correctamente.")

# ... (código posterior igual)
# ...código existente...

elif view == 'Regresión Logística':
    st.title('Modelo de Regresión Logística')
    st.write('Selecciona las variables independientes para predecir si el precio es alto o bajo.')

    # Crear variable objetivo binaria
    df['high_price'] = (df['price'] > df['price'].median()).astype(int)

    # Opciones de selección en la barra lateral
    st.sidebar.header('Opciones de Regresión Logística')
    predefined_features = [
        'accommodates', 'bathrooms', 'beds', 'review_scores_rating',
        'minimum_nights', 'number_of_reviews', 'availability_30',
        'room_type', 'host_is_superhost'
    ]
    selected_features = st.sidebar.multiselect(
        'Selecciona las variables independientes:',
        predefined_features,
        default=['accommodates', 'bathrooms']
    )

    if selected_features:
        # Preparar los datos
        temp_df = df[selected_features + ['high_price']].dropna()
        X = temp_df[selected_features]
        y = temp_df['high_price']

        # Codificar variables categóricas
        X_encoded = pd.get_dummies(X, drop_first=True)

        # Escalar las características
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)

        # Dividir en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        # Entrenar el modelo
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Predicciones
        y_pred = model.predict(X_test)

        # Métricas
        st.subheader('Métricas del Modelo')
        st.write(f'Precisión: {accuracy_score(y_test, y_pred):.2f}')
        st.write(f'Precisión (Positive Predictive Value): {precision_score(y_test, y_pred):.2f}')
        st.write(f'Recuperación (Sensibilidad): {recall_score(y_test, y_pred):.2f}')

        # Matriz de confusión
        st.subheader('Matriz de Confusión')
        cm = confusion_matrix(y_test, y_pred)
        fig5, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Valor Real')
        st.pyplot(fig5)
    else:
        st.error("Por favor selecciona al menos una variable independiente.")

# ...código existente...
