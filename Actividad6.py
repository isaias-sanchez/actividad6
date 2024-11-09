import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_and_preprocess_data(file_path):
    """
    Loads and preprocesses customer data
    """
    df = pd.read_excel(file_path)
    print(f"\nCantidad inicial de clientes: {len(df)}")
    
    # calcular gasto total
    gasto_columns = ['gastos_anual_tecnologia', 'gastos_anual_ropa', 
                     'gastos_anual_comida', 'gastos_anual_belleza', 
                     'gastos_anual_aseo']
    
    df['Gasto_Total'] = df[gasto_columns].sum(axis=1)
    
    # Conversión de frecuencia a valor numerico 
    frequency_map = {'Alta': 3, 'Media': 2, 'Baja': 1}
    df['Frecuencia_Numerica'] = df['frecuencia_de_compra'].map(frequency_map)
    
    return df

def prepare_features(df):
    """
    Prepares features for clustering
    """
    #selección de caracteristicas a segmentar
    features = [
        'Edad',
        'Gasto_Total',
        'Frecuencia_Numerica',
        'Puntuación_de_lealtad',
        'Número_Comentarios'
    ]
    
    X = df[features].copy()
    
    # manejo de outliers
    for col in X.columns:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        X.loc[:, col] = X[col].clip(lower_bound, upper_bound)
    
    # escalar caracteristicas y estandarización 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    #matriz con caracteristicas normalizadas
    return X_scaled, features

#busqueda del numero optimo de clusters
def find_optimal_clusters(X, max_clusters=10):
    """
    Finds optimal number of clusters using silhouette method
    """
    #aplicación de puntuación de silhouette
    silhouette_scores = []
    
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)
    
    #Valor optimo de puntuación de silhouette
    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
    return optimal_k

def analyze_and_name_segments(df, segments):
    """
    Analyzes and names segments based on their characteristics
    """
    df['Segmento'] = segments
    segment_profiles = {}
    
    for segment in range(len(np.unique(segments))):
        segment_data = df[df['Segmento'] == segment]
        
        profile = {
            'tamaño': len(segment_data),
            'porcentaje': (len(segment_data) / len(df)) * 100,
            'edad_media': segment_data['Edad'].mean(),
            'gasto_total_medio': segment_data['Gasto_Total'].mean(),
            'frecuencia_media': segment_data['Frecuencia_Numerica'].mean(),
            'lealtad_media': segment_data['Puntuación_de_lealtad'].mean(),
            'comentarios_medio': segment_data['Número_Comentarios'].mean(),
            'red_social_preferida': segment_data['Red_Social_Favorita'].mode().iloc[0],
            'region_principal': segment_data['Región'].mode().iloc[0],
            'categorias_principales': get_top_categories(segment_data)
        }
        
        nombre = name_segment(profile, df)
        segment_profiles[nombre] = profile
        
    return segment_profiles

def get_top_categories(data):
    """
    Gets top spending categories for each segment
    """
    categories = ['gastos_anual_tecnologia', 'gastos_anual_ropa', 
                 'gastos_anual_comida', 'gastos_anual_belleza', 
                 'gastos_anual_aseo']
    
    category_means = data[categories].mean()
    return category_means.nlargest(2).index.tolist()

def name_segment(profile, df):
    """
    Assigns meaningful name to segment based on characteristics
    """
    if profile['lealtad_media'] > df['Puntuación_de_lealtad'].mean() * 1.2:
        if profile['gasto_total_medio'] > df['Gasto_Total'].mean() * 1.2:
            return "Clientes Premium"
        else:
            return "Clientes Leales"
    elif profile['frecuencia_media'] > df['Frecuencia_Numerica'].mean() * 1.2:
        return "Compradores Frecuentes"
    elif profile['comentarios_medio'] > df['Número_Comentarios'].mean() * 1.2:
        return "Clientes Sociales"
    elif profile['gasto_total_medio'] > df['Gasto_Total'].mean() * 1.2:
        return "Grandes Compradores"
    else:
        return "Compradores Estándar"

def generate_marketing_recommendations(profile):
    """
    Generates marketing recommendations for each segment
    """
    recommendations = {
        "Clientes Premium": {
            "estrategias": [
                "Programa VIP personalizado",
                "Acceso anticipado a nuevos productos",
                "Servicio al cliente dedicado",
                "Eventos exclusivos"
            ],
            "canales": [
                "Comunicación personalizada multicanal",
                "Atención preferencial en tienda"
            ],
            "ofertas": [
                "Descuentos exclusivos",
                "Regalos especiales",
                "Experiencias premium"
            ]
        },
        "Clientes Leales": {
            "estrategias": [
                "Programa de fidelización avanzado",
                "Comunicación regular personalizada",
                "Beneficios por antigüedad"
            ],
            "canales": [
                "Email marketing personalizado",
                "Notificaciones móviles"
            ],
            "ofertas": [
                "Descuentos por frecuencia",
                "Programa de puntos premium"
            ]
        },
        "Compradores Frecuentes": {
            "estrategias": [
                "Programa de recompensas por frecuencia",
                "Ofertas de suscripción",
                "Recordatorios personalizados"
            ],
            "canales": [
                "App móvil",
                "Email marketing frecuente"
            ],
            "ofertas": [
                "Descuentos por volumen",
                "Beneficios por suscripción"
            ]
        },
        "Clientes Sociales": {
            "estrategias": [
                "Campaña de embajadores",
                "Contenido compartible",
                "Programa de referidos"
            ],
            "canales": [
                "Redes sociales",
                "Influencer marketing",
                "Contenido generado por usuarios"
            ],
            "ofertas": [
                "Recompensas por compartir",
                "Descuentos grupales"
            ]
        },
        "Grandes Compradores": {
            "estrategias": [
                "Cross-selling premium",
                "Ofertas por categoría",
                "Beneficios por volumen"
            ],
            "canales": [
                "Email marketing personalizado",
                "Remarketing premium"
            ],
            "ofertas": [
                "Descuentos por categoría",
                "Bundles premium"
            ]
        },
        "Compradores Estándar": {
            "estrategias": [
                "Programa básico de fidelización",
                "Comunicación regular",
                "Incentivos de upgrade"
            ],
            "canales": [
                "Email marketing general",
                "Redes sociales"
            ],
            "ofertas": [
                "Promociones generales",
                "Descuentos de primera compra"
            ]
        }
    }
    
    return recommendations.get(profile, {})

def main():
    # Load and preprocess data
    df = load_and_preprocess_data("dataset.xlsx")
    
    # Prepare features
    X_scaled, features = prepare_features(df)
    
    # Find optimal number of clusters
    optimal_k = find_optimal_clusters(X_scaled)
    print(f"\nNúmero óptimo de clusters: {optimal_k}")
    
    # Perform clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    segments = kmeans.fit_predict(X_scaled)
    
    # Analyze and name segments
    segment_profiles = analyze_and_name_segments(df, segments)
    
    # Print results and recommendations
    print("\nANÁLISIS DE SEGMENTACIÓN Y RECOMENDACIONES DE MARKETING")
    print("=" * 80)
    
    for nombre, profile in segment_profiles.items():
        print(f"\n{nombre.upper()}")
        print("-" * 50)
        print(f"Tamaño: {profile['tamaño']} clientes ({profile['porcentaje']:.1f}%)")
        print(f"Edad media: {profile['edad_media']:.1f} años")
        print(f"Gasto total medio: ${profile['gasto_total_medio']:,.2f}")
        print(f"Frecuencia media: {profile['frecuencia_media']:.1f}")
        print(f"Puntuación de lealtad: {profile['lealtad_media']:.1f}/5")
        print(f"Promedio de comentarios: {profile['comentarios_medio']:.1f}")
        print(f"Red social preferida: {profile['red_social_preferida']}")
        print(f"Región principal: {profile['region_principal']}")
        print(f"Categorías principales: {', '.join(c.replace('gastos_anual_', '') for c in profile['categorias_principales'])}")
        
        print("\nRECOMENDACIONES DE MARKETING")
        recommendations = generate_marketing_recommendations(nombre)
        
        print("\nEstrategias:")
        for strategy in recommendations.get("estrategias", []):
            print(f"- {strategy}")
            
        print("\nCanales de comunicación:")
        for channel in recommendations.get("canales", []):
            print(f"- {channel}")
            
        print("\nOfertas recomendadas:")
        for offer in recommendations.get("ofertas", []):
            print(f"- {offer}")
    
    # Visualize segment distribution
    plt.figure(figsize=(12, 6))
    segment_sizes = df['Segmento'].value_counts()
    segment_sizes.plot(kind='bar')
    plt.title('Distribución de Clientes por Segmento')
    plt.xlabel('Segmento')
    plt.ylabel('Número de Clientes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("distribucion_segmentos.png")
    plt.close()

if __name__ == "_main_":
    main()