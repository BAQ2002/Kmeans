from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Função para carregar e preparar dados
def load_and_prepare_2Ddata(dataPath, inicialRow, SampleSize):
    # Carregar e preparar dados
    df = pd.read_csv(dataPath, skiprows=inicialRow).head(SampleSize)

    # Correção de dados
    media_idade = df['age'][df['age'] >= 0].mean()
    df.loc[df['age'] < 0, 'age'] = media_idade
    df['age'] = df['age'].fillna(media_idade)

    media_renda = df['income'][df['income'] >= 0].mean()
    df.loc[df['income'] < 0, 'income'] = media_renda
    df['income'] = df['income'].fillna(media_renda)

    # Escalonamento
    scaler = MinMaxScaler()
    df[['age', 'income']] = scaler.fit_transform(df[['age', 'income']])
    
    return df[['age', 'income']] 

def Ajuste_kmeans(data, max_weight, max_clusters):
    best_score = -1  # Inicia com um valor muito baixo
    best_weights = None
    best_data = None
    best_n_clusters = None

    # Loop para testar todas as combinações de pesos de 1 a max_weight e número de clusters
    for age_weight in range(1, max_weight + 1):
        for income_weight in range(1, max_weight + 1):
            # Aplica os pesos às variáveis
            data_scaled = data.copy()
            data_scaled['age'] *= age_weight
            data_scaled['income'] *= income_weight

            # Normaliza os dados
            scaler = MinMaxScaler()
            data_scaled[['age', 'income']] = scaler.fit_transform(data_scaled[['age', 'income']])

            # Loop para testar diferentes números de clusters
            for n_clusters in range(2, max_clusters + 1):
                # Realiza o KMeans com n_clusters
                km = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
                km_predict = km.fit_predict(data_scaled[['age', 'income']])

                # Calcula o Silhouette Score
                score = silhouette_score(data_scaled[['age', 'income']], km_predict)

                # Se o score for o melhor encontrado, armazena os dados, pesos e número de clusters
                if score > best_score:
                    best_score = score
                    best_weights = {'age': age_weight, 'income': income_weight}
                    best_data = data_scaled
                    best_n_clusters = n_clusters

    joblib.dump(km, 'Modelos2D/kmeans_model.pkl')   
    joblib.dump(best_weights, 'Pesos2D/best_weights.pkl')             
    
    return best_data, best_weights, best_n_clusters

class TreinamentoKmeans:
    def __init__(self, data, n_cluster, n_init):
        self.data = data
        self.n_cluster = n_cluster
        self.n_init = n_init

    def train(self):
        # Instancia o KMeans e treina o modelo
        self.km = KMeans(n_clusters=self.n_cluster, init='k-means++', n_init=self.n_init)
        self.data['cluster'] = self.km.fit_predict(self.data[['age', 'income']])  # Criação de uma nova coluna no DataFrame referente ao cluster rotulado

    def plot_clusters(self):
        # Configuração do gráfico 2D para exibir clusters
        plt.figure(figsize=(8, 6))
        
        # Definir cores para cada cluster
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        # Plotar os dados por cluster
        for i in range(self.n_cluster):
            cluster_data = self.data[self.data['cluster'] == i]
            plt.scatter(cluster_data['age'], cluster_data['income'], c=colors[i], label=f'Cluster {i}', s=50)

        plt.xlabel('Idade')
        plt.ylabel('Renda')
        plt.legend()
        plt.title('Clusters em 2D (Idade vs Renda)')
        plt.show()

class AvaliacaoModelo:
    def __init__(self, data, max_clusters):
        self.data = data
        self.max_clusters = max_clusters

    def plot_silhouette_scores(self):
        # Armazenar os Silhouette Scores
        silhouette_scores = []
        cluster_range = range(2, self.max_clusters)  # Quantidade de clusters a serem testados

        for n_clusters in cluster_range:
            km = KMeans(n_clusters=n_clusters)
            y_predict = km.fit_predict(self.data[['age', 'income']])  # Considera apenas 'age' e 'income'

            # Calcula o Silhouette Score para o modelo atual
            score = silhouette_score(self.data[['age', 'income']], y_predict)
            silhouette_scores.append(score)

        # Criar o gráfico de colunas
        plt.figure(figsize=(8, 6))
        plt.bar(cluster_range, silhouette_scores, color='skyblue')
        plt.xlabel('Número de Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score para diferentes números de Clusters')
        plt.xticks(cluster_range)
        plt.show()

#Uso
data = load_and_prepare_2Ddata('credit_data.csv', inicialRow=0, SampleSize=150)

# Instanciar e treinar o modelo
kmeans_model = TreinamentoKmeans(data, 3, 25)
kmeans_model.train()

# Visualizar clusters
kmeans_model.plot_clusters()

#Avaliacao do Modelo
Evaluation = AvaliacaoModelo(data, 25)
Evaluation.plot_silhouette_scores()

#Ajuste do modelo
data_scaled, best_weight, best_n_clusters = Ajuste_kmeans(data, 10, 25)

#Avaliacao do Modelo Ajustado
Evaluation = AvaliacaoModelo(data_scaled, 25)
Evaluation.plot_silhouette_scores()

# Visualizar clusters
kmeans_model.plot_clusters()
