from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Função para carregar e preparar dados
def load_and_prepare_3Ddata(dataPath, inicialRow, SampleSize):
    # Carregar e preparar dados
    df = pd.read_csv(dataPath, skiprows= inicialRow).head(SampleSize)

    # Correção de dados
    media_idade = df['age'][df['age'] >= 0].mean()
    df.loc[df['age'] < 0, 'age'] = media_idade
    df['age'] = df['age'].fillna(media_idade)

    media_renda = df['income'][df['income'] >= 0].mean()
    df.loc[df['income'] < 0, 'income'] = media_renda
    df['income'] = df['income'].fillna(media_renda)

    media_divida = df['loan'][df['loan'] >= 0].mean()
    df.loc[df['loan'] < 0, 'loan'] = media_divida
    df['loan'] = df['loan'].fillna(media_divida)

    # Escalonamento
    scaler = MinMaxScaler()
    df[['age', 'income', 'loan']] = scaler.fit_transform(df[['age', 'income', 'loan']])
    
    return df[['age', 'income', 'loan']]

def Ajuste_kmeans(data, max_weight, max_clusters):
    best_score = -1  # Inicia com um valor muito baixo
    best_weights = None
    best_data = None
    best_n_clusters = None

    # Loop para testar todas as combinações de pesos de 1 a max_weight e número de clusters
    for age_weight in range(1, max_weight + 1):
        for income_weight in range(1, max_weight + 1):
            for loan_weight in range(1, max_weight + 1):
                # Aplica os pesos às variáveis
                data_scaled = data.copy()
                data_scaled['age'] *= age_weight
                data_scaled['income'] *= income_weight
                data_scaled['loan'] *= loan_weight

                # Normaliza os dados
                scaler = MinMaxScaler()
                data_scaled[['age', 'income', 'loan']] = scaler.fit_transform(data_scaled[['age', 'income', 'loan']])

                # Loop para testar diferentes números de clusters
                for n_clusters in range(2, max_clusters + 1):
                    # Realiza o KMeans com n_clusters
                    km = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
                    km_predict = km.fit_predict(data_scaled[['age', 'income', 'loan']])

                    # Calcula o Silhouette Score
                    score = silhouette_score(data_scaled[['age', 'income', 'loan']], km_predict)

                    # Se o score for o melhor encontrado, armazena os dados, pesos e número de clusters
                    if score > best_score:
                        best_score = score
                        best_weights = {'age': age_weight, 'income': income_weight, 'loan': loan_weight}
                        best_data = data_scaled
                        best_n_clusters = n_clusters
                        
    joblib.dump(km, 'Modelos3D/kmeans_model.pkl') 
    with open('Pesos3D/best_weights.txt', 'w') as file:
        for key, value in best_weights.items():
            file.write(f"{key}: {value}\n") 

    return best_data, best_weights, best_n_clusters

class TreinamentoKmeans:
    def __init__(self, data, n_cluster, n_init):
        self.data = data
        self.n_cluster = n_cluster
        self.n_init = n_init

    def train(self):
       # Instancia o KMeans e treina o modelo
        self.km = KMeans(n_clusters= self.n_cluster, init='k-means++', n_init = self.n_init)
        self.data['cluster'] = self.km.fit_predict(self.data[['age', 'income', 'loan']])##Criação de uma nova coluna no DataFrame referente ao cluster rotulado  

    def plot_clusters(self):
        # Configuração do gráfico 3D para exibir clusters
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Definir cores para cada cluster
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        for i in range(self.n_cluster):
            cluster_data = self.data[self.data['cluster'] == i]
            ax.scatter(cluster_data['age'], cluster_data['income'], cluster_data['loan'], 
                       c=colors[i], label=f'Cluster {i}', s=50)

        ax.set_xlabel('Idade')
        ax.set_ylabel('Renda')
        ax.set_zlabel('Dívida')
        plt.legend()
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
            y_predict = km.fit_predict(data[['age', 'income', 'loan']])

            # Calcula o Silhouette Score para o modelo atual
            score = silhouette_score(data[['age', 'income', 'loan']], y_predict)
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
data = load_and_prepare_3Ddata('credit_data.csv', inicialRow=0, SampleSize=150)

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


    

    