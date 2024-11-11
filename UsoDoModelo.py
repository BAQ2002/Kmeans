from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class Uso2D:
    def __init__(self, dataPath, modelPath, weightPath, inicialRow, SampleSize):
        self.dataPath = dataPath
        self.modelPath = modelPath
        self.weightPath = weightPath
        self.inicialRow = inicialRow
        self.SampleSize = SampleSize
        
    def test_2D(self):
        df = pd.read_csv(self.dataPath, skiprows=self.inicialRow).head(self.SampleSize)

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

        weights = joblib.load(self.weightPath)
        data_test_scaled = df.copy
        data_test_scaled['age'] *= weights['age']
        data_test_scaled['income'] *= weights['income']
        scaler = MinMaxScaler()
        data_test_scaled[['age', 'income']] = scaler.fit_transform(data_test_scaled[['age', 'income']])

        test_clusters = kmeans.predict(data_test_scaled[['age', 'income']])
        data_test_scaled['cluster'] = test_clusters

        kmeans = joblib.load(self.modelPath)
    
        plt.figure(figsize=(8, 6))

        # Plotar os dados com cores diferentes para cada cluster
        plt.scatter(data_test_scaled['age'], data_test_scaled['income'], c=data_test_scaled['cluster'], cmap='viridis', marker='o')

        # Adicionar título e rótulos
        plt.title('Clusters de Teste - KMeans')
        plt.xlabel('Idade (age)')
        plt.ylabel('Renda (income)')

        # Adicionar a legenda
        plt.colorbar(label='Cluster')

        # Exibir o gráfico
        plt.show()

teste = Uso2D('credit_data.csv', 'Modelos2D/kmeans_model.pkl', 'Pesos2D/best_weights.pkl', 0, 50)
teste.test_2D()
