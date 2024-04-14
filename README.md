# Machine-Learning-na-Pr-tica-no-Azure-ML
BOOT CAMP MICROSFT AZURE
Este repositório contém um projeto desenvolvido utilizando o Azure Machine Learning Studio para treinar um modelo preditivo de aluguel de bicicletas. O objetivo é prever o número de aluguéis de bicicletas em um determinado dia com base em características sazonais e meteorológicas utilizando aprendizado de máquina automatizado (AutoML).

Dados Utilizados
Os dados utilizados neste projeto são derivados da Capital Bikeshare e estão em conformidade com o contrato de licença de dados publicado. O conjunto de dados foi criado e configurado com as seguintes especificações:

Tipo de dados: Tabular
Fonte: https://aka.ms/bike-rentals
Formato do arquivo: Delimitado por vírgula
Codificação: UTF-8
Configuração do Projeto
Criação do Trabalho de ML Automatizado
No Azure Machine Learning Studio, foi criado um novo trabalho de ML automatizado com as seguintes configurações:

Nome do trabalho: mslearn-bike-automl
Tipo de tarefa: Regressão
Conjunto de dados: aluguel de bicicletas
Coluna de destino: Aluguéis
Métrica primária: raiz do erro quadrático médio normalizado
Modelos permitidos: RandomForest e LightGBM
Limites e Validações
Máximo de testes: 3
Tempo limite de iteração: 15 minutos
Tipo de validação: divisão de validação de trem com 10% dos dados para validação
Avaliação do Modelo
Após o término do treinamento, o melhor modelo foi avaliado usando as métricas disponíveis e gráficos de desempenho como o gráfico de resíduos e predito vs. real.

Implantação do Modelo
O melhor modelo foi implantado como um serviço web no Azure com a seguinte configuração:

Nome: prever-aluguéis
Tipo de computação: Instância de Contêiner do Azure
Habilitar autenticação: Sim
Teste do Serviço Implantado
O serviço web implantado foi testado utilizando o seguinte JSON de entrada para prever o número de aluguéis:

json
Copy code
{
  "Inputs": { 
    "data": [
      {
        "day": 1,
        "mnth": 1,   
        "year": 2022,
        "season": 2,
        "holiday": 0,
        "weekday": 1,
        "workingday": 1,
        "weathersit": 2, 
        "temp": 0.3, 
        "atemp": 0.3,
        "hum": 0.3,
        "windspeed": 0.3 
      }
    ]    
  },   
  "GlobalParameters": 1.0
}
Os resultados do teste confirmaram a precisão do modelo com a previsão do número de aluguéis.

