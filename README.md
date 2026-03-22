# Machine Learning na Pratica no Azure ML

Este repositorio deixou de ser apenas um resumo do portal e virou um pipeline real de machine learning para prever aluguel de bicicletas. O projeto agora treina, avalia e executa inferencia localmente com o dataset classico de bike rentals, mas ja inclui artefatos prontos para portar o fluxo ao Azure ML.

## O que o projeto entrega

- treino local de modelo de regressao
- comparacao entre modelos candidatos
- avaliacao com `RMSE`, `MAE` e `R²`
- script de inferencia a partir de JSON
- dataset real do exemplo de bike rentals
- arquivos base para command job no Azure ML
- testes automatizados

## Estrutura

- `src/train.py`: treino e selecao do melhor modelo
- `src/evaluate.py`: avaliacao do modelo salvo
- `src/predict.py`: inferencia com payload JSON
- `src/data_utils.py`: carregamento do dataset e definicao de features
- `azureml/command-job.yml`: job base para Azure ML
- `azureml/environment.yml`: ambiente do job
- `examples/sample-request.json`: payload de exemplo
- `data/bike-rentals/bike-data/daily-bike-share.csv`: dataset usado no pipeline

## Como executar localmente

### Instalar dependencias

```bash
pip install -r requirements.txt
```

### Treinar

```bash
python src/train.py
```

### Avaliar

```bash
python src/evaluate.py
```

### Prever

```bash
python src/predict.py
```

## Caso de uso

O modelo tenta prever a quantidade de alugueis de bicicletas com base em:

- sazonalidade
- mes e dia
- dia util e feriado
- condicao climatica
- temperatura
- umidade
- velocidade do vento

## Artefatos gerados

Depois do treino, o projeto salva em `models/`:

- `model.joblib`
- `metrics.json`

## Resultado validado localmente

No treino executado neste ambiente, o melhor modelo foi `gradient_boosting` com:

- `RMSE`: `263.871`
- `MAE`: `167.8235`
- `R²`: `0.8213`

Modelos comparados:

- `linear_regression`
- `random_forest`
- `gradient_boosting`

No payload de exemplo em [sample-request.json](C:/Users/vitor/OneDrive/Documentos/Playground/repo-azure-ml-pratica/examples/sample-request.json), a inferencia local retornou previsao de `217.3` alugueis.

## Azure ML

O projeto inclui uma base simples para subir o treino como command job no Azure ML usando:

- [command-job.yml](C:/Users/vitor/OneDrive/Documentos/Playground/repo-azure-ml-pratica/azureml/command-job.yml)
- [environment.yml](C:/Users/vitor/OneDrive/Documentos/Playground/repo-azure-ml-pratica/azureml/environment.yml)

Isso ajuda a mostrar o fluxo completo:

- validacao local
- empacotamento do treino
- migracao para o workspace gerenciado

O job YAML usa `uri_file` como entrada e `uri_folder` como saida, seguindo o estilo atual do Azure ML CLI v2.

## Referencias oficiais

Para alinhar o projeto com o estado atual da plataforma, usei como base a documentacao oficial da Microsoft sobre Azure Machine Learning:

- [What is Azure Machine Learning?](https://learn.microsoft.com/en-us/azure/machine-learning/overview-what-is-azure-machine-learning?view=azureml-api-2)
- [What is automated machine learning (AutoML)?](https://learn.microsoft.com/en-us/azure/machine-learning/concept-automated-ml?view=azureml-api-2)
- [CLI v2 command job YAML schema](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-job-command?view=azureml-api-2)
- [Online endpoints for real-time inference](https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints-online?view=azureml-api-2)

## Validacao

```bash
pytest
```

O teste cobre treino, avaliacao e inferencia ponta a ponta.

## Proximos passos

- adicionar endpoint de API para predicao
- registrar modelos por versao
- incluir tracking de experimentos
- criar deployment para online endpoint
