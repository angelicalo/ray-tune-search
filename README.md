New experiment

Clonar o repositório

Acessar: 

Criar uma pasta para seu experimento no seguinte caminho:
ray-tune-search/tree/main/experiments

Nova pasta: ray-tune-search/tree/main/experiments/nova_pasta

Criar um arquivo de configurações básicas na nova_pasta:
base_config.yaml - determina estimators, extra, reducer e reducer_dataset

Criar um arquivo de configuração dos hiperparâmetros na nova_pasta
exploration_config.yaml - resources, search_space e initial_params

Executa no terminal o seguinte comando:
python hsearch_main.py --data ../data/ --cpu 1 --gpu 0.5 --max_concurrent 2 --experiment nova_pasta
com ../data/ apontando para a pasta dos datasets do experimento
os demais valores são opcionais (-cpu 1 --gpu 0.5) que possuem seus defaults
