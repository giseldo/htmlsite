# Regressão Linear

A Regressão linear é uma técnica estatística que tenta modelar a relação entre duas ou mais variáveis usando uma linha reta. É uma ferramenta poderosa para prever valores de uma variável dependente (a que você está tentando prever) com base nos valores de uma ou mais variáveis independentes (aquelas que você usa para fazer a previsão).

Imagine que você quer saber se existe uma relação entre o número de horas que um estudante estuda e a nota que ele tira em um teste. Você pode usar a regressão linear para verificar se existe uma relação linear entre essas duas variáveis e prever a nota do estudante com base no número de horas que ele estuda.

O código a seguir cria os dados necessários e apresenta um gráfico de pontos da relação entre as duas variáveis, $x$ a variável dependente e $y$ a variável independente.

Primeiro vamos instalar as dependências necessárias.

```
!pip install numpy matplotlib statsmodels seaborn
```

Em seguida o código Python

```Python
# bibliotecas utilizadas
import numpy as np
import matplotlib.pyplot as plt

# criação do conjunto de dados
np.random.seed(42)
x = np.random.rand(100)
y = 2 * x + 1 + np.random.normal(0, 0.2, 100)

# gráfico
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gráfico de Dispersão')
plt.show()
```
![alt text](image-1.png)

Os dados foram gerados aleatóriamente de forma que exista uma relação linear entre as variaveis. Visualmente podemos perceber que parece que existe uma relação linear.

# Normalidade dos Resíduos

A normalidade dos resíduos é uma das suposições da regressão linear. Os resíduos são as diferenças entre os valores observados e os valores previstos pelo modelo de regressão.  A fórmula para calcular o resíduo é apresentada a seguir.

$Resíduo = y_i - \hat{y}_i$

onde:

* $y_i$ é o valor observado da variável dependente

* $\hat{y}_i$ é o valor previsto pelo modelo de regressão

Para verificar os resíduos podemos utilizar dois tipos de gráficos: 

* Um histograma dos resíduos 
* Um QQ plot dos resíduos

Vamos primeiro criar um modelo de regressão e exibir os erros. Utilizaremos todos os dados para o treino, não separaremos estes dados entre treino e teste pois nosso objetivo não é verificar o quão bom esse modelo se ajusta aos dados e sim simplesmente verificar a normalidade dos resíduos. 

```Python
import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(42)
x = np.random.rand(100)
y = 2 * x + 1 + np.random.normal(0, 0.2, 100)
modelo = LinearRegression()
modelo.fit(x.reshape(-1, 1), y)
erros = y - modelo.predict(x.reshape(-1, 1))
erros
```

A saída é uma lista com o erro de cada uma das observações.

```
array([ 0.00883089, -0.0153981 ,  ....,  0.13233854])
```
## Histograma dos resíduos

Um histograma ser utilizado para analisar a normalidade dos resíduos.

```Python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.linear_model import LinearRegression

np.random.seed(42)
x = np.random.rand(100)
y = 2 * x + 1 + np.random.normal(0, 0.2, 100)
modelo = LinearRegression()
modelo.fit(x.reshape(-1, 1), y)
erros = y - modelo.predict(x.reshape(-1, 1))
sn.histplot(erros, bins=20, kde=True)
plt.xlabel('Erros')
plt.ylabel('Frequência')
plt.title('Histograma dos Erros')
plt.show()
```
![alt text](image-2.png)

É possível perceber visualmente que o histograma dos resíduos segue uma distribuição parecida com a distribuição normal.

## QQ Plot

Um QQ Plot (Quantile-Quantile Plot) é uma ferramenta gráfica usada para comparar a distribuição de um conjunto de dados com uma distribuição teórica específica, geralmente a distribuição normal. No contexto de regressão linear, um QQ Plot dos resíduos é útil para verificar a suposição de normalidade dos erros.

```Python
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.random.rand(100)
y = 2 * x + 1 + np.random.normal(0, 0.2, 100)
modelo = LinearRegression()
modelo.fit(x.reshape(-1, 1), y)
erros = y - modelo.predict(x.reshape(-1, 1))
sm.qqplot(erros, line='45', fit=True)
plt.show()
```

![alt text](image-3.png)