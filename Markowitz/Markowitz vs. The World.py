import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
start = datetime(2010,1,4)
import pandas_datareader.data as web

#Parametros
dias=600
training_start = datetime(2010,1,4)
training_finish = datetime(2018,6,15)
testing_start = datetime(2019,1,1)
testing_finish = datetime(2018,6,15)

#Market Data
market_data = web.get_data_yahoo(["^GSPC", "^GDAXI", "^FCHI",  "^N225", "^HSI", "^BVSP", "^MXX", "XWD.TO"], start)
market_data = market_data["Close"].fillna(method="ffill")
activos = ["^GSPC", "^GDAXI", "^FCHI",  "^N225", "^HSI", "^BVSP", "^MXX"] 

#Risk Free
US_rates = web.get_data_fred(["DTB3", "DGS3MO", "DTB6", "DGS6MO", "DTB1YR", "DGS2", "DGS10"], start).fillna(method="ffill")
US_rates = (US_rates["DTB1YR"].loc[training_start:training_finish].mean()/100)

training = market_data.loc[training_start:training_finish,activos].pct_change(dias).dropna().mean() - US_rates

#FUNCIÃ“N DE PORTAFOLIO Ã“PTIMO
def portafolio_optimo(retornos):
    retornos_portafolio = []                                                 
    volatilidad_portafolio = [] 
    sharpe_ratio = []
    pesos_activos = []
    numero_activos = len(activos) 
    simulaciones = 50000 
    cov = np.cov(retornos) 
    for portfolio_n in range(simulaciones):
        pesos = np.random.random(numero_activos) 
        pesos /= np.sum(pesos) 
        retorno = np.dot(pesos, retornos) 
        volatilidad = np.sqrt(np.dot(pesos.T, np.dot(cov, pesos))) 
        sharpe = retorno/volatilidad 
        retornos_portafolio.append(retorno) 
        volatilidad_portafolio.append(volatilidad) 
        sharpe_ratio.append(sharpe) 
        pesos_activos.append(pesos)
    portafolio = {'Retornos': retornos_portafolio,
             'Volatilidad': volatilidad_portafolio,
             'Ratio de Sharpe': sharpe_ratio}
    for contador, activo in enumerate(activos):
        portafolio[activo+' Peso'] = [Peso[contador] for Peso in pesos_activos]
    global combinaciones, max_sharpe, info_sharpe, pesos_optimos 
    
    #Portafolio de Sharpe
    combinaciones = pd.DataFrame(portafolio) 
    max_sharpe = combinaciones["Ratio de Sharpe"].max() 
    info_sharpe = combinaciones[combinaciones["Ratio de Sharpe"] == max_sharpe] 
    pesos_optimos = info_sharpe.iloc[:,3:] 

#OBTENGO PORTAFOLIO OPTIMO
portafolio_optimo(training) 

#Universo Ploteado
plt.scatter(x=combinaciones["Volatilidad"], y=combinaciones["Retornos"], c=combinaciones["Ratio de Sharpe"], cmap='RdYlBu',edgecolors="b")
plt.xlabel("Volatilidad")
plt.ylabel("Retornos")
plt.title("Rendimientos de posibles portafolios: Universo de Combinaciones")
plt.show()

#Testeo Portafolio Optimo
retornos_listado = market_data.loc[testing_start:,activos].pct_change().dropna().T 
retorno_acumulado = (np.matmul(pesos_optimos,retornos_listado)+1).cumprod()

mercado = (market_data.loc[testing_start:, "XWD.TO"].pct_change().dropna()+1).cumprod()

#Grafico Comparacion
comparacion = np.column_stack([mercado, retorno_acumulado])  
comparacion = pd.DataFrame(comparacion, columns=["MSCI World", "Portafolio Óptimo"], index=mercado.index).dropna()-1
grafico_backtest = comparacion.plot()    
plt.axhline(0, color="black", linewidth=1)
grafico_backtest.set_ylim(-0.1, 0.2)
plt.title("Backtest combinaciones óptimas")
plt.show()