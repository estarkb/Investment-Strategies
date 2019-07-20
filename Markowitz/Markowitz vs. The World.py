import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set()
import numpy as np
from datetime import datetime
start = datetime(2010,1,4)
import pandas_datareader.data as web

#Market Data
market_data = web.get_data_yahoo(["^GSPC", "^GDAXI", "^FCHI",  "^N225", "^HSI", "^BVSP", "^MXX", "XWD.TO"], start)
market_data = market_data["Close"].fillna(method="ffill")
activos = ["^GSPC", "^GDAXI", "^FCHI",  "^N225", "^HSI", "^BVSP", "^MXX"] 
market_data.tail(3)

#Benchmark
msci_world = web.get_data_yahoo(["XWD.TO"], start)
msci_world = msci_world["Close"].fillna(method="ffill")

#Risk Free
US_rates = web.get_data_fred(["DTB3", "DGS3MO", "DTB6", "DGS6MO", "DTB1YR", "DGS2", "DGS10"], start)
US_rates = US_rates.fillna(method="ffill")
US_rates.tail(3)

#Observación: Esta tabla nos muestra la tasa (el anterior nos muestra el valor del índice) por tanto los tratamientos varían
retornos = market_data.iloc[0:2200,:7].pct_change(20).dropna() #obetemos los retornos de los primeros 2000 días
retornos_mercado = market_data.iloc[0:2200,:7].pct_change(240).dropna().mean() #obtenemos los retornos promedio anuales [trading days] 
retornos_rf = US_rates.iloc[0:2200,4].mean()/100 #obtenemos retornos de tasa a un año (recordar, estaba expresado en tasa)
retornos_exceso = retornos_mercado-retornos_rf #finalmente sacamos los Premios por Riesgo

retornos1 = market_data.iloc[2300:,:7].pct_change(20).dropna() #obetemos los retornos en periodo de testeo
retornos_mercado1 = market_data.iloc[2300:,:7].pct_change(240).dropna().mean() #obtenemos los retornos promedio anuales [trading days] 
retornos_rf1 = US_rates.iloc[2300:,4].mean()/100 #obtenemos retornos de tasa a un año (recordar, estaba expresado en tasa)
retornos_exceso1 = retornos_mercado1-retornos_rf1 #finalmente sacamos los Premios por Riesgo

#FUNCIÃ“N DE PORTAFOLIO Ã“PTIMO
def portafolio_optimo(retornos):
    retornos_portafolio = [] #Listas vacÃ­as
    volatilidad_portafolio = [] #Listas vacÃ­as
    sharpe_ratio = [] #Listas vacÃ­as
    pesos_activos = [] #Listas vacÃ­as
    numero_activos = len(activos) #AsÃ­ hago una matriz de pesos
    numero_portafolios = 50000 #NÃºmero de portafolios a simular
    cov = np.cov(retornos) #Covarianza de los retornos
    for portfolio_n in range(numero_portafolios):
        pesos = np.random.random(numero_activos) #Obtengo pesos aleatorios
        pesos /= np.sum(pesos) #Hago que dichos pesos sumen 1
        retorno = np.dot(pesos, retornos) #Retornos de los portafolios simulados
        volatilidad = np.sqrt(np.dot(pesos.T, np.dot(cov, pesos))) #Volatilidad de Portafolios simulados
        #np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
        sharpe = retorno/volatilidad #Sharpe de cada uno
        retornos_portafolio.append(retorno) #Relleno lista de retornos
        volatilidad_portafolio.append(volatilidad) #Relleno lista de volatilidades
        sharpe_ratio.append(sharpe) #Relleno lista de ratios de Sharpe
        pesos_activos.append(pesos) #Relleno lista de pesos aleatorios
    portafolio = {'Retornos': retornos_portafolio, #Diccionario que reÃºne las listas
             'Volatilidad': volatilidad_portafolio,
             'Ratio de Sharpe': sharpe_ratio}
    for contador, activo in enumerate(activos):
        portafolio[activo+' Peso'] = [Peso[contador] for Peso in pesos_activos]
    global resultados_combinaciones, resultados_max_sharpe, resultados_info_sharpe, resultados_pesos_optimos #Para que sea visbles en exploradpor de variables, las convierto de locales a globales
    
    #Portafolio de Sharpe
    resultados_combinaciones = pd.DataFrame(portafolio) #Universo de Combinaciones, utilizable luego para promediar primeros 100 o 200 //suavizar y no hacer overfit del pasado
    resultados_max_sharpe = resultados_combinaciones["Ratio de Sharpe"].max() #MÃ¡ximo Sharpe
    resultados_info_sharpe = resultados_combinaciones[resultados_combinaciones["Ratio de Sharpe"] == resultados_max_sharpe] #InformaciÃ³n asociada a mÃ¡ximo Sharpe
    resultados_pesos_optimos = resultados_info_sharpe.iloc[:,3:] #Lista de pesos Ã³ptimos
    
portafolio_optimo(retornos_exceso) #OBTENGO PORTAFOLIO OPTIMO
#Universo Ploteado
plt.scatter(x=resultados_combinaciones["Volatilidad"], y=resultados_combinaciones["Retornos"], c=resultados_combinaciones["Ratio de Sharpe"], cmap='RdYlBu',edgecolors="b")
plt.xlabel("Volatilidad")
plt.ylabel("Retornos")
plt.title("Rendimientos de posibles portafolios: Universo de Combinaciones")
plt.show()

#Testeo Portafolio Optimo
global retornos_listado, retornos_diarios_port_sharp, retornos_diarios_port_sharp1, retorno_acumulado_portafolio, benchmark, comparacion

retornos_listado = market_data.loc["2019-01-04":,"^BVSP":].pct_change().dropna() #Le muestro info para poner a prueba combinaciÃ³n selecionada
retornos_listado = retornos_listado.T #reemplazo ceros para no multiplicar por cero mÃ¡s abajo, trasnpongo para hacer multiplicable y obtener resultado diario
retornos_diarios_port_sharp = np.matmul(resultados_pesos_optimos,retornos_listado) #obtengo retornos diarios
retornos_diarios_port_sharp1 = retornos_diarios_port_sharp +1 #a los retornos les sumo 1 (y asÃ­ al multiplicarlos los hago acumulables)
retorno_acumulado_portafolio = (retornos_diarios_port_sharp1).cumprod() #multiplicaciÃ³n acumulada portafolio Ã³ptimo


#benchmark = msci_world.loc["2018-12-31":,:]
#benchmark = (benchmark.pct_change()+1).cumprod() #multiplicaciÃ³n acumulada benchmark elegido
comparacion = np.column_stack([(market_data.iloc[2344:,7].pct_change()+1).cumprod(), retorno_acumulado_portafolio]) #Hago df con listado de retornos de portafolio y benchmark 
comparacion = pd.DataFrame(comparacion, columns=["MSCI World", "Portafolio Óptimo"], index=market_data.iloc[2344:,:].index)-1
grafico_backtest = comparacion.plot() #Visualizo la comparaciÃ³n    
plt.axhline(0, color="black", linewidth=1)
grafico_backtest.set_ylim(-0.1, 0.2)
plt.title("Backtest combinaciones óptimas")
plt.show()