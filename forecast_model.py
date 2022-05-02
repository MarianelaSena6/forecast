from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from skforecast.model_selection import grid_search_forecaster
from clean import *
from graficos import *
import joblib


# Separación datos train-test
def separacion_train_test(df, steps):
    datos_train = df[:-steps]
    datos_test = df[-steps:]

    print(f"Fechas train : {datos_train.index.min()} --- {datos_train.index.max()}  (n={len(datos_train)})")
    print(f"Fechas test  : {datos_test.index.min()} --- {datos_test.index.max()}  (n={len(datos_test)})")
    return datos_train, datos_test


# Crear y entrenar forecaster
def create_forecaster(datos_train, max_depth_num, n_estimators_num, random_state, lags_num):
    regressor = RandomForestRegressor(max_depth=max_depth_num, n_estimators=n_estimators_num, random_state=random_state)
    forecaster_obj = ForecasterAutoreg(
        regressor=regressor,
        lags=lags_num
    )

    forecaster_obj.fit(y=datos_train)
    return forecaster_obj


# Error test con error cuadrático medio.
def error_test(datos_test, predic):
    error_mse = mean_squared_error(
        y_true=datos_test,
        y_pred=predic)
    return error_mse


# Grid search de hiperparámetros
def ajuste_parametros(datos_train, steps, lags_num, random_state):
    forecaster_obj = ForecasterAutoreg(
        regressor=RandomForestRegressor(random_state=random_state),
        lags=lags_num  # Este valor será remplazado en el grid search
    )

    # Lags utilizados como predictores
    lags_grid = [10, 20]

    # Hiperparámetros del regresor
    param_grid = {'n_estimators': [200, 300, 500],
                  'max_depth': [3, 5, 10]}

    resultados_grid = grid_search_forecaster(
        forecaster=forecaster_obj,
        y=datos_train,
        param_grid=param_grid,
        lags_grid=lags_grid,
        steps=steps,
        refit=True,
        metric='mean_squared_error',
        initial_train_size=int(len(datos_train) * 0.5),
        fixed_train_size=False,
        return_best=True,
        verbose=False
    )
    return resultados_grid


if __name__ == "__main__":
    df_series = pd.read_csv('series.csv', sep=',')

    # SEPARAR DATAFRAME POR CATEGORIAS
    CAT1_df = df_series.groupby('CATEGORY').get_group('CATEG-1').sort_values('DATE')
    CAT2_df = df_series.groupby('CATEGORY').get_group('CATEG-2').sort_values('DATE')
    CAT3_df = df_series.groupby('CATEGORY').get_group('CATEG-3').sort_values('DATE')

    # LIMPIAR DATAFRAME
    CAT1_df = clean_df(CAT1_df)
    CAT1_df = delete_nulls(CAT1_df)

    # Calcular steps a predecir, o el conjunto de test
    steps_predict = round((CAT1_df.shape[0] * 10) / 100)

    # Se separa el dataset para entrenar de la parte de test
    train_test = separacion_train_test(CAT1_df, steps_predict)
    unidades_vendidas_train = train_test[0]['UNITS_SOLD']
    unidades_vendidas_test = train_test[1]['UNITS_SOLD']
    plot_train_test(unidades_vendidas_train, unidades_vendidas_test)

    # Entrenar modelo con parámetros de prueba
    forecaster = create_forecaster(unidades_vendidas_train, None, 100, 123, 15)

    # Se obtiene las predicciones
    predicciones = forecaster.predict(steps=steps_predict)
    print(predicciones.head(5))
    plot_predicciones(unidades_vendidas_train, unidades_vendidas_test, predicciones)
    print(f"Error de test (mse): {error_test(unidades_vendidas_test, predicciones)}")

    # Se realiza un ajuste de parámetros
    ajuste = ajuste_parametros(unidades_vendidas_train, steps_predict, 15, 123)

    # Parámetro obtenidos
    max_depth = ajuste.loc[ajuste.metric == ajuste.metric.min()].max_depth.tolist()[0]
    n_estimators = ajuste.loc[ajuste.metric == ajuste.metric.min()].n_estimators.tolist()[0]
    lags = len(ajuste.loc[ajuste.metric == ajuste.metric.min()].lags.tolist()[0])

    # Se vuelve a entrenar el modelo con los nuevos parámetros
    forecaster = create_forecaster(unidades_vendidas_train, max_depth, n_estimators, 123, lags)

    # Se obtiene las predicciones
    predicciones = forecaster.predict(steps=steps_predict)
    plot_predicciones(unidades_vendidas_train, unidades_vendidas_test, predicciones)
    print(f"Error de test (mse): {error_test(unidades_vendidas_test, predicciones)}")

    # Guardo el modelo.
    joblib.dump(forecaster, 'modelo_entrenado.pkl')
    # Carga del modelo.
    forecaster = joblib.load('modelo_entrenado.pkl')
