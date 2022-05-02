import matplotlib.pyplot as plt


def plot_train_test(datos_train, datos_test):
    fig, ax = plt.subplots(figsize=(9, 4))
    datos_train.plot(ax=ax, label='train')
    datos_test.plot(ax=ax, label='test')
    ax.legend()
    plt.show()


def plot_predicciones(datos_train, datos_test, predic):
    fig, ax = plt.subplots(figsize=(9, 4))
    datos_train.plot(ax=ax, label='train')
    datos_test.plot(ax=ax, label='test')
    predic.plot(ax=ax, label='predicciones')
    ax.legend()
    plt.show()
