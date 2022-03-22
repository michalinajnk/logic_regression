# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 3: Regresja logistyczna
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba, P. Dąbrowski
#  2019
# --------------------------------------------------------------------------
import functools

import numpy as np
import math

def sigmoid(x):
    """
    Wylicz wartość funkcji sigmoidalnej dla punktów *x*.
    :param x: wektor wartości *x* do zaaplikowania funkcji sigmoidalnej Nx1
    :return: wektor wartości funkcji sigmoidalnej dla wartości *x* Nx1
    """
    return np.divide(1, np.add(1, np.exp(-x)))


def logistic_cost_function(w, x_train, y_train):
    """
    Wylicz wartość funkcji logistycznej oraz jej gradient po parametrach.

    :param w: wektor parametrów modelu Mx1
    :param x_train: zbiór danych treningowych NxM
    :param y_train: etykiety klas dla danych treningowych Nx1
    :return: krotka (log, grad), gdzie *log* to wartość funkcji logistycznej,
        a *grad* jej gradient po parametrach *w* Mx1
    """
    sigma_array = sigmoid(x_train @ w)
    cost = -np.log(np.prod(np.abs(sigma_array + y_train - 1)))

    gradient = x_train.transpose() @ (sigma_array - y_train) / sigma_array.shape[0]
    return cost/y_train.shape[0], gradient


def gradient_descent(obj_fun, w0, epochs, eta):
    """
    Dokonaj *epochs* aktualizacji parametrów modelu metodą algorytmu gradientu
    prostego, korzystając z kroku uczenia *eta* i zaczynając od parametrów *w0*.
    Wylicz wartość funkcji celu *obj_fun* w każdej iteracji. Wyznacz wartość
    parametrów modelu w ostatniej epoce.

    :param obj_fun: optymalizowana funkcja celu, przyjmująca jako argument
        wektor parametrów *w* [wywołanie *val, grad = obj_fun(w)*]
    :param w0: początkowy wektor parametrów *w* Mx1
    :param epochs: liczba epok algorytmu gradientu prostego
    :param eta: krok uczenia
    :return: krotka (w, log_values), gdzie *w* to znaleziony optymalny
        punkt *w*, a *log_values* to lista wartości funkcji celu w każdej
        epoce (lista o długości *epochs*)
    """
    w = w0
    log_vals = np.empty((epochs,))
    _, grad = obj_fun(w0)
    for i in range(epochs):
        w = w - eta * grad
        val, grad = obj_fun(w)
        log_vals[i] = val
    return w, log_vals


def stochastic_gradient_descent(obj_fun, x_train, y_train, w0, epochs, eta, mini_batch):
    """
    Dokonaj *epochs* aktualizacji parametrów modelu metodą stochastycznego
    algorytmu gradientu prostego, korzystając z kroku uczenia *eta*, paczek
    danych o rozmiarze *mini_batch* i zaczynając od parametrów *w0*. Wylicz
    wartość funkcji celu *obj_fun* w każdej iteracji. Wyznacz wartość parametrów
    modelu w ostatniej epoce.

    :param obj_fun: optymalizowana funkcja celu, przyjmująca jako argumenty
        wektor parametrów *w*, paczkę danych składających się z danych
        treningowych *x* i odpowiadających im etykiet *y*
        [wywołanie *val, grad = obj_fun(w, x, y)*]
    :param w0: początkowy wektor parametrów *w* Mx1
    :param epochs: liczba epok stochastycznego algorytmu gradientu prostego
    :param eta: krok uczenia
    :param mini_batch: rozmiar paczki danych / mini-batcha
    :return: krotka (w, log_values), gdzie *w* to znaleziony optymalny
        punkt *w*, a *log_values* to lista wartości funkcji celu dla całego
        zbioru treningowego w każdej epoce (lista o długości *epochs*)
    """
    M = int(y_train.shape[0] / mini_batch)
    x_mini_batch = np.vsplit(x_train, M)
    y_mini_batch = np.vsplit(y_train, M)

    w = w0
    log_vals = np.empty((epochs,))
    for i in range(epochs):
        for x, y in zip(x_mini_batch, y_mini_batch):
            _, grad = obj_fun(w, x, y)
            w = w - eta * grad
        val, delta_w = obj_fun(w, x_train, y_train)
        log_vals[i] = val
    return w, log_vals

def regularized_logistic_cost_function(w, x_train, y_train, regularization_lambda):
    """
    Wylicz wartość funkcji logistycznej z regularyzacją l2 oraz jej gradient
    po parametrach.

    :param w: wektor parametrów modelu Mx1
    :param x_train: zbiór danych treningowych NxM
    :param y_train: etykiety klas dla danych treningowych Nx1
    :param regularization_lambda: parametr regularyzacji l2
    :return: krotka (log, grad), gdzie *log* to wartość funkcji logistycznej
        z regularyzacją l2, a *grad* jej gradient po parametrach *w* Mx1
    """

    ws = np.delete(w, 0)
    sigArr = sigmoid(x_train @ w)
    w = w.transpose()
    norm = regularization_lambda / 2 * (np.linalg.norm(ws) ** 2)
    outArr = np.divide(y_train * np.log(sigArr) + (1 - y_train) * np.log(1 - sigArr), -1 * sigArr.shape[0])
    wz = w.copy().transpose()
    wz[0] = 0
    grad = (x_train.transpose() @ (sigArr - y_train)) / sigArr.shape[0] + regularization_lambda * wz
    return (np.sum(outArr) + norm, grad)

def prediction(x, w, theta):
    """
    Wylicz wartości predykowanych etykiet dla obserwacji *x*, korzystając
    z modelu o parametrach *w* i progu klasyfikacji *theta*.

    :param x: macierz obserwacji NxM
    :param w: wektor parametrów modelu Mx1
    :param theta: próg klasyfikacji z przedziału [0,1]
    :return: wektor predykowanych etykiet ze zbioru {0, 1} Nx1
    """
    sigArr = sigmoid(x @ w)
    return (sigArr > theta).astype(int).reshape(x.shape[0], 1)


def f_measure(y_true, y_pred):
    """
    Wylicz wartość miary F (F-measure) dla zadanych rzeczywistych etykiet
    *y_true* i odpowiadających im predykowanych etykiet *y_pred*.

    :param y_true: wektor rzeczywistych etykiet Nx1
    :param y_pred: wektor etykiet predykowanych przed model Nx1
    :return: wartość miary F (F-measure)
    """
    TP = np.sum(np.bitwise_and(y_true, y_pred))
    FP = np.sum(np.bitwise_and(np.bitwise_not(y_true), y_pred))
    FN = np.sum(np.bitwise_and(y_true, np.bitwise_not(y_pred)))
    return (2 * TP) / (2 * TP + FP + FN)


def model_selection(x_train, y_train, x_val, y_val, w0, epochs, eta, mini_batch, lambdas, thetas):
    """
    Policz wartość miary F dla wszystkich kombinacji wartości regularyzacji
    *lambda* i progu klasyfikacji *theta. Wyznacz parametry *w* dla modelu
    z regularyzacją l2, który najlepiej generalizuje dane, tj. daje najmniejszy
    błąd na ciągu walidacyjnym.

    :param x_train: zbiór danych treningowych NxM
    :param y_train: etykiety klas dla danych treningowych Nx1
    :param x_val: zbiór danych walidacyjnych NxM
    :param y_val: etykiety klas dla danych walidacyjnych Nx1
    :param w0: początkowy wektor parametrów *w* Mx1
    :param epochs: liczba epok stochastycznego algorytmu gradientu prostego
    :param eta: krok uczenia
    :param mini_batch: rozmiar paczki danych / mini-batcha
    :param lambdas: lista wartości parametru regularyzacji l2 *lambda*,
        które mają być sprawdzone
    :param thetas: lista wartości progów klasyfikacji *theta*,
        które mają być sprawdzone
    :return: krotka (regularization_lambda, theta, w, F), gdzie
        *regularization_lambda* to wartość regularyzacji *lambda* dla
        najlepszego modelu, *theta* to najlepszy próg klasyfikacji,
        *w* to parametry najlepszego modelu, a *F* to macierz wartości miary F
        dla wszystkich par *(lambda, theta)* #lambda x #theta
    """

    f_mes_list = []
    par_w_list = []
    tuples = []
    min_idx = 0
    idx = 0

    for i in range(len(lambdas)):
        w, _ = stochastic_gradient_descent(
            lambda x,y,par_w: regularized_logistic_cost_function(x,y,par_w,lambdas[i]),
            x_train, y_train, w0, epochs, eta, mini_batch)
        par_w_list.append(w)
        for j in range(len(thetas)):
            measure = f_measure(y_val, prediction(x_val, w, thetas[j]))
            tuples.append((i, j, w))
            f_mes_list.append(measure)
            if (f_mes_list[min_idx] < measure):
                min_idx = idx
            idx += 1


    return (lambdas[tuples[min_idx][0]], thetas[tuples[min_idx][1]], tuples[min_idx][2],
            np.array(f_mes_list).reshape(len(lambdas), len(thetas)))







