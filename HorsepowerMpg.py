# --------------------------------------NAME: Fuseini Salifu---------------------------------------------

""" This program to compare Horsepower versus Miles per Gallon for a variety of automotive vehicles by the big three 
     automakers Chevrolet, Ford, and Toyota.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def piecewise_kde(X, Y, npiece):
    X = np.array(X)
    Y = np.array(Y)
    x0, x1 = np.min(X), np.max(X)
    sigma = 0.25 * (x1 - x0) / npiece
    root2pi = 2.50662827463
    npoint = npiece + 1
    xout = np.linspace(x0, x1, npoint)
    yout = np.zeros(npoint)

    for i in range(npoint):
        mu = xout[i]
        weight = 1.0 / (sigma * root2pi) * np.exp(-0.5 * (X - mu) ** 2 / sigma ** 2)
        yout[i] = np.sum(Y * weight) / np.sum(weight)

    return xout, yout


def polynomial_fit(x, y, degree):
    coeffs = np.polyfit(x, y, degree)
    return np.poly1d(coeffs)


def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


file_path = 'C:/Users/fsali/PycharmProjects/newBeginning/CSC315Class/cars.csv'
cars_df = pd.read_csv(file_path)
makers = ['Chevrolet', 'Ford', 'Toyota']
filtered_df = cars_df[cars_df['Make'].isin(makers)]
other_df = cars_df[~cars_df['Make'].isin(makers)]
colors = {'Chevrolet': 'orange', 'Ford': 'blue', 'Toyota': 'red'}


def plot_histograms():
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    make_counts = cars_df['Make'].value_counts().sort_index()
    make_counts.plot(kind='barh', color=[colors.get(make, 'black') for make in make_counts.index])
    plt.xlabel('Number of Vehicles')
    plt.ylabel('Car Makes')
    plt.title('Number of Vehicles by Make (Alphabetical)')

    plt.subplot(1, 2, 2)
    make_counts = cars_df['Make'].value_counts().sort_values()
    make_counts.plot(kind='barh', color=[colors.get(make, 'black') for make in make_counts.index])
    plt.xlabel('Number of Vehicles')
    plt.ylabel('Car Makes')
    plt.title('Number of Vehicles by Make (Ascending Count)')

    plt.tight_layout()
    plt.show()


def plot_regression():
    degrees = [1, 2, 3, 4, 5]

    for make in makers:
        subset = filtered_df[filtered_df['Make'] == make]
        x, y = subset['Horsepower'], subset['Mpg']

        for degree in degrees:
            plt.figure(figsize=(10, 6))
            plt.scatter(x, y, color=colors[make], label=f'{make} Data')

            poly_eq = polynomial_fit(x, y, degree)
            x_range = np.linspace(min(x), max(x), 100)
            y_pred = poly_eq(x)
            mae_poly = calculate_mae(y, poly_eq(x))
            plt.plot(x_range, poly_eq(x_range), linestyle='dashed', color='gray',
                     label=f'Poly {degree} (MAE: {mae_poly:.2f})')

            x_segments, y_segments = piecewise_kde(x, y, degree)
            mae_piecewise = calculate_mae(y, np.interp(x, x_segments, y_segments))
            plt.plot(x_segments, y_segments, color='black', marker='^',
                     label=f'Piecewise Fit {degree} Segments (MAE: {mae_piecewise:.2f})')

            plt.xlabel('Horsepower')
            plt.ylabel('Mpg')
            plt.title(f'{make} Horsepower vs Mpg - Polynomial Degree {degree}')
            plt.legend()
            plt.show()


plot_histograms()
plot_regression()
