"""Functions that convert metric and imperial units"""


def cel2fah(c):
    """Converts a temperature in Celsius to Fahrenheit"""

    temp_in_fah = 9 / 5 * c + 32
    return temp_in_fah


def fah2cel(f):
    """Converts a temperature in Fahrenheit to Celsius"""
    temp_in_cel = 5 / 9 * (f - 32)
    return temp_in_cel


def km2mi(km):
    """Converts a distance in kilometers to miles"""
    dist_in_mi = km / 1.60934
    return dist_in_mi


def mi2km(mi):
    """Converts a distance in miles to kilometers"""

    dist_in_km = mi * 1.60934
    return dist_in_km


if __name__ == "__main__":

    for i in range(0, 41):

        if i % 5 == 0:
            if i <= 20:
                converted_temp = round(cel2fah(i))
                print(i, 'C is', converted_temp, "F")
            if 20 <= i <= 40:
                temp = round(fah2cel(i))
                print(i, 'F is', temp, 'C')
    print()

    for k in range(1, 10):
        if k < 6:
            dist1 = round(km2mi(k), 1)
            print(k, 'km is', dist1, 'mi')
        if k > 4:
            dist2 = round(mi2km(k), 1)
            print(k, 'mi is', dist2, 'km')

