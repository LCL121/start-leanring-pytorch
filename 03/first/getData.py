import random
import csv


def get_x():
    return random.random() * 100


def get_y(x):
    return 10.4563 * x + 0.2873 + random.random() * 0.001


def run(num):
    all_data = []
    for i in range(num):
        x = get_x()
        y = get_y(x)
        # print("{0},{1}".format(x, y))
        all_data.append((x, y))
    return all_data


if __name__ == "__main__":
    all_data = run(10000)
    with open("data.csv", "w", newline="") as infile:
        csv_writer = csv.writer(infile)
        for data in all_data:
            csv_writer.writerow(data)

