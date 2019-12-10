import csv
import sys

FILENAME = "betting/nba_money_lines2017.csv"


def build_line_dict(filename=FILENAME):
    line_dict = {}
    with open(filename, "r") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        next(readCSV, None)
        seq = 21700105
        for row in readCSV:
            # print (f'key value in dictionary {int(row[0])}')
            # print("key value", int(row[0]))
            if int(row[0]) != seq:
                print(f"Game id: {int(row[0]) - 1} is missing")
                seq += 1
            line_dict[int(row[0])] = int(row[5])
            seq += 1
    return line_dict


def get_lines(line_dict, games):
    return [line_dict[game] for game in games]


if __name__ == "__main__":
    d = build_line_dict()
    print(get_lines(d, [21700294]))
