import csv


FILENAME = "betting/nba_money_lines.csv"


def build_line_dict():
    line_dict = {}
    with open(FILENAME, "r") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        next(readCSV, None)
        for row in readCSV:
            line_dict[int(row[0])] = int(row[4])
    return line_dict


def get_lines(line_dict, game_ids):
    return [line_dict[game] for game in game_ids]


if __name__ == "__main__":
    d = build_line_dict()
    print(get_lines(d, [21700105]))
