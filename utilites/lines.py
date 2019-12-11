import csv
import sys
from . import file_dumps
from nba_api.stats.static import teams


FILENAME_2017 = "betting/nba_money_lines2017.csv"
FILENAME_2018 = "betting/nba_money_lines2018.csv"

# Read the csv in the 2017 format
def build_2017_line_dict():
    line_dict = {}
    with open(FILENAME_2017, "r") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        next(readCSV, None)
        seq = 21700105
        for row in readCSV:
            if int(row[0]) != seq:
                print(f"Game id: {int(row[0]) - 1} is missing")
                seq += 1
            line_dict[int(row[0])] = (int(row[4]), int(row[5]))
            seq += 1
    return line_dict


# Read the csv in the 2018 format
def build_2018_line_dict():
    new_lines = []
    with open(FILENAME_2018, "r") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        next(readCSV, None)
        readCSV = list(readCSV)
        for i in range(0, len(readCSV) - 1, 2):
            new_lines.append(
                (readCSV[i][3], readCSV[i + 1][3], readCSV[i][11], readCSV[i + 1][11])
            )

    new_and_improved_lines = []
    for game in new_lines:
        full_away_team_info = teams.find_teams_by_city(game[0])
        if len(full_away_team_info) == 0:
            new_name = resolve_ambiguous_name(game[0])
            full_away_team_info = teams.find_team_by_abbreviation(new_name)

        full_home_team_info = teams.find_teams_by_city(game[1])
        if len(full_home_team_info) == 0:
            new_name = resolve_ambiguous_name(game[1])
            full_home_team_info = teams.find_team_by_abbreviation(new_name)

        new_games = []

        if type(full_away_team_info) is list:
            full_away_team_info = full_away_team_info[0]

        if type(full_home_team_info) is list:
            full_home_team_info = full_home_team_info[0]

        new_games.append(full_away_team_info["id"])
        new_games.append(full_home_team_info["id"])
        new_games.append(game[2])
        new_games.append(game[3])

        new_and_improved_lines.append(new_games)

    game_dict = {}
    game_counter = 21800001
    for i in range(1230):
        game_dict[game_counter + i] = 0

    final_game_list = []
    for game in new_and_improved_lines:
        for key in game_dict:
            if game_dict[key] == 0:
                boxscore = file_dumps.read_json(f"games2018/00{key}")
                away_team_id = boxscore[0][1]
                home_team_id = boxscore[-1][1]

                if away_team_id == game[0] and home_team_id == game[1]:
                    game_dict[key] = 1
                    final_game_list.append(
                        (boxscore[0][0], game[0], game[1], game[2], game[3])
                    )
                    break
    line_dict = {}
    for game in final_game_list:
        line_dict[int(game[0])] = (int(game[3]), int(game[4]))
    return line_dict


# Helper method to account for data from two sources
def resolve_ambiguous_name(name):
    if name == "LAClippers":
        return "LAC"
    if name == "NewYork":
        return "NYK"
    if name == "SanAntonio":
        return "SAS"
    if name == "OklahomaCity":
        return "OKC"
    if name == "GoldenState":
        return "GSW"
    if name == "LALakers":
        return "LAL"
    if name == "NewOrleans":
        return "NOP"


# simple dictionary selector
def get_lines(line_dict, games):
    return [line_dict[game] for game in games]


# This is the function you should call from outside lines.py
def get_line_dict(season):
    if season == "2017":
        return build_2017_line_dict()
    if season == "2018":
        return build_2018_line_dict()


# if __name__ == '__main__':
#     print(resolve_ambiguous_name("LAClippers"))
