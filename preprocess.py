#!~/.pyvenvs/nba_dl
# -*- coding: utf-8 -*-

"""Downloads data from stats.nba.com and builds data structures

Downloads are slow. You should use pre-downloaded data if possible

"""

import os
import sys
import bisect
import json
import numpy as np
import file_dumps
from datetime import datetime
from collections import OrderedDict

## NBA APIs
from nba_api.stats.endpoints import boxscoreadvancedv2
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.static import teams
from nba_api.stats.static import players


SEASON = "2018-19"
SEASON_TYPE = "Regular Season"

NUM_GAMES = 1230  # 82*30/2
NUM_STATS = 23
# Gamelog constants
GAMELOG_GAME_ID_INDEX = 1
GAMELOG_DATE_INDEX = 2

# Boxscore constants
BOXSCORE_TEAM_ID = 1
BOXSCORE_PLAYER_ID = 4
BOXSCORE_STAT_START = 8


def win_loss_per_roster(list_game_ids, season_games, season):
    data = []

    path = ""
    if season == "2017":
        path = "games2017"
    elif season == "2018":
        path = "games2018"
    else:
        print(season)
        print(season == "2017")
        print(season == "2018")
        print("SEASON NOT AVAILABLE")
        sys.exit(0)

    for game_id in list_game_ids:
<<<<<<< HEAD
        boxscore = file_dumps.read_json("games/" + str(game_id))
=======
        boxscore = file_dumps.read_json(f"{path}/{game_id}")
>>>>>>> 8a0abd5929a6aec9bdef7ba872b7e5c6035206b7
        game_info = season_games[game_id]
        home_team_players = []
        away_team_players = []

        for player_info in boxscore:
            if player_info[BOXSCORE_TEAM_ID] == game_info["home_team"]:
                home_team_players.append(player_info[BOXSCORE_PLAYER_ID])
            else:
                away_team_players.append(player_info[BOXSCORE_PLAYER_ID])

        data.append(
            [game_id, home_team_players, away_team_players, game_info["home_team_wins"]]
        )

    return data


def fill_player_dict(all_games):
    player_dict = {}
    for player in players.get_players():
        player_dict[player["id"]] = dict.fromkeys([game[1] for game in all_games])
        for game in all_games:
            player_dict[player["id"]][game[1]] = None

    return player_dict


def get_games_api(team_ids):
    print("Loading game lists...")
    team_games = []
    all_games = []
    for tid in team_ids:
        team_log = teamgamelog.TeamGameLog(
            season=SEASON, season_type_all_star=SEASON_TYPE, team_id=tid
        ).get_dict()
        results = team_log["resultSets"][0]["rowSet"]
        game_ids = [res[GAMELOG_GAME_ID_INDEX] for res in results]
        dates = [
            datetime.strptime(res[GAMELOG_DATE_INDEX].replace(",", ""), "%b %d %Y")
            for res in results
        ]

        for pair in zip(dates, game_ids):
            bisect.insort_left(all_games, pair)
        # Reverse the game id list!
        team_games.append((tid, game_ids[::-1]))
        name = teams.find_team_name_by_id(tid)["full_name"]
        print(name + " done...")

    # deduplicate games
    all_games = OrderedDict((x, True) for x in all_games).keys()
    return team_games, list(all_games)


def create_player_matrix_from_api(team_ids):
    # dictionary with every player id

    team_games, all_games = get_games_api(team_ids)

    players_dict = fill_player_dict(all_games)
    player_game_count_dict = dict.fromkeys(
        [player["id"] for player in players.get_players()]
    )
    for key in player_game_count_dict:
        player_game_count_dict[key] = 0

    print("-------------------------------------------------")
    print("All games loaded. Beginning game by game calculations...")
    print("-------------------------------------------------")

    for game_info in team_games:
        team_id = game_info[0]
        print(
            "Running calculations for " + str({teams.find_team_name_by_id(team_id)['full_name']}) + "..."
        )
        game_set = game_info[1]
        i = 0
        for game_id in game_set:
            print(str(i) + " games done")
            i += 1
            boxscore = boxscoreadvancedv2.BoxScoreAdvancedV2(
                game_id=game_id
            ).get_dict()["resultSets"][0]["rowSet"]

            print("writing temp file for game: " + str(game_id))
            with open("games/{" + str(game_id), "w") as file:
                file.write(json.dumps(boxscore))

            for player_line in boxscore:
                if player_line[BOXSCORE_TEAM_ID] == team_id:
                    # update player matrix
                    if player_line[BOXSCORE_STAT_START] is None:
                        # DNP (coach's decision) case
                        continue
                    player_line[BOXSCORE_STAT_START] = int(
                        player_line[BOXSCORE_STAT_START].split(":")[0]
                    ) * 60 + int(player_line[BOXSCORE_STAT_START].split(":")[1])
                    players_dict[player_line[BOXSCORE_PLAYER_ID]][game_id] = np.array(
                        player_line[BOXSCORE_STAT_START:]
                    )
        print("-------------------------------------------------")

    prev_game_id = all_games[0][1]
    for chron_game in all_games:
        chron_game_id = chron_game[1]
        for p_id in players_dict:
            if players_dict[p_id][chron_game_id] is not None:
                if chron_game_id != prev_game_id:
                    if players_dict[p_id][prev_game_id] is not None:
                        players_dict[p_id][chron_game_id] = (
                            (player_game_count_dict[p_id] - 1)
                            * players_dict[p_id][chron_game_id]
                        ) + players_dict[p_id][prev_game_id]
                        player_game_count_dict[p_id] += 1
                        players_dict[p_id][chron_game_id] = (
                            players_dict[p_id][chron_game_id]
                            / player_game_count_dict[p_id]
                        )
                else:
                    player_game_count_dict[p_id] = 1
            else:
                players_dict[p_id][chron_game_id] = players_dict[p_id][prev_game_id]
        prev_game_id = chron_game_id

    for player in players_dict:
        for game in players_dict[player]:
            if players_dict[player][game] is None:
                players_dict[player][game] = np.zeros(23)
    # The final dictionary is: (num_players, num_games, num_stats)
    # each player id and game id is a key. The stats are stored in a numpy array
    return players_dict, all_games


def remove_away_teams(games):
    res = []
    for game in games:
        if "@" not in game[6]:
            res.append(game)
    return res


def create_wl_per_roster_from_local(outfile, season_str):
    _, ag = get_games_local(season_str)
    print("loaded games...")

    schedule_path = ""
    if season_str == "2017":
        schedule_path = "schedule/nba2017_18"
    elif season_str == "2018":
        schedule_path = "schedule/nba2018_19"
    else:
        print(season_str)
        print(season_str == "2017")
        print(season_str == "2018")
        print("SEASON NOT AVAILABLE")
        sys.exit(0)

    season = file_dumps.read_json(schedule_path)
    rows = season["resultSets"][0]["rowSet"]
    rows.sort(key=lambda x: x[4])
    ag = list(ag)
    ag.sort(key=lambda x: x[1])
    ag = np.array(ag)
    rows = remove_away_teams(rows)

    season_games = {}
    for game_id, row in zip(ag, rows):
        season_games[game_id[1]] = {}
        season_games[game_id[1]]["home_team"] = row[1]
        season_games[game_id[1]]["home_team_wins"] = True if row[7] == "W" else False
    print("loaded season...")

    wlpr = win_loss_per_roster([x[1] for x in ag], season_games, season_str)
    wlpr = np.array(wlpr)
    file_dumps.write_np_arr(wlpr, outfile)


def get_games_local(season):
    if season == "2017":
        ag_path = "games2017/all_games.npy"
        tg_path = "games2017/team_games.npy"
    elif season == "2018":
        ag_path = "games2018/all_games.npy"
        tg_path = "games2018/team_games.npy"
    else:
        print(season)
        print(season == "2017")
        print(season == "2018")
        print("SEASON NOT AVAILABLE")
        sys.exit(0)
    ag = file_dumps.read_numpy_arr(ag_path)
    tg = file_dumps.read_numpy_arr(tg_path)

    return tg, ag


def create_player_matrix_from_local(filename, season):
    path = ""
    if season == "2017":
        path = "games2017"
    elif season == "2018":
        path = "games2018"
    else:
        print(season)
        print(season == "2017")
        print(season == "2018")
        print("SEASON NOT AVAILABLE")
        sys.exit(0)

    team_games, all_games = get_games_local(season)

    players_dict = fill_player_dict(all_games)

    player_game_count_dict = dict.fromkeys(
        [player["id"] for player in players.get_players()]
    )

    for key in player_game_count_dict:
        player_game_count_dict[key] = 0

    print("-------------------------------------------------")
    print("All games loaded. Beginning game by game calculations...")
    print("-------------------------------------------------")

    for game_info in team_games:
        team_id = game_info[0]
        print(
            "Running calculations for " + str({teams.find_team_name_by_id(team_id)['full_name']}) + "..."
        )
        game_set = game_info[1]
        i = 0
        for game_id in game_set:
            i += 1
<<<<<<< HEAD
            boxscore = file_dumps.read_json("games/" + str(game_id))

            # print(f'writing temp file for game: {game_id}')
            # with open(f'games/{game_id}', "w") as file:
            #     file.write(json.dumps(boxscore))
=======
            boxscore = file_dumps.read_json(f"{path}/{game_id}")
>>>>>>> 8a0abd5929a6aec9bdef7ba872b7e5c6035206b7

            for player_line in boxscore:
                if player_line[BOXSCORE_TEAM_ID] == team_id:
                    # update player matrix
                    if player_line[BOXSCORE_STAT_START] is None:
                        # DNP (coach's decision) case
                        continue
                    player_line[BOXSCORE_STAT_START] = int(
                        player_line[BOXSCORE_STAT_START].split(":")[0]
                    ) * 60 + int(player_line[BOXSCORE_STAT_START].split(":")[1])
                    players_dict[player_line[BOXSCORE_PLAYER_ID]][game_id] = np.array(
                        player_line[BOXSCORE_STAT_START:], dtype="float32"
                    )
    print("-------------------------------------------")
    print("Filling in time step blanks...")
    prev_game_id = all_games[0][1]
    for chron_game in all_games:
        chron_game_id = chron_game[1]
        for p_id in players_dict:
            if players_dict[p_id][chron_game_id] is not None:
                if chron_game_id != prev_game_id:
                    if players_dict[p_id][prev_game_id] is not None:
                        players_dict[p_id][chron_game_id] = (
                            (player_game_count_dict[p_id] - 1)
                            * players_dict[p_id][chron_game_id]
                        ) + players_dict[p_id][prev_game_id]
                        player_game_count_dict[p_id] += 1
                        players_dict[p_id][chron_game_id] = (
                            players_dict[p_id][chron_game_id]
                            / player_game_count_dict[p_id]
                        )
                else:
                    player_game_count_dict[p_id] = 1
            else:
                players_dict[p_id][chron_game_id] = players_dict[p_id][prev_game_id]
        prev_game_id = chron_game_id

    for player in players_dict:
        for game in players_dict[player]:
            if players_dict[player][game] is None:
                players_dict[player][game] = np.zeros(23, dtype="float32")
    # The final dictionary is: (num_players, num_games, num_stats)
    # each player id and game id is a key. The stats are stored in a numpy array
    print("-------------------------------------------")
    file_dumps.write_player_dict(players_dict, filename)


def get_data(roster_file, matrix_file):
    nparr = file_dumps.read_numpy_arr(roster_file)
    pd = file_dumps.read_json(matrix_file)

    return pd, nparr


def get_2d_data(wl_per_rosters, player_matrix):
    data = []
    games = []

    for game in wl_per_rosters:
        # home roster
        row = []
        limit = min(len(game[1]), 13)
        for player in game[1][:limit]:
            if player not in player_matrix:
                if players.find_player_by_id(player) is not None:
                    print("ERROR has occured!!")
                    sys.exit(1)
            else:
                stats = np.array(player_matrix[player][game[0]], dtype="float32")
                row.append(stats)

        while len(row) < 13:
            row.append(np.zeros(23, dtype="float32"))

        limit = min(len(game[1]), 13)
        for player in game[2][:limit]:
            if player not in player_matrix:
                if players.find_player_by_id(player) is not None:
                    print("ERROR has occured!!")
                    sys.exit(1)
                else:
                    print("This player is not in the NBA apparently!")
            else:
                stats = np.array(player_matrix[player][game[0]], dtype="float32")
                row.append(stats)

        while len(row) < 26:
            row.append(np.zeros(23, dtype="float32"))

        row = np.array(row, dtype="float32").flatten()

        if row.shape != (598,):
            print("RED ALERT!")
            print(game)
            print(stats)
            sys.exit(1)

        data.append(row)
        games.append(int(game[0]))

<<<<<<< HEAD
if __name__ == "__main__":
    create_wl_per_roster_from_local("final_data/wl_per_rosters_2.npy")
    create_player_matrix_from_local("final_data/player_dict_2.json")


=======
    data = np.array(data)
    games = np.array(games)
>>>>>>> 8a0abd5929a6aec9bdef7ba872b7e5c6035206b7

    return data, games


if __name__ == "__main__":

    create_player_matrix_from_local("final_data/player_dict_2017.json", "2017")
    print("1")
    create_wl_per_roster_from_local("final_data/wl_per_rosters_2017.npy", "2017")
    print("2")
    create_player_matrix_from_local("final_data/player_dict_2018.json", "2018")
    print("3")
    create_wl_per_roster_from_local("final_data/wl_per_rosters_2018.npy", "2018")
