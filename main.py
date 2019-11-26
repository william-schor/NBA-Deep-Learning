#!~/.pyvenvs/nba_dl
# -*- coding: utf-8 -*-

"""Module Summary

Additional details...

Functions
-----------
f(str, int): finds the int in the str and returns the index
"""

import os
import sys
import bisect
import numpy as np
from datetime import datetime
from collections import OrderedDict

## NBA APIs
from nba_api.stats.endpoints import boxscoreadvancedv2
from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.static import teams
from nba_api.stats.static import players


SEASON = "2017-18"
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


def win_loss_per_roster():
    data = []
    for spec_game_id in list_game_ids:
        print(f"Current Game ID {spec_game_id}")
        boxscore = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=spec_game_id)
        converted_boxscore = json.loads(boxscore.get_json())
        # print (f'Type of converted boxscore {type(converted_boxscore)}')
        team_one_id = ""
        team_one_players = []
        team_two_players = []
        for playerInfo in converted_boxscore["resultSets"][0]["rowSet"]:
            if team_one_id == "":
                team_one_id = playerInfo[1]
            playerId = playerInfo[4]
            if playerInfo[1] == team_one_id:
                team_one_players.append(playerId)
            else:
                team_two_players.append(playerId)
        traditionalBoxScore = boxscoretraditionalv2.BoxScoreTraditionalV2(
            game_id=spec_game_id
        )
        converted_traditional = json.loads(traditionalBoxScore.get_json())
        plus_minus_team_one = converted_traditional["resultSets"][0]["rowSet"][0][-1]
        team_one_win = 0
        if plus_minus_team_one > 0:
            team_one_win = 1
        data.append([spec_game_id, team_one_players, team_two_players, team_one_win])

        return data


def fill_player_dict(all_games):
    player_dict = {}
    for player in players.get_players():
        player_dict[player["id"]] = dict.fromkeys([game[1] for game in all_games])
        for game in all_games:
            player_dict[player["id"]][game[1]] = None

    return player_dict


def get_games(team_ids):
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
        print(f"{name} done...")

    # deduplicate games
    all_games = OrderedDict((x, True) for x in all_games).keys()
    return team_games, list(all_games)


def player_matrix():
    # dictionary with every player id
    team_ids = [team["id"] for team in teams.get_teams()]
    team_games, all_games = get_games(team_ids)

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
            f"Running calculations for {teams.find_team_name_by_id(team_id)['full_name']}..."
        )
        game_set = game_info[1]
        i = 0
        for game_id in game_set:
            print(f"{i} games done")
            i += 1
            boxscore = boxscoreadvancedv2.BoxScoreAdvancedV2(
                game_id=game_id
            ).get_dict()["resultSets"][0]["rowSet"]
            for player_line in boxscore:
                if player_line[BOXSCORE_TEAM_ID] == team_id:
                    # update player matrix
                    if player_line[BOXSCORE_STAT_START] is None:
                        # DNP case
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

    # The final dictionary is: (num_players, num_games, num_stats)
    # each player id and game id is a key. The stats are stored in a numpy array
    return players_dict


def main():
    return player_matrix(), win_loss_per_roster()


# This is a test example with Taurean Prince (ATL). His PID is 1627752
if __name__ == "__main__":
    pd = player_matrix()
    i = 0
    for gid in pd[1627752]:
        if pd[1627752][gid] is not None:
            i += 1
            print(float(pd[1627752][gid][0]))
    print(i)
