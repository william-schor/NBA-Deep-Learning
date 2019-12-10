import os
import sys
import bisect
import json
import numpy as np
import file_dumps
import preprocess
from datetime import datetime
from collections import OrderedDict

## NBA APIs
from nba_api.stats.endpoints import boxscoreadvancedv2
from nba_api.stats.endpoints import boxscoresummaryv2
from nba_api.stats.endpoints import leaguegamefinder
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

def get_2d_data(wl_per_rosters, player_matrix):
    data = []
    games = []

    for game in wl_per_rosters:
        # home roster
        row = np.zeros((26, 23), dtype='float32')
        i = 0
        for player in game[1]:
            if player in player_matrix.keys():
                stats = np.array(player_matrix[player][game[0]], dtype='float32')
            row[i] = stats
            i += 1
        for player in game[2]:
            if player in player_matrix.keys():
                stats = np.array(player_matrix[player][game[0]], dtype='float32')
            row[i] = (stats)
            i += 1

        row = np.array(row).flatten()
        data.append(row)
        games.append(int(game[0]))

    return np.array(data), games

def get_today_input_data():
  game_ids = ["0021900343", "0021900344", "0021900345", "0021900346", "0021900347", \
                "0021900348", "0021900349", "0021900350", "0021900351"]
  data = []
  players_dict = {}
  for gid in game_ids:
    print(str(gid) + " fetch 1...")
    boxscore = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=gid).get_dict()["resultSets"][0]["rowSet"]
    print("done. fetch 2...")
    home_team_id = boxscoresummaryv2.BoxScoreSummaryV2(game_id=gid).get_dict()["resultSets"][0]["rowSet"][0][6] if gid != "0021900351" else 1610612744
    print("done.")
    home_team_players = []
    away_team_players = []
    for player_info in boxscore:
        if str(player_info[BOXSCORE_TEAM_ID]) == str(home_team_id):
            home_team_players.append(player_info[BOXSCORE_PLAYER_ID])
        else:
            away_team_players.append(player_info[BOXSCORE_PLAYER_ID])
    data.append([gid, home_team_players, away_team_players, 0])

    for player_line in boxscore:
        # update player matrix
        if player_line[BOXSCORE_STAT_START] is None:
            # DNP (coach's decision) case
            continue
        else:
          players_dict[player_line[BOXSCORE_PLAYER_ID]] = {}
        player_line[BOXSCORE_STAT_START] = int(
            player_line[BOXSCORE_STAT_START].split(":")[0]
        ) * 60 + int(player_line[BOXSCORE_STAT_START].split(":")[1])

        players_dict[player_line[BOXSCORE_PLAYER_ID]][gid] = np.array(
            player_line[BOXSCORE_STAT_START:]
        )
  all_data, games = get_2d_data(data, players_dict)
  return all_data
