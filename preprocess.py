from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.endpoints import boxscoreadvancedv2
from nba_api.stats.endpoints import teamgamelog
import json

import pandas
# Anthony Davis
career = playercareerstats.PlayerCareerStats(player_id='203076')
# print(career.get_json())


game_id = '0021700807'
game_advanced_stats = boxscoreadvancedv2.BoxScoreAdvancedV2(end_period=0, game_id=game_id, start_period=0)
# print(game_advanced_stats.get_json())

team_game_log = teamgamelog.TeamGameLog(season='2018-19', season_type_all_star='Regular Season', team_id='1610612739')
team_game_log = json.loads(team_game_log.get_json())
print(team_game_log)

