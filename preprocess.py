from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.endpoints import boxscoreadvancedv2
import pandas
# Anthony Davis
career = playercareerstats.PlayerCareerStats(player_id='203076')
# print(career.get_json())


game_id = '0021700807'
game_advanced_stats = boxscoreadvancedv2.BoxScoreAdvancedV2(end_period=0, game_id=game_id, start_period=0)
print(game_advanced_stats.get_json())


