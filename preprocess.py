from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.endpoints import boxscoreadvancedv2
import pandas
# Anthony Davis
career = playercareerstats.PlayerCareerStats(player_id='203076')
# print(career.get_json())


game_id = '0021700807'
game_advanced_stats = boxscoreadvancedv2.player_stats(end_period=0, end_range=None, game_id=game_id, range_type=None, start_period=0, start_range=None)
game_advanced_stats = game_advanced_stats.get_json()
print(game_advanced_stats)


