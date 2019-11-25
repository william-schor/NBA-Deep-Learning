from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.endpoints import boxscoreadvancedv2
import pandas
# Anthony Davis
career = playercareerstats.PlayerCareerStats(player_id='203076')
print(career.get_json())

