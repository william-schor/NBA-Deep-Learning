from nba_api.stats.endpoints import playercareerstats
import pandas
# Anthony Davis
career = playercareerstats.PlayerCareerStats(player_id='203076')
print(career.get_json())