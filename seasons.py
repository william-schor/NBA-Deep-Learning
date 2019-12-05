## THIS IS DEPRECATED!!!

from basketball_reference_web_scraper import client
import json
import datetime
from nba_api.stats.static import teams


def read(filename, all_games):
    season_dict = {}
    with open(filename, "r") as f:
        data = json.load(f)
        i = 0
        for date, game in all_games:
            # remove start time (not needed here)
            data[i].pop("start_time")
            home_team_win = (
                True
                if data[i].pop("home_team_score") - data[i].pop("away_team_score") > 0
                else False
            )
            data[i]["home_team_wins"] = home_team_win
            season_dict[game] = data[i]
            i += 1
    return season_dict


def download(outfile_name):
    schedule = client.season_schedule(season_end_year=2018)

    for game in schedule:
        for key in game:
            if isinstance(game[key], datetime.datetime):
                game[key] = game[key].__str__()
            elif key == "away_team" or key == "home_team":
                name = game[key].__str__().split(".")[1].replace("_", " ")
                game[key] = teams.find_teams_by_full_name(name)[0]["id"]

    with open(outfile_name, "w") as f:
        f.write(json.dumps(schedule))
    print("file downloaded!")
