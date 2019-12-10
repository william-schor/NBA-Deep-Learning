import matplotlib.pyplot as plt


def profit_function(pred, outcome, moneyline, total_money):
    away_line, home_line = moneyline

    BETSIZE = total_money / 5

    home_odds = implied_odds(home_line)
    away_odds = implied_odds(away_line)

    odds = home_odds if abs(pred - home_odds) > abs(pred - away_odds) else away_odds
    profit = 0

    if pred <= odds:
        moneyline = away_line
        # bet on away team
        bet = abs(pred - odds) * BETSIZE

        if outcome == 0:  # home team loses

            profit = moneyline_profit(bet, moneyline)

        if outcome == 1:  # home team wins
            profit = -bet

    elif pred > odds:  # bet on home team
        moneyline = home_line
        bet = abs(pred - odds) * BETSIZE

        if outcome == 0:  # home team loses
            profit = -bet

        if outcome == 1:  # home team wins
            profit = moneyline_profit(bet, moneyline)

    return profit, bet


def implied_odds(line):
    if line > 0:
        line = abs(line)
        odds = 100 / (100 + line)
    else:
        line = abs(line)
        odds = line / (100 + line)

    return odds


def moneyline_profit(bet, moneyline):
    if moneyline < 0:  # favoite
        return bet * abs(100.0 / moneyline)

    else:  # underdog
        return bet * abs(moneyline / 100.0)


def evaluate_model(predictions, outcomes, moneylines):
    profits = []
    profit = 0
    bet = 0
    total_money = 100
    for prediction, outcome, moneyline in zip(predictions, outcomes, moneylines):
        p, b = profit_function(prediction, outcome, moneyline, total_money)
        print("bet", str(b[0])[:4], " |||  profit", str(p[0])[:5])

        profit += p[0]
        bet += b[0]
        profits.append(profit)

        if p[0] > 0:
            total_money += 0.8 * p[0]
        else:
            total_money += p[0]

    print("Final Profit:", profit, "Total bet size:", bet)
    print(f"ROI as a percent: {(profit * 100) /bet}")

    # plt.plot(profits)
    # plt.ylabel("Dollars ($)")
    # plt.xlabel("Number of Games")
    # plt.show()
