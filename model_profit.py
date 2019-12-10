import matplotlib.pyplot as plt


def profit_function(pred, outcome, moneyline):
    BETSIZE = 100
    odds = implied_odds(moneyline)
    profit = 0

    if pred <= odds:
        # bet on away team
        bet = abs(pred - odds) * BETSIZE

        if outcome == 0:  # home team loses

            profit = moneyline_profit(bet, moneyline)

        if outcome == 1:  # home team wins
            profit = -bet

    elif pred > odds:  # bet on home team
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
    for prediction, outcome, moneyline in zip(predictions, outcomes, moneylines):
        p, b = profit_function(prediction, outcome, moneyline)
        print("bet", str(b[0])[:4], " |||  profit", str(p[0])[:5])

        profit += p[0]
        bet += b[0]
        profits.append(profit)

    print("Final Profit:", profit, "Total bet size:", bet)
    print(f"ROI as a percent: {(profit * 100) /bet}")

    # plt.plot(profits)
    # plt.ylabel("Dollars ($)")
    # plt.xlabel("Number of Games")
    # plt.show()
