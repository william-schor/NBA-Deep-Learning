def loss(line, team1_wins, prediction):		
	if(line > 0):
		line = abs(line)
		team1_odds = (100 / (100 + line))
	else: 
		line = abs(line)
		team1_odds = (line/(100 + line))

	if (team1_odds > prediction):
		#bet on team 2
		if(team1_wins == True):
			profit = -BET_SIZE
		else:
			profit = (us2dec(line) - 1) * BET_SIZE
	else:
		#bet on team 1
		if(team1_wins == True):
			profit = (us2dec(line) - 1) * BET_SIZE
		else:
			profit = -BET_SIZE
	return 1/(profit + BET_SIZE + 1) ## want to minimize
		

		
def us2dec(line):
	if (line > 0):
		dec = 1 + (line / 100)
	else:
		dec = 1 - (100 / line)
	return dec
