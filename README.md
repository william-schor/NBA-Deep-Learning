# NBA Deep Learning


## How to get the data

Running `python preprocess.py` should create all the data required to run the model.
It will take a few minutes to run. It takes all the boxscores in games2017 and
games2018 and synthesizes it into our data format. 

Note: You may need to make a folder called "final_data". While testing, we have found that on some systems, python can create the directory, while on others, it will throw an error if the directory does not exist. 


## How to run the model

Run `python model_framework.py --[model option]` will run the model. 

* The flag `--dense` uses the dense model.
* The flag `--team_conv` uses the team level convolution model
* the flag `--player_conv` uses the player level convolution model