from agent import NonStationaryAgent, NonStationaryAgentBet, TDZero
from game_environment import MultiArmedGame
from settings import START_POINT, BET
from utils import DB_Operations


class TrainAgent:
    def __init__(self, game, main_agent, bet_agent):
        self.game = game
        self.main_agent = main_agent
        self.bet_agent = bet_agent

    def train(self, games=1000):
        rewards.clear()
        betting.clear()
        for _ in range(games):
            state = self.bet_agent.get_state()
        
            choose_dealer = self.main_agent.choose_action()
            action_bet = self.bet_agent.choose_action()

            reward = self.game.apply_action(choose_dealer, bet=action_bet)

            self.game.play_step()

            last_bet = self.game.last_bet
            is_game_end = self.bet_agent.update_points(last_bet, reward)

            state_next = self.bet_agent.get_state()

            self.main_agent.update_estimates(choose_dealer, reward - last_bet)
            self.bet_agent.update_estimates(state, reward - last_bet, state_next, is_game_end)

            rewards.append(reward)
            betting.append(last_bet)

            if is_game_end:
                break

        return self.bet_agent.points

    def train_epoch(self, epochs_quantity=1, games_in_epoch=1000):
        cost = 0
        db_operations = DB_Operations()
        for epoch in range(epochs_quantity):
            epoch_reward = self.train(games=games_in_epoch)
            epoch_reward = epoch_reward - START_POINT

            cost += epoch_reward
            db_operations.add_epoch_to_db(epoch, epoch_reward, self.bet_agent, rewards, betting)

        return cost


if __name__ =='__main__':
    k = 1  # Number of bandits
    epsilon = 0.1  # Exploration probability
    alpha = 0.1
    games = 1000
    epochs = 1
    rewards = []
    betting = []

    game = MultiArmedGame(k, speed=60, is_rendering=False) 
    main_agent = NonStationaryAgent(k, epsilon, alpha)
    bet_agent = TDZero(len(BET), epsilon, alpha)

    ta = TrainAgent(game=game, main_agent=main_agent, bet_agent=bet_agent)
    print(ta.train_epoch(epochs_quantity=epochs, games_in_epoch=games))
