from agent import NonStationaryAgent, NonStationaryAgentBet, TDZero
from game_environment import MultiArmedGame
from settings import START_POINT, BET
from utils import DB_Operations


class TrainAgent:
    def __init__(self, game, main_agent, bet_agent):
        self.game = game
        self.main_agent = main_agent
        self.bet_agent = bet_agent
        self._rewards = []
        self._betting = []

    def train(self, games=1000):
        cost = 0
        max_game_reward = float('-inf')
        db_operations = DB_Operations()

        for game in range(games):
            game_reward = self._train_single_game()
            game_reward = game_reward - START_POINT
            cost += game_reward
            
            db_operations.add_epoch_to_db(game, game_reward, self.bet_agent, self._rewards, self._betting)
            self.bet_agent.reset_points()

            if max_game_reward < game_reward:
                self.bet_agent.save()

            self.bet_agent.train_epoch()
            print(game_reward)

        return cost

    def _train_single_game(self):
        self._rewards.clear()
        self._betting.clear()
        self.bet_agent.reset_points()
        is_game_end = False

        while not is_game_end:
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

            self._rewards.append(reward)
            self._betting.append(last_bet)

        return self.bet_agent.points


if __name__ =='__main__':
    k = 1  # Number of bandits
    epsilon = 0.1  # Exploration probability
    alpha = 0.1
    games = 1000

    is_load_bet_weights = True

    game = MultiArmedGame(k, speed=60, is_rendering=False) 
    main_agent = NonStationaryAgent(k, epsilon, alpha)
    bet_agent = TDZero(len(BET), epsilon, alpha, is_load_weights=is_load_bet_weights)

    ta = TrainAgent(game=game, main_agent=main_agent, bet_agent=bet_agent)
    print(ta.train(games=games))
