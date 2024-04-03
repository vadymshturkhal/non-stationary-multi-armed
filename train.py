from agents.tdzero import TDZero
from agents.agents import NonStationaryAgent
from game_environment import MultiArmedGame
from settings import START_POINT
from utils import DB_Operations


class TrainAgent:
    def __init__(self, game, main_agent, bet_agent):
        self.game = game
        self.main_agent = main_agent
        self.bet_agent = bet_agent
        self._rewards = []
        self._betting = []
        self._loss = []

    def train(self, games=1000, is_load_bet_weights=False):
        cost = 0
        db_operations = DB_Operations(is_clear=not is_load_bet_weights)

        for game in range(games):
            game_reward = self._train_single_game()
            game_reward = game_reward - START_POINT
            cost += game_reward
            
            db_operations.add_epoch_to_db(game, game_reward, self.game, self._rewards, self._betting, self._loss)
            self.game.reset()
            # self.bet_agent.train_epoch()

            if game_reward > START_POINT:
                self.bet_agent.save(self.game.game_stats)
                print(f'{game_reward=}, {game=}, saved')
            else:
                print(f'{game_reward=}, {game=}')

        return cost

    def _train_single_game(self):
        self._rewards.clear()
        self._betting.clear()
        self.game.reset()
        is_game_end = False

        while not is_game_end:
            state = self.game.get_state()

            choose_dealer = self.main_agent.choose_action()
            action_bet = self.bet_agent.choose_action()

            reward = self.game.apply_action(choose_dealer, bet=action_bet)

            self.game.play_step()

            last_bet = self.game.last_bet
            is_game_end = self.game.update_points(last_bet, reward)

            state_next = self.game.get_state()

            self.main_agent.update_estimates(choose_dealer, reward - last_bet)
            loss = self.bet_agent.update_estimates(state, reward - last_bet, state_next, is_game_end)

            self._loss.append(loss)
            self._rewards.append(reward)
            self._betting.append(last_bet)

        return self.game.points


if __name__ =='__main__':
    k = 1  # Number of arms
    epsilon = 0.1
    alpha = 0.01
    gamma = 0.4
    games = 1000
    is_load_bet_weights = False

    game = MultiArmedGame(k, speed=60, is_rendering=False) 
    main_agent = NonStationaryAgent(k, epsilon, alpha)
    bet_agent = TDZero(game, alpha, epsilon, gamma, is_load_weights=is_load_bet_weights)

    ta = TrainAgent(game=game, main_agent=main_agent, bet_agent=bet_agent)
    print(ta.train(games=games))
