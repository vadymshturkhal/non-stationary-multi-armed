from agents.expected_sarsa import ExpectedSARSA
from agents.qlearning import QLearning
from agents.tdzero import TDZero
from agents.sarsa import SARSA
from agents.agents import NonStationaryAgent
from game_environment import MultiArmedGame
from settings import END_MULTIPLIER, REWARD_WIN, REWARD_LOOSE, START_POINT
from utils import DB_Operations


class TrainAgent:
    def __init__(self, game, main_agent, bet_agent):
        self.game = game
        self.main_agent = main_agent
        self.bet_agent = bet_agent
        self._states = []
        self._actions = []
        self._rewards = []
        self._betting = []

    def train(self, games=1000, is_load_bet_weights=False):
        cost = 0
        db_operations = DB_Operations(is_clear=not is_load_bet_weights)

        games = self.game.total_games_remain
        for game in range(games):
            game_reward = self._train_single_game()

            # Train Agent
            episode_loss = self.bet_agent.update_episode_estimates(self._states, self._actions, self._rewards)

            game_reward = game_reward - START_POINT
            cost += game_reward
            
            db_operations.add_epoch_to_db(game, game_reward, self.game, self._rewards, self._betting, episode_loss)

            # Save model
            if game_reward + START_POINT > START_POINT * END_MULTIPLIER:
                self.bet_agent.save(self.game.game_stats)
                print(f'{game_reward=}, {game=}, saved')
            else:
                print(f'{game_reward=}, {game=}')
            
            self._clear_epoch_data()

        return cost

    def _train_single_game(self):
        self.game.reset()
        is_game_end = False

        while not is_game_end:
            state = self.game.get_state()

            choose_dealer = self.main_agent.choose_action()
            action_bet = self.bet_agent.choose_action(state)

            reward = self.game.apply_action(choose_dealer, bet=action_bet)

            self.game.play_step()

            last_bet = self.game.last_bet
            is_game_end = self.game.update_points(last_bet, reward)

            self.main_agent.update_estimates(choose_dealer, reward - last_bet)

            agent_reward = reward - last_bet

            self._states.append(state)
            self._actions.append(action_bet)
            self._rewards.append(agent_reward)
            self._betting.append(last_bet)

        return self.game.points

    def _clear_epoch_data(self):
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._betting.clear()


if __name__ =='__main__':
    k = 1  # Number of arms
    min_epsilon = 0.1
    alpha = 0.5
    gamma = 0.9
    games = 400
    is_load_bet_weights = False

    game = MultiArmedGame(k, total_games=games, speed=60, is_rendering=False) 
    main_agent = NonStationaryAgent(k, min_epsilon, alpha)
    bet_agent = ExpectedSARSA(game, alpha, min_epsilon, gamma, is_load_weights=is_load_bet_weights)

    ta = TrainAgent(game=game, main_agent=main_agent, bet_agent=bet_agent)
    print(ta.train())
