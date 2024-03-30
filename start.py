from agent import NonStationaryAgent, NonStationaryAgentBet, NonStationaryAgentUCB
from game_environment import MultiArmedGame
from plot_data import plot_rewards
from settings import START_POINT, nonstationary_bandit_data_average_reward, BET
from utils import DB_Operations, create_average_data


def train(game: MultiArmedGame, main_agent, support_agent, steps=1000):
    rewards.clear()
    betting.clear()
    for _ in range(steps):
        choose_dealer = main_agent.choose_action()
        choose_bet = support_agent.choose_action()

        reward = game.apply_action(choose_dealer, bet=choose_bet)
        game.play_step()
        last_bet = game.last_bet

        main_agent.update_estimates(choose_dealer, reward - last_bet)
        support_agent.update_estimates(choose_bet, reward - last_bet)
        is_end = support_agent.update_points(last_bet, reward)

        rewards.append(reward)
        betting.append(last_bet)

        if is_end:
            break
    return support_agent.points

def start_epoch(main_agent, main_agent_params, support_agent, support_agent_params):
    cost = 0
    db_operations = DB_Operations()
    for epoch in range(epochs):
        game = MultiArmedGame(k, speed=60, is_rendering=False)
        agent_instance = main_agent(*main_agent_params)
        support_agent_instance = support_agent(*support_agent_params)

        epoch_reward = train(game, agent_instance, support_agent_instance, steps)
        if epoch_reward < START_POINT:
            epoch_reward = epoch_reward - START_POINT

        cost += epoch_reward
        create_average_data(nonstationary_bandit_data_average_reward, support_agent_instance, rewards, betting)
        db_operations.add_epoch_to_db(epoch, epoch_reward, support_agent_instance, rewards, betting)

    return cost


if __name__ =='__main__':
    k = 1  # Number of actions (bandits)
    epsilon = 0.2  # Exploration probability
    alpha = 0.5
    steps = 1000
    epochs = 40
    rewards = []
    betting = []

    main_agent = NonStationaryAgent
    main_agent_params = [k, epsilon, alpha]

    support_agent = NonStationaryAgentBet
    support_agent_params = [len(BET), epsilon, alpha]

    # main_agent = NonStationaryAgentUCB
    # main_agent_params = [k, alpha]

    # support_agent = NonStationaryAgentUCB
    # support_agent_params = [k, alpha]

    print(start_epoch(main_agent, main_agent_params, support_agent, support_agent_params))
    # plot_rewards(nonstationary_bandit_data_average_reward)
