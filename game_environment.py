import pygame
import sys
from game_stats import GameStats

from rewards import DealerRewards
from settings import BET, END_MULTIPLIER, MIN_POINTS_MULTIPLIER, START_POINT


# Screen dimensions
WIDTH = 600
HEIGHT = 400

# Colors
background_color = (30, 30, 30)
text_color = (255, 255, 255)


class MultiArmedGame:
    def __init__(self, k, total_games, speed=30, is_rendering=True, ai_agent=None):
        self.last_dealer = 0
        self.k = k
        self.total_games_remain = total_games
        self.game_speed = speed
        self.is_rendering = is_rendering
        self.ai_agent = ai_agent
        self.dealers = [DealerRewards() for _ in range(k)]
        self.game_stats = GameStats()
        self.points = START_POINT
        self.last_bet = 0  
        self.rewards = []

        self._upper_bound = START_POINT * END_MULTIPLIER
        self._lower_bound = START_POINT * MIN_POINTS_MULTIPLIER  

        # init display
        if self.is_rendering:
            pygame.init()
            self.display = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption('Multi Armed')
            self.clock = pygame.time.Clock()

            # Fonts
            pygame.font.init()  # Initialize font module
            self.font = pygame.font.Font(None, 36)

    def reset(self):
        self.points = START_POINT
        self.last_bet = 0 
        self.rewards.clear()
        self.total_games_remain -= 1

        if self.total_games_remain < 0:
            raise Exception('self.total_games_remain < 0')

    def play_step(self):
        if self.is_rendering:
            self._handle_events()
            self._update_ui()
            self.clock.tick(self.game_speed)
            pygame.display.flip()

    def apply_action(self, action: int, bet:int):
        if action < 0:
            raise Exception(f'Unavailable action {action}')
        
        if action <= self.k:
            self.last_dealer = action + 1
            self.last_bet = BET[bet]
            self.last_reward = self.dealers[action].get_reward(bet)

            # Game stats
            self.game_stats.add_hand_reward(self.last_bet, bet, self.last_reward)

            return self.last_reward
        else:
            raise Exception(f"Can't pull arm {action}")
    
    def get_state(self):
        return self.game_stats.get_state(points=self.points)

    def update_points(self, bet, reward):
        self.points += reward - bet
        self.rewards.append(self.points)
        if self.points >= self._upper_bound:
            return True

        if self.points <= self._lower_bound:
            return True
        
        return False

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print('Quit')
                pygame.quit()
                sys.exit()
            
            if self.ai_agent is None:
                print()
                if event.type == pygame.KEYDOWN:
                    if pygame.K_1 <= event.key <= pygame.K_9:
                        chosen_bandit = event.key - pygame.K_1
                        self.apply_action(chosen_bandit)

    def _draw_text(self, text, position):
        text_surface = self.font.render(text, True, text_color)
        rect = text_surface.get_rect(center=position)
        self.display.blit(text_surface, rect)

    def _update_ui(self):
        # Fill the background
        self.display.fill(background_color)

        # Display instructions
        self._draw_text("Press keys 1-{} to pull a bandit's lever".format(self.k), (WIDTH // 2, 50))
        # Display scores
        self._draw_text("Total Score: {}".format(self.total_score), (WIDTH // 2, 150))
        self._draw_text("Last Reward: {}".format(self.last_reward), (WIDTH // 2, 200))
        if self.last_choice is not None:
            self._draw_text("Last Bandit Chosen: {}".format(self.last_choice + 1), (WIDTH // 2, 250))
