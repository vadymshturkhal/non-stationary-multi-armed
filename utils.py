import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extensions import register_adapter, AsIs
from db_setting import DB_NAME, USER, PASSWORD, HOST, PORT


def create_bar_data(file_path: str, k: int, agent, true_reward_probabilities):
    # Creating a DataFrame to hold the results
    results = pd.DataFrame({
        'Action': np.arange(1, k+1),
        'Estimated Action Values': agent.Q,
        'True Action Values': true_reward_probabilities,
        'Number of Times Chosen': agent.N,
    })

    # Save the DataFrame to a CSV file
    results.to_csv(file_path, index=False)
    return results

def create_average_data(file_path: str, agent, rewards, betting):
    # Creating a DataFrame to hold the results
    results = pd.DataFrame({
        'Steps': np.arange(1, len(rewards) + 1),
        'Bet': betting,
        'Reward': rewards,
        'Points': agent.rewards,
        'Average': np.cumsum(rewards) / np.arange(1, len(rewards) + 1),
    })

    # Save the DataFrame to a CSV file
    results.to_csv(file_path, index=False)
    return results

class DB_Operations():
    def __init__(self, is_clear=True, total_epochs=0):
        self._total_epochs = total_epochs
        self._conn = psycopg2.connect(
            dbname=DB_NAME, 
            user=USER, 
            password=PASSWORD, 
            host=HOST, 
            port=PORT,
        )

        if is_clear:
            self._clear_epochs()
        register_adapter(np.int64, self.addapt_numpy_int64)

    def _clear_epochs(self):
        cur = self._conn.cursor()
        cur.execute("DELETE FROM average_data;")
        cur.execute("DELETE FROM epochs;")
        cur.execute("ALTER SEQUENCE epochs_fk_epoch_id_seq RESTART WITH 1;")
        cur.execute("ALTER SEQUENCE average_data_data_id_seq RESTART WITH 1;")
        self._conn.commit()

    # Function to adapt np.int64
    def addapt_numpy_int64(self, numpy_int64):
        return AsIs(numpy_int64)

    def add_epoch_to_db(self, epoch, epoch_reward, game, rewards, betting):
        # Register the adapter
        cur = self._conn.cursor()

        cur.execute("INSERT INTO epochs (reward, description) VALUES (%s, %s)", (epoch_reward, ''))
        self._conn.commit()

        points = game.rewards
        average = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)

        for step in range(len(rewards)):
            cur.execute(
            "INSERT INTO average_data (step, bet, reward, points, average, fk_epoch_id) VALUES (%s, %s, %s, %s, %s, %s);",
            (step + 1, betting[step], rewards[step], points[step], average[step], epoch + 1)
        )

        self._conn.commit()
        self._reduce_epochs()

    def _reduce_epochs(self):
        self._total_epochs -= 1
        if self._total_epochs == 0:
            # Close cursor and connection
            cur = self._conn.cursor()
            cur.close()
            self._conn.close()

    def get_epoch_rewards(self, epochs_amount:int):
        cur = self._conn.cursor()
        # cur.execute(
        #     """
        #         SELECT reward FROM (
        #         SELECT reward FROM epochs
        #         ORDER BY fk_epoch_id DESC
        #         LIMIT (%s)
        #         ) AS last_rewards
        #         ORDER BY reward ASC;
        #     """, (epochs_amount,))

        cur.execute(
            """
                SELECT reward FROM epochs;
            """, (epochs_amount,))

        rows = cur.fetchall()
        epoch_rewards = [row[0] for row in rows]

        cur.close()
        self._conn.close()

        return epoch_rewards
