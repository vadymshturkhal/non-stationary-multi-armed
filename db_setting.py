DB_NAME="multi_armed"
USER="agent"
PASSWORD="5271"
HOST="localhost"
PORT=5432

"""
CREATE TABLE epochs (
    fk_epoch_id SERIAL PRIMARY KEY,
    reward INT,
    description TEXT,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
"""

"""
CREATE TABLE train_data (
    data_id SERIAL PRIMARY KEY,
    step INT,
    bet INT,
    reward INT,
    points INT,
    loss FLOAT,
    fk_epoch_id INT,
    FOREIGN KEY (fk_epoch_id) REFERENCES epochs(fk_epoch_id)
);
"""

"GRANT ALL PRIVILEGES ON DATABASE multi_armed TO agent;"
"GRANT ALL PRIVILEGES ON TABLE train_data TO agent;"
"GRANT ALL PRIVILEGES ON TABLE epochs TO agent;"
"GRANT USAGE, SELECT ON SEQUENCE train_data_data_id_seq TO agent;"
"GRANT USAGE, SELECT ON SEQUENCE epochs_fk_epoch_id_seq TO agent;"

"""
ALTER TABLE train_data ADD CONSTRAINT fk_epoch_id FOREIGN KEY (fk_epoch_id) REFERENCES epochs(fk_epoch_id) ON DELETE CASCADE;
"""

"ALTER SEQUENCE epochs_fk_epoch_id_seq RESTART WITH 1;"
"ALTER TABLE epochs OWNER TO agent;"
"ALTER TABLE train_data OWNER TO agent;"
