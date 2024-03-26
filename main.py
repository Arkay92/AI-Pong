import turtle, random, time, json, os, threading, multiprocessing, argparse

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

model_path = 'pong_ai_model.keras'
model = None if not os.path.exists(model_path) else tf.keras.models.load_model(model_path)
turtle_initialized = False
win = turtle.Screen()

INITIAL_BALL_SPEED = 5.5
POINTS_TO_WIN = 2

# Define game_data_queue at the global level for inter-process communication
game_data_queue = multiprocessing.Queue()

# Set up argument parser
parser = argparse.ArgumentParser(description='Run Pong game with optional AI control for both paddles.')
parser.add_argument('--auto', action='store_true', help='Enable AI control for both paddles.')

# Parse arguments
args = parser.parse_args()

# Flag to indicate if AI should control both paddles
ai_controls_both = args.auto

def initialize_game():
    global turtle_initialized
    if not turtle_initialized:
        global win, score_display, left_paddle, right_paddle, ball, game_data, ai_delay, player_score, computer_score, model
        game_data, ai_delay = [], 0
        win.title("Python Pong")
        win.bgcolor("black")
        win.setup(width=600, height=600)

        player_score, computer_score = 0, 0
        score_display = turtle.Turtle(visible=False)
        score_display.color("white")
        score_display.penup()
        score_display.goto(0, 260)
        update_score(0, 0)  # Initialize score display

        left_paddle, right_paddle = create_paddle(-250), create_paddle(250)
        ball = create_ball()
        turtle_initialized = True

def create_paddle(x):
    paddle = turtle.Turtle(shape="square", visible=False)
    paddle.color("white")
    paddle.shapesize(stretch_wid=6, stretch_len=1)
    paddle.penup()
    paddle.goto(x, 0)
    paddle.showturtle()
    return paddle

def create_ball():
    ball = turtle.Turtle(shape="square", visible=False)
    ball.color("white")
    ball.penup()
    ball.goto(0, 0)
    ball.dx, ball.dy = INITIAL_BALL_SPEED, -INITIAL_BALL_SPEED
    ball.showturtle()
    return ball

def paddle_movement(paddle, direction):
    if not ai_controls_both:  # Skip if AI controls both paddles
        y = paddle.ycor() + (20 * direction)
        if -240 <= y <= 250:
            paddle.sety(y)
            record_event(paddle, y)

def record_event(paddle, new_y):
    # Determine the event type based on whether the paddle is moving up or down
    event_type = 'left_paddle_up' if new_y > paddle.ycor() else 'left_paddle_down'
    event_data = {
        'event': event_type,
        'paddle_y': new_y,
        'ball_pos': (ball.xcor(), ball.ycor()),
        'ball_dx': ball.dx,
        'ball_dy': ball.dy,
        'timestamp': time.time(),
        'label': 'paddle_up' if new_y > paddle.ycor() else 'paddle_down'
    }
    game_data_queue.put(event_data)  # Use queue to collect game data

if not ai_controls_both:
    win.listen()
    win.onkeypress(lambda: paddle_movement(left_paddle, 1), "w")
    win.onkeypress(lambda: paddle_movement(left_paddle, -1), "s")
    win.onkeypress(lambda: paddle_movement(right_paddle, 1), "Up")
    win.onkeypress(lambda: paddle_movement(right_paddle, -1), "Down")

# Modified functions to move paddles to record events
def left_paddle_up():
    y = left_paddle.ycor()
    if y < 250:
        new_y = y + 20
        left_paddle.sety(new_y)
        record_event(left_paddle, new_y)

def left_paddle_down():
    y = left_paddle.ycor()
    if y > -240:
        new_y = y - 20
        left_paddle.sety(new_y)
        record_event(left_paddle, new_y)
        
def right_paddle_up():
    y = right_paddle.ycor()
    if y < 250:
        new_y = y + 20
        right_paddle.sety(new_y)
        record_event(right_paddle, new_y)

def right_paddle_down():
    y = right_paddle.ycor()
    if y > -240:
        new_y = y - 20
        right_paddle.sety(new_y)
        record_event(right_paddle, new_y)

# Modified update_score function to record score updates
def update_score(player, computer):
    global player_score, computer_score
    player_score, computer_score = player, computer
    score_display.clear()
    score_display.write(f"Computer: {computer}  Player: {player}", align="center", font=("Courier", 24, "normal"))

def reset_game():
    global player_score, computer_score, ball, left_paddle, right_paddle
    player_score, computer_score = 0, 0
    ball.goto(0, 0)
    ball.dx = 5.5 * random.choice([-1, 1])
    ball.dy = 5.5 * random.choice([-1, 1])
    left_paddle.goto(-250, 0)
    right_paddle.goto(250, 0)
    update_score(player_score, computer_score)  # Refresh the score display

def declare_winner(winner):
    print(f"{winner} wins!")
    score_display.clear()
    score_display.write(f"{winner} wins!", align="center", font=("Courier", 24, "normal"))
    time.sleep(2)
    # Check if the queue has data to train on
    if not game_data_queue.empty():
        print("Training model on collected game data...")
        train_model_async(game_data_queue)
    reset_game()

def train_model_async(queue):
    game_data = []
    while not queue.empty():
        game_data.append(queue.get())
    if len(game_data) > 100:
        # Convert game_data to the format expected by your training function
        # (preprocess_data, create_model, train_and_save_model functions remain the same...)
        features, labels = preprocess_data(game_data)
        if features.size > 0:
            model = get_or_create_model(features.shape[1:])
            print("Training model...")
            model.fit(features, labels, epochs=10, verbose=1)
            model.save(model_path)
            print("Model trained and saved.")
        else:
            print("No features available for training.")

def update_ai_paddle():
    global ai_delay, model  # Ensure model is accessible

    ball_x, ball_y = ball.xcor(), ball.ycor()
    paddle_y = left_paddle.ycor()

    # Initialize ai_target_y with the current paddle position as a default
    ai_target_y = paddle_y

    # Normalize features for model input
    features = np.array([[ball_x / 600, ball_y / 600, paddle_y / 300]])

    if model is not None:
        # Use the model for AI decision-making if it exists and is trained
        prediction = model.predict(features)[0]  # First and only prediction
        action = np.argmax(prediction)  # 0 for 'paddle_up', 1 for 'paddle_down'
        confidence = np.max(prediction)  # Confidence of the chosen action

        if confidence > 0.5:  # Threshold confidence level can be adjusted
            if action == 0 and paddle_y < 250:  # 'paddle_up'
                left_paddle_up()
                # Record model-based AI movement with all necessary data
                game_data_queue.put({
                    'features': features.tolist(),
                    'action': 'up',
                    'confidence': confidence,
                    'timestamp': time.time(),
                    'ball_dx': ball.dx,
                    'ball_dy': ball.dy
                })
            elif action == 1 and paddle_y > -240:  # 'paddle_down'
                left_paddle_down()
                # Record model-based AI movement with all necessary data
                game_data_queue.put({
                    'features': features.tolist(),
                    'action': 'down',
                    'confidence': confidence,
                    'timestamp': time.time(),
                    'ball_dx': ball.dx,
                    'ball_dy': ball.dy
                })
            return  # Exit the function if action was taken based on model's prediction

    # Fallback AI logic
    if ai_delay <= 0:
        difficulty = max(1, 6 - player_score // 2)
        ai_error = random.randint(-50, 50)
        if random.randint(0, difficulty) > 0:
            ai_target_y = ball.ycor() + ai_error  # Now ai_target_y is guaranteed to be defined
            ai_delay -= 0.02 * (7 - difficulty)  # Adjust ai_delay based on difficulty
            ai_delay = max(ai_delay, 0)  # Ensure ai_delay does not go below 0

            # Record fallback AI decision with all necessary data
            game_data_queue.put({
                'event': 'ai_decision',
                'ai_target_y': ai_target_y,
                'ai_error': ai_error,
                'difficulty': difficulty,
                'timestamp': time.time(),
                'ball_dx': ball.dx,
                'ball_dy': ball.dy
            })
    else:
        ai_delay -= 0.02  # Normal decrement of ai_delay

    # Use ai_target_y after it's been defined
    if abs(paddle_y - ai_target_y) > 30:
        if paddle_y < ai_target_y:
            left_paddle_up()
            # Record AI movement event with all necessary data
            game_data_queue.put({
                'event': 'ai_paddle_up',
                'paddle_y': paddle_y + 20,
                'ball_pos': (ball_x, ball_y),
                'ball_dx': ball.dx,
                'ball_dy': ball.dy,  # Corrected from 'ball_dy' to 'ball.dy'
                'timestamp': time.time()
            })
        elif paddle_y > ai_target_y:
            left_paddle_down()
            # Record AI movement event with all necessary data
            game_data_queue.put({
                'event': 'ai_paddle_down',
                'paddle_y': paddle_y - 20,
                'ball_pos': (ball_x, ball_y),
                'ball_dx': ball.dx,
                'ball_dy': ball.dy,  # Corrected from 'ball_dy' to 'ball.dy'
                'timestamp': time.time()
            })

def update_right_ai_paddle():
    global ai_delay, model  # Ensure model is accessible

    ball_x, ball_y = ball.xcor(), ball.ycor()
    paddle_y = right_paddle.ycor()

    # Initialize ai_target_y with the current paddle position as a default
    ai_target_y = paddle_y

    # Normalize features for model input
    features = np.array([[ball_x / 600, ball_y / 600, paddle_y / 300]])

    if model is not None:
        # Use the model for AI decision-making if it exists and is trained
        prediction = model.predict(features)[0]  # First and only prediction
        action = np.argmax(prediction)  # 0 for 'paddle_up', 1 for 'paddle_down'
        confidence = np.max(prediction)  # Confidence of the chosen action

        if confidence > 0.5:  # Threshold confidence level can be adjusted
            if action == 0 and paddle_y < 250:  # 'paddle_up'
                right_paddle_up()
                # Record model-based AI movement with all necessary data
                game_data_queue.put({
                    'features': features.tolist(),
                    'action': 'up',
                    'confidence': confidence,
                    'timestamp': time.time(),
                    'ball_dx': ball.dx,
                    'ball_dy': ball.dy
                })
            elif action == 1 and paddle_y > -240:  # 'paddle_down'
                right_paddle_down()
                # Record model-based AI movement with all necessary data
                game_data_queue.put({
                    'features': features.tolist(),
                    'action': 'down',
                    'confidence': confidence,
                    'timestamp': time.time(),
                    'ball_dx': ball.dx,
                    'ball_dy': ball.dy
                })
            return  # Exit the function if action was taken based on model's prediction

    # Fallback AI logic
    if ai_delay <= 0:
        difficulty = max(1, 6 - player_score // 2)
        ai_error = random.randint(-50, 50)
        if random.randint(0, difficulty) > 0:
            ai_target_y = ball.ycor() + ai_error  # Now ai_target_y is guaranteed to be defined
            ai_delay -= 0.02 * (7 - difficulty)  # Adjust ai_delay based on difficulty
            ai_delay = max(ai_delay, 0)  # Ensure ai_delay does not go below 0

            # Record fallback AI decision with all necessary data
            game_data_queue.put({
                'event': 'ai_decision',
                'ai_target_y': ai_target_y,
                'ai_error': ai_error,
                'difficulty': difficulty,
                'timestamp': time.time(),
                'ball_dx': ball.dx,
                'ball_dy': ball.dy
            })
    else:
        ai_delay -= 0.02  # Normal decrement of ai_delay

    # Use ai_target_y after it's been defined
    if abs(paddle_y - ai_target_y) > 30:
        if paddle_y < ai_target_y:
            right_paddle_up()
            # Record AI movement event with all necessary data
            game_data_queue.put({
                'event': 'ai_paddle_up',
                'paddle_y': paddle_y + 20,
                'ball_pos': (ball_x, ball_y),
                'ball_dx': ball.dx,
                'ball_dy': ball.dy,  # Corrected from 'ball_dy' to 'ball.dy'
                'timestamp': time.time()
            })
        elif paddle_y > ai_target_y:
            right_paddle_down()
            # Record AI movement event with all necessary data
            game_data_queue.put({
                'event': 'ai_paddle_down',
                'paddle_y': paddle_y - 20,
                'ball_pos': (ball_x, ball_y),
                'ball_dx': ball.dx,
                'ball_dy': ball.dy,  # Corrected from 'ball_dy' to 'ball.dy'
                'timestamp': time.time()
            })

# Save game data to a file
def save_data():
    directory = 'data/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, 'pong_game_data.json')
    with open(filepath, 'w') as file:
        json.dump(game_data, file)
    print(f"Game data saved to {filepath}")

# Save final game board snapshot
def save_canvas(win, filename):
    try:
        canvas = win.getcanvas()
        canvas.postscript(file=f"{filename}.eps")
        from PIL import Image
        img = Image.open(f"{filename}.eps")
        img.save(f"{filename}.png", "png")
    except Exception as e:
        print(f"Error saving the game board: {e}")

# Split data into training, validation, and test sets
def split_data(data, train_ratio=0.7, val_ratio=0.15):
    np.random.shuffle(data)
    n_total = len(data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * (train_ratio + val_ratio))

    train_data = data[:n_train]
    val_data = data[n_train:n_val]
    test_data = data[n_val:]

    return train_data, val_data, test_data

# Save split data to multiple files
def save_split_data(data, filename):
    with open(filename, 'w') as file:
        json.dump(data.tolist(), file)

# Adjustments in preprocess_data function to skip incomplete records
def preprocess_data(game_data):
    features, labels = [], []
    for event in game_data:
        if 'ball_pos' in event and 'paddle_y' in event and 'label' in event:
            ball_x, ball_y = event['ball_pos']
            paddle_y = event['paddle_y']
            ball_dx = event.get('ball_dx', INITIAL_BALL_SPEED)
            ball_dy = event.get('ball_dy', -INITIAL_BALL_SPEED)
            label = 1 if event['label'] == 'paddle_up' else 0
            features.append([ball_x / 600, ball_y / 600, paddle_y / 300, ball_dx / INITIAL_BALL_SPEED, ball_dy / INITIAL_BALL_SPEED])
            labels.append(label)
    return np.array(features), np.array(labels)

# Define a custom callback to print the epoch number during training
class EpochPrintCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}/{self.params['epochs']} - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f}")

# Train and Save the Model
def train_and_save_model(model, train_features, train_labels, model_path):
    # Create an instance of the custom callback
    epoch_print_callback = EpochPrintCallback()
    model.fit(train_features, train_labels, epochs=10, callbacks=[epoch_print_callback])
    model.save(model_path)
    print("Model trained and saved.")

def get_or_create_model(input_shape):
    if os.path.exists(model_path):
        print("Loading existing model...")
        return tf.keras.models.load_model(model_path)
    else:
        print("Creating new model with input shape:", input_shape)
        return create_model(input_shape)

def create_model(input_shape):
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def background_model_training():
    print("Background training started...")
    try:
        # Preprocess the game data
        features, labels = preprocess_data(game_data)

        # Debug prints to verify shapes
        print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

        if features.size > 0:
            global model
            model = get_or_create_model(features.shape[1:])
            print("Training model...")
            model.fit(features, labels, epochs=10, verbose=1)
            model.save(model_path)
            print("Model trained and saved.")
        else:
            print("No features available for training.")

    except Exception as e:
        print(f"Error during model training: {e}")
        
def main_game_loop():
    initialize_game()
    global player_score, computer_score

    while True:
        win.update()

        # Move the ball
        ball.setx(ball.xcor() + ball.dx)
        ball.sety(ball.ycor() + ball.dy)

        # Border checking
        if ball.ycor() > 290 or ball.ycor() < -290:
            ball.dy *= -1
        
        if ball.xcor() > 290:  # Computer scores
            ball.goto(0, 0)
            time.sleep(1)  # Pause for a moment before continuing
            ball.dx *= -1.05  # Slightly increase ball's speed in negative direction
            ball.dy *= 1.05  # Increase speed in the y-direction
            computer_score += 1  # Increment computer's score
            update_score(player_score, computer_score)

        if ball.xcor() < -290:  # Player scores
            ball.goto(0, 0)
            time.sleep(1)  # Pause for a moment before continuing
            ball.dx *= 1.05  # Slightly increase ball's speed
            ball.dy *= 1.05  # Increase speed in the y-direction
            player_score += 1  # Increment player's score
            update_score(player_score, computer_score)
            ball.dx = -ball.dx  # Reset ball direction to the left

        # Paddle and ball collisions
        if (ball.xcor() > 240 and ball.xcor() < 250) and (ball.ycor() < right_paddle.ycor() + 50 and ball.ycor() > right_paddle.ycor() - 50):
            ball.setx(240)
            ball.dx *= -1
        
        if (ball.xcor() < -240 and ball.xcor() > -250) and (ball.ycor() < left_paddle.ycor() + 50 and ball.ycor() > left_paddle.ycor() - 50):
            ball.setx(-240)
            ball.dx *= -1

        # Update AI paddle movement
        update_ai_paddle()

        if ai_controls_both:
            # Update right paddle with AI logic as well if --auto flag is set
            update_right_ai_paddle()

        # Periodically save data, e.g., every 100 events
        if len(game_data) % 100 == 0:
            save_data()

        if player_score >= POINTS_TO_WIN or computer_score >= POINTS_TO_WIN:
            declare_winner("Player" if player_score >= POINTS_TO_WIN else "Computer")
            reset_game()

def main():
    # Any initialization code you have before the game loop starts
    main_game_loop()

    if not game_data_queue.empty():
        train_model_async(game_data_queue)

if __name__ == '__main__':
    main()