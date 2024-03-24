import turtle, random, time, json, os, threading
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

BALL_SPEED = 5.5
POINTS_NEEDED = 2

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

model_path = 'pong_ai_model.keras'
model = None

# Initialize game data list and set up the window
game_data, game_on, ai_delay = [], True, 0
win = turtle.Screen()
win.title("Python Pong")
win.bgcolor("black")
win.setup(width=600, height=600)

# Score variables and display
player_score, computer_score = 0, 0
score_display = turtle.Turtle()
score_display.speed(0)
score_display.color("white")
score_display.penup()
score_display.hideturtle()
score_display.goto(0, 260)
score_display.write("Computer: 0  Player: 0", align="center", font=("Courier", 24, "normal"))

# Left paddle
left_paddle = turtle.Turtle()
left_paddle.speed(0)
left_paddle.shape("square")
left_paddle.color("white")
left_paddle.shapesize(stretch_wid=6, stretch_len=1)
left_paddle.penup()
left_paddle.goto(-250, 0)

# Right paddle
right_paddle = turtle.Turtle()
right_paddle.speed(0)
right_paddle.shape("square")
right_paddle.color("white")
right_paddle.shapesize(stretch_wid=6, stretch_len=1)
right_paddle.penup()
right_paddle.goto(250, 0)

# Ball
ball = turtle.Turtle()
ball.speed(40)
ball.shape("square")
ball.color("white")
ball.penup()
ball.goto(0, 0)
ball.dx = BALL_SPEED  # Increased initial ball speed
ball.dy = -BALL_SPEED

# Modified functions to move paddles to record events
def left_paddle_up():
    y = left_paddle.ycor()
    if y < 250:
        left_paddle.sety(y + 20)
        game_data.append({
            'event': 'left_paddle_up',
            'paddle_y': y + 20,
            'ball_pos': (ball.xcor(), ball.ycor()),
            'timestamp': time.time(),
            'label': 'paddle_up'  # Add action label
        })

def left_paddle_down():
    y = left_paddle.ycor()
    if y > -240:
        left_paddle.sety(y - 20)
        game_data.append({
            'event': 'left_paddle_down',
            'paddle_y': y - 20,
            'ball_pos': (ball.xcor(), ball.ycor()),
            'timestamp': time.time(),
            'label': 'paddle_down'
        })
        
def right_paddle_up():
    y = right_paddle.ycor()
    if y < 250:
        right_paddle.sety(y + 20)
        game_data.append({
            'event': 'right_paddle_up',
            'paddle_y': y + 20,
            'ball_pos': (ball.xcor(), ball.ycor()),
            'timestamp': time.time(),
            'label': 'paddle_up'  # Add action label
        })

def right_paddle_down():
    y = right_paddle.ycor()
    if y > -240:
        right_paddle.sety(y - 20)
        game_data.append({
            'event': 'right_paddle_down',
            'paddle_y': y - 20,
            'ball_pos': (ball.xcor(), ball.ycor()),
            'timestamp': time.time(),
            'label': 'paddle_down'  # Add action label
        })
        
# Keyboard bindings for graceful exit
def end_game():
    global game_on
    game_on = False

# You can also add another key binding to quit the game, e.g., 'Esc' key
def quit_game():
    global game_on
    game_on = False

win.listen()
win.onkeypress(end_game, "q")  # Press 'q' to quit the game
win.onkeypress(left_paddle_up, "w")
win.onkeypress(left_paddle_down, "s")
win.onkeypress(right_paddle_up, "Up")
win.onkeypress(right_paddle_down, "Down")
win.onkeypress(quit_game, "Escape")  # Press 'Escape' to quit the game

# Modified update_score function to record score updates
def update_score(player, computer):
    global player_score, computer_score, game_on
    player_score, computer_score = player, computer
    score_display.clear()
    score_display.write("Computer: {}  Player: {}".format(computer, player), align="center", font=("Courier", 24, "normal"))

    # Check for a win
    if player_score >= POINTS_NEEDED:
        declare_winner("Player")
    elif computer_score >= POINTS_NEEDED:
        declare_winner("Computer")

def reset_game():
    global player_score, computer_score, ball, left_paddle, right_paddle
    player_score, computer_score = 0, 0
    ball.goto(0, 0)
    ball.dx = BALL_SPEED * random.choice([-1, 1])
    ball.dy = BALL_SPEED * random.choice([-1, 1])
    left_paddle.goto(-250, 0)
    right_paddle.goto(250, 0)
    score_display.clear()
    score_display.write("Computer: 0  Player: 0", align="center", font=("Courier", 24, "normal"))

def declare_winner(winner):
    global countdown_timer
    score_display.clear()
    score_display.goto(0, 260)
    score_display.write(f"{winner} wins!", align="center", font=("Courier", 32, "normal"))
    countdown_timer = 120  # Sets a 2-second countdown assuming a 60 FPS game loop

# Update AI paddle movement
def update_ai_paddle():
    global ai_delay, ai_target_y, model  # Ensure model and ai_target_y are accessible

    ball_x, ball_y = ball.xcor(), ball.ycor()
    paddle_y = left_paddle.ycor()

    # Initialize ai_target_y at the beginning of the function
    # You can set it to the current paddle position as a default
    ai_target_y = paddle_y

    # Normalize features
    features = np.array([[ball_x / 600, ball_y / 600, paddle_y / 300]])

    if model is not None:
        # Use the model for AI decision-making if it exists and is trained

        # Get prediction and confidence from the model
        prediction = model.predict(features)[0]  # First and only prediction
        action = np.argmax(prediction)  # 0 for 'paddle_up', 1 for 'paddle_down'
        confidence = np.max(prediction)  # Confidence of the chosen action

        # Execute the action and record the event with confidence score
        if action == 0 and paddle_y < 250:  # 'paddle_up'
            left_paddle_up()
            game_data.append({'features': features.tolist(), 'action': 'up', 'confidence': confidence, 'timestamp': time.time()})
        elif action == 1 and paddle_y > -240:  # 'paddle_down'
            left_paddle_down()
            game_data.append({'features': features.tolist(), 'action': 'down', 'confidence': confidence, 'timestamp': time.time()})
    else:
        # Fallback AI logic based on difficulty and randomness
        if ai_delay <= 0:
            difficulty = max(1, 6 - player_score // 2)
            ai_error = random.randint(-50, 50)
            if random.randint(0, difficulty) > 0:
                ai_target_y = ball.ycor() + ai_error  # Now ai_target_y is guaranteed to be defined
                
                # Adjust the decrement in ai_delay based on difficulty
                ai_delay -= 0.02 * (7 - difficulty)

                # Ensure ai_delay does not go below 0
                ai_delay = max(ai_delay, 0)

                # Record AI decision event for future training
                game_data.append({'event': 'ai_decision', 'ai_target_y': ai_target_y, 'ai_error': ai_error, 'difficulty': difficulty, 'timestamp': time.time()})
        else:
            ai_delay -= 0.02  # Normal decrement of ai_delay

        # Use ai_target_y after it's been defined
        if abs(paddle_y - ai_target_y) > 30:
            if paddle_y < ai_target_y:
                left_paddle_up()
                # Record AI movement event
                game_data.append({'event': 'ai_paddle_up', 'paddle_y': paddle_y + 20, 'ball_pos': (ball_x, ball_y), 'timestamp': time.time()})
            elif paddle_y > ai_target_y:
                left_paddle_down()
                # Record AI movement event
                game_data.append({'event': 'ai_paddle_down', 'paddle_y': paddle_y - 20, 'ball_pos': (ball_x, ball_y), 'timestamp': time.time()})

# Save game data to a file
def save_data():
    directory = 'data/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(directory + 'pong_game_data.json', 'w') as file:
        json.dump(game_data, file)

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

# Preprocess game data for training
def preprocess_data(game_data):
    features, labels = [], []
    for event in game_data:
        if 'ball_pos' in event and 'paddle_y' in event and 'label' in event:
            ball_x, ball_y = event['ball_pos']
            paddle_y = event['paddle_y']
            features.append([ball_x / 600, ball_y / 600, paddle_y / 300])
            labels.append(1 if event['label'] == 'paddle_up' else 0)
    return np.array(features), np.array(labels)

# Function to load or create and train the model
def get_or_create_model():
    global model
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        # Assuming you have a function to preprocess and split your data
        train_features, train_labels = preprocess_data(game_data)
        if len(train_features) > 0:
            model = create_model(input_shape=train_features.shape[1:])
            # Optional: Train the model immediately or defer training to a later stage
            # model.fit(train_features, train_labels, epochs=10)
            # model.save(model_path)
        else:
            print("Insufficient data for training.")
            model = None  # Ensure model is set to None if no model is created
    return model

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

def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=input_shape),  # Note the use of `shape` instead of `input_shape`
        # Dense layers
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        # Output layer
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def train_model_in_background(train_features, train_labels):
    global model
    if model is None:
        model = create_model(input_shape=train_features.shape[1:])
    train_and_save_model(model, train_features, train_labels, model_path)

def finalize_game():
    global model

    # Final game state, data saving, and preprocessing...
    game_data.append({'event': 'game_end', 'final_player_score': player_score, 'final_computer_score': computer_score, 'timestamp': time.time()})
    save_canvas(win, "final_game_board")
    save_data()

    train_data, val_data, test_data = split_data(np.array(game_data))
    save_split_data(train_data, 'train_data.json')
    save_split_data(val_data, 'val_data.json')
    save_split_data(test_data, 'test_data.json')

    train_features, train_labels = preprocess_data(train_data)

    # Move the model training to a background thread
    if len(train_features) > 0 and len(train_labels) > 0:
        training_thread = threading.Thread(target=train_model_in_background, args=(train_features, train_labels))
        training_thread.start()
    else:
        print("Insufficient data for training.")

    reset_game()

# Main game loop modified for graceful exit
while True:
    if not game_on:
        finalize_game()  # Call finalize_game when the game ends
        break  # If game_on is False, then break the loop and end the game

    win.update()

    # Move the ball
    ball.setx(ball.xcor() + ball.dx)
    ball.sety(ball.ycor() + ball.dy)

    # Border checking
    if ball.ycor() > 290 or ball.ycor() < -290:
        ball.dy *= -1
    
    if ball.xcor() > 290:  # Computer scores
        ball.goto(0, 0)
        ball.dx *= -1
        computer_score += 1  # Increment computer's score
        update_score(player_score, computer_score)

    if ball.xcor() < -290:  # Player scores
        ball.goto(0, 0)
        time.sleep(1)  # Pause for a moment before continuing
        ball.dx *= 1.05  # Slightly increase ball's speed
        ball.dy *= 1.05
        ball.dx = max(ball.dx, 1.5)  # Ensure the ball speed does not go below the starting speed
        ball.dy = max(ball.dy, 1.5)
        player_score += 1  # Increment player's score
        update_score(player_score, computer_score)
        ball.dx = -ball.dx  # Reset ball direction

    # Paddle and ball collisions
    if (ball.xcor() > 240 and ball.xcor() < 250) and (ball.ycor() < right_paddle.ycor() + 50 and ball.ycor() > right_paddle.ycor() - 50):
        ball.setx(240)
        ball.dx *= -1
    
    if (ball.xcor() < -240 and ball.xcor() > -250) and (ball.ycor() < left_paddle.ycor() + 50 and ball.ycor() > left_paddle.ycor() - 50):
        ball.setx(-240)
        ball.dx *= -1

    # Update AI paddle movement
    update_ai_paddle()

    # Periodically save data, e.g., every 100 events
    if len(game_data) % 100 == 0:
        save_data() 

    # Check for game end condition (e.g., player or computer reaches 10 points)
    if player_score >= POINTS_NEEDED or computer_score >= POINTS_NEEDED:
        finalize_game()

win.bye()