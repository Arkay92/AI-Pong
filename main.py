import turtle
import random  # Import the random module
import time

# Additional variables for AI control
ai_delay = 0
ai_target_y = 0

# Set up the window
win = turtle.Screen()
win.title("Python Pong")
win.bgcolor("black")
win.setup(width=600, height=600)

# Score variables
player_score = 0
computer_score = 0

# Score display
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
ball.dx = 2.5  # Increased initial ball speed
ball.dy = -2.5

# Functions to move the paddles
def left_paddle_up():
    y = left_paddle.ycor()
    if y < 250:
        left_paddle.sety(y + 20)

def left_paddle_down():
    y = left_paddle.ycor()
    if y > -240:
        left_paddle.sety(y - 20)

def right_paddle_up():
    y = right_paddle.ycor()
    if y < 250:
        right_paddle.sety(y + 20)

def right_paddle_down():
    y = right_paddle.ycor()
    if y > -240:
        right_paddle.sety(y - 20)

# Keyboard bindings
win.listen()
win.onkeypress(left_paddle_up, "w")
win.onkeypress(left_paddle_down, "s")
win.onkeypress(right_paddle_up, "Up")
win.onkeypress(right_paddle_down, "Down")

# Function to update score
def update_score(player, computer):
    score_display.clear()
    score_display.write("Computer: {}  Player: {}".format(computer, player), align="center", font=("Courier", 24, "normal"))

def update_ai_paddle():
    global ai_delay, ai_target_y
    min_movement_threshold = 30  # Minimum distance from target before AI paddle starts moving

    if ai_delay <= 0:  # If AI delay has elapsed, decide on a new movement
        difficulty = max(1, 6 - player_score // 2)  # AI difficulty
        ai_error = random.randint(-50, 50)  # Extended range for error in AI's target position

        # Randomly decide whether AI will 'think' before moving again
        if random.randint(0, difficulty) > 0:
            ai_target_y = ball.ycor() + ai_error  # New target position for AI
            ai_delay = random.uniform(0.2, 0.8)  # Adjusted delay for next move, simulating reaction time

    else:  # AI is in 'thinking' mode, decrement the delay
        ai_delay -= 0.02  # Decrease delay, adjust for your game's update rate

    # Smoothly move AI paddle towards target position with a minimum threshold
    if abs(left_paddle.ycor() - ai_target_y) > min_movement_threshold:
        if left_paddle.ycor() < ai_target_y:
            left_paddle_up()
        elif left_paddle.ycor() > ai_target_y:
            left_paddle_down()

# Main game loop
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
