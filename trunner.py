# tf_runner.py

import pygame
import math
import random
import numpy as np
import tensorflow as tf
from config import *
from utils import (
    create_platforms, refill_platforms,
    draw_text, draw_hud, smooth_scroll, draw_network
)

# --- Hyperparameters ---
EPOCHS            = 500      # number of training episodes
GAMMA             = 0.99     # discount factor
LR                = 1e-3     # learning rate for the policy network
GRACE_FRAMES      = 10
STALL_FRAMES      = 3 * 60
JUMP_PENALTY_FRM  = 30
JUMP_PENALTY      = 1.0
ASCENSION_REWARD  = 5.0
CLOSE_INCENTIVE   = 0.07

INPUT_SIZE        = 4 + 5*2   # x,y,vy,can_jump + 5×(dir,dist)
HIDDEN_UNITS      = [64, 64]
OUTPUT_SIZE       = 3        # left, right, jump

SHOW_NETWORK      = True

# Game constants
GRACE_FRAMES       = 10
MAX_FALL_SPEED     = 10       # ← add this
STALL_FRAMES       = 3 * 60
CLOSE_INCENTIVE    = 0.07
JUMP_PENALTY_FRAMES= 30
JUMP_PENALTY       = 1.0
ASCENSION_REWARD   = 5.0

# --- Build the policy network ---
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(INPUT_SIZE,)),
        tf.keras.layers.Dense(HIDDEN_UNITS[0], activation='relu'),
        tf.keras.layers.Dense(HIDDEN_UNITS[1], activation='relu'),
        tf.keras.layers.Dense(OUTPUT_SIZE, activation='softmax'),
    ])
    return model

# --- Discount and normalize rewards ---
def discount_and_normalize(rewards):
    discounted = []
    r = 0
    for reward in reversed(rewards):
        r = reward + GAMMA * r
        discounted.append(r)
    discounted = discounted[::-1]
    discounted = np.array(discounted)
    return (discounted - discounted.mean()) / (discounted.std() + 1e-8)

# --- Run one episode and collect trajectories ---
def run_episode(env, model):
    state = env.reset()
    states, actions, rewards = [], [], []
    done = False
    frame = 0
    last_jump = 0
    while not done:
        # get action probabilities
        probs = model(tf.expand_dims(state,0))[0].numpy()
        action = np.random.choice(3, p=probs)
        nxt_state, reward, done, info = env.step(action, frame, last_jump)
        # record
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        # update for next
        if action == 2 and state[3]==1:  # jump occurred
            if frame - last_jump < JUMP_PENALTY_FRM:
                rewards[-1] -= JUMP_PENALTY
            last_jump = frame
        state = nxt_state
        frame += 1
    # compute discounted returns
    returns = discount_and_normalize(rewards)
    return states, actions, returns, sum(rewards)

# --- Define environment wrapper ---
class JumperEnv:
    def reset(self):
        self.platforms = create_platforms()
        spawn = max(self.platforms, key=lambda p: p.y)
        self.p = pygame.Rect(spawn.x, spawn.y-30, 30, 30)
        self.vy = 0
        self.can_jump = False
        self.scroll = 0
        self.min_y = self.p.y
        self.frame = 0
        return self._get_obs()
    def _get_obs(self):
        close = sorted(self.platforms, key=lambda pl: abs(pl.y - self.p.y))[:5]
        obs = [self.p.x/WIDTH, self.p.y/HEIGHT, self.vy/MAX_FALL_SPEED, float(self.can_jump)]
        for pl in close:
            flag = 1 if pl.x+pl.width < self.p.x else -1 if pl.x>self.p.x else 0
            d = math.hypot(pl.x-self.p.x, pl.y-self.p.y)
            obs += [flag, d/math.hypot(WIDTH,HEIGHT)]
        obs += [0,1.0]*(5-len(close))
        return np.array(obs, dtype=np.float32)
    def step(self, action, frame, last_jump):
        # action: 0 left,1 right,2 jump
        if action==0: self.p.x -= 5
        if action==1: self.p.x += 5
        if action==2 and self.can_jump:
            self.vy = -16
            self.can_jump = False
        # physics
        self.p.x = max(0, min(WIDTH-self.p.width, self.p.x))
        self.vy = min(self.vy+0.5, MAX_FALL_SPEED)
        self.p.y += self.vy
        reward = (self.scroll * 0.001) - (self.vy * 0.01)
        done = False
        # off-screen or stall?
        if frame > GRACE_FRAMES and self.p.y - self.scroll > HEIGHT:
            reward -= 20; done=True
        # landing
        for pl in self.platforms:
            if self.p.colliderect(pl) and self.vy>0:
                self.p.bottom, self.vy, self.can_jump = pl.top, 0, True
                if pl.y < self.min_y:
                    reward += ASCENSION_REWARD + (self.min_y-pl.y)/HEIGHT*20
                    self.min_y = pl.y
                break
        # world update
        self.scroll = smooth_scroll([self.p], self.platforms, [self.min_y], [frame], self.scroll, self.min_y)
        refill_platforms(self.platforms)
        self.frame += 1
        return self._get_obs(), reward, done, {}

# --- Training loop ---
def evolve():
    env = JumperEnv()
    model = build_model()
    optimizer = tf.keras.optimizers.Adam(LR)

    for ep in range(EPOCHS):
        # collect episode
        states, actions, returns, total_reward = run_episode(env, model)
        # perform one policy-gradient update
        with tf.GradientTape() as tape:
            logits = model(tf.stack(states))
            action_masks = tf.one_hot(actions, 3)
            log_probs = tf.reduce_sum(action_masks * tf.math.log(logits + 1e-8), axis=1)
            loss = -tf.reduce_mean(log_probs * returns)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        print(f"Episode {ep}  reward={total_reward:.2f}")

        # visualize best every 10 episodes
        if ep % 10 == 0:
            visualize(env, model, ep)

# --- Visualization with Pygame ---
def visualize(env, model, generation):
    obs = env.reset()
    done = False
    frame = 0
    while not done:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); return

        probs = model(tf.expand_dims(obs,0))[0].numpy()
        action = np.argmax(probs)
        obs, _, done, _ = env.step(action, frame, frame)
        frame += 1

        screen.fill(BLACK)
        for plat in env.platforms:
            pygame.draw.rect(screen, GREEN, plat)
        pygame.draw.rect(screen, RED, env.p)
        if SHOW_NETWORK:
            draw_network(screen, model, position=(10,10))
        draw_hud(env.scroll, generation, generation//25+1, 1)
        pygame.display.flip()
        clock.tick(60)

if __name__=='__main__':
    pygame.init()
    evolve()
