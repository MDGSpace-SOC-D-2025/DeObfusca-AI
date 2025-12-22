"""
Reinforcement Learning training harness for decompiler.

Uses PPO to optimize decompilation quality based on compilation + fuzzing rewards.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque
import requests

class PPOAgent:
    """
    Proximal Policy Optimization agent for decompilation.
    
    State: Sanitized P-Code features
    Action: Select decompilation strategy/parameters
    Reward: Compilation success (0.5) + Behavioral match (10.0)
    """
    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=lr)
        
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.K_epochs = 4
        
        self.memory = []
    
    def select_action(self, state):
        """Select action using current policy."""
        state = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.policy(state)
            dist = Categorical(action_probs)
            action = dist.sample()
        
        return action.item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def update(self):
        """Update policy using PPO."""
        if len(self.memory) < 32:
            return
        
        # Convert memory to tensors
        states = torch.FloatTensor([m[0] for m in self.memory])
        actions = torch.LongTensor([m[1] for m in self.memory])
        rewards = torch.FloatTensor([m[2] for m in self.memory])
        
        # Calculate returns
        returns = self._calculate_returns(rewards)
        
        # Old action probabilities
        old_action_probs = self.policy(states)
        old_dist = Categorical(old_action_probs)
        old_log_probs = old_dist.log_prob(actions)
        
        # PPO update for K epochs
        for _ in range(self.K_epochs):
            # Current action probabilities
            action_probs = self.policy(states)
            dist = Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            
            # Ratio
            ratios = torch.exp(log_probs - old_log_probs.detach())
            
            # Value predictions
            values = self.value(states).squeeze()
            advantages = returns - values.detach()
            
            # PPO loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = nn.MSELoss()(values, returns)
            
            # Update policy
            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()
            
            # Update value
            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()
        
        # Clear memory
        self.memory = []
    
    def _calculate_returns(self, rewards):
        """Calculate discounted returns."""
        returns = []
        R = 0
        
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns

class PolicyNetwork(nn.Module):
    """Policy network for PPO."""
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.fc(x)

class ValueNetwork(nn.Module):
    """Value network for PPO."""
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.fc(x)

def train_rl_agent(num_episodes=1000):
    """Train PPO agent on decompilation task."""
    
    STATE_DIM = 128  # Feature dimension
    ACTION_DIM = 4   # Number of decompilation strategies
    
    agent = PPOAgent(STATE_DIM, ACTION_DIM)
    
    for episode in range(num_episodes):
        # Get training sample
        state = get_training_sample()
        
        # Select action
        action = agent.select_action(state)
        
        # Execute decompilation with selected strategy
        source_code = execute_decompilation(state, action)
        
        # Verify and get reward
        reward = verify_decompilation(source_code)
        
        # Store transition
        agent.store_transition(state, action, reward, state, True)
        
        # Update agent
        agent.update()
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {reward:.2f}")
    
    # Save model
    torch.save(agent.policy.state_dict(), '/app/models/ppo_policy.pth')
    print("Training complete")

def get_training_sample():
    """Get a training sample from preprocessed dataset."""
    # In production, load from actual dataset of obfuscated binaries
    # For now, simulate realistic P-Code features
    state_features = {
        'num_instructions': np.random.randint(10, 200),
        'num_branches': np.random.randint(2, 20),
        'num_loops': np.random.randint(0, 5),
        'stack_operations': np.random.randint(5, 50),
        'arithmetic_ops': np.random.randint(10, 100)
    }
    
    # Convert to feature vector
    feature_vector = np.array([
        state_features['num_instructions'] / 200.0,
        state_features['num_branches'] / 20.0,
        state_features['num_loops'] / 5.0,
        state_features['stack_operations'] / 50.0,
        state_features['arithmetic_ops'] / 100.0
    ])
    
    # Pad to expected dimension
    full_vector = np.zeros(128)
    full_vector[:len(feature_vector)] = feature_vector
    full_vector[len(feature_vector):] = np.random.randn(128 - len(feature_vector)) * 0.1
    
    return full_vector

def execute_decompilation(state, action):
    """Execute decompilation with selected strategy based on action."""
    # Action space: 0=conservative, 1=aggressive, 2=balanced, 3=type-focused
    strategies = {
        0: "// Conservative decompilation\nint function(int x) {\n    return x + 1;\n}",
        1: "// Aggressive decompilation\nint function(int x) {\n    int result = x;\n    result++;\n    return result;\n}",
        2: "// Balanced decompilation\nint function(int x) {\n    return (x + 1);\n}",
        3: "// Type-focused decompilation\nint32_t function(int32_t x) {\n    return (int32_t)(x + 1);\n}"
    }
    
    return strategies.get(action, strategies[0])

def verify_decompilation(source_code):
    """Verify decompilation and return reward."""
    try:
        response = requests.post(
            'http://rl-service:5004/verify',
            json={'source_code': source_code}
        )
        return response.json()['reward']
    except:
        return -1.0

if __name__ == '__main__':
    train_rl_agent()
