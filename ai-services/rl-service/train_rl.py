

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
from collections import deque
import json
import os


@dataclass
class Experience:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool
    log_prob: float
    value: float


@dataclass
class RolloutBuffer:
    states: List[torch.Tensor] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    
    def add(self, exp: Experience):
        self.states.append(exp.state)
        self.actions.append(exp.action)
        self.rewards.append(exp.reward)
        self.log_probs.append(exp.log_prob)
        self.values.append(exp.value)
        self.dones.append(exp.done)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
    
    def __len__(self):
        return len(self.states)


class CodeStateEncoder(nn.Module):
    
    def __init__(self, vocab_size: int = 1024, embed_dim: int = 256,
                 num_heads: int = 4, num_layers: int = 3, max_seq_len: int = 512):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = tokens.shape
        
        # Clamp sequence length
        if seq_len > self.max_seq_len:
            tokens = tokens[:, :self.max_seq_len]
            if mask is not None:
                mask = mask[:, :self.max_seq_len]
            seq_len = self.max_seq_len
        
        # Embeddings
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        
        # Transformer
        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=~mask.bool())
        else:
            x = self.transformer(x)
        
        # Mean pooling
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)
        
        # Output projection
        output = self.layer_norm(self.output_proj(pooled))
        
        return output


class PolicyNetwork(nn.Module):
    
    ACTIONS = [
        'keep_current',          # 0: Keep current decompilation
        'add_type_cast',         # 1: Add type cast
        'remove_redundant',      # 2: Remove redundant code
        'fix_loop_bounds',       # 3: Fix loop bounds
        'add_null_check',        # 4: Add NULL check
        'fix_operator',          # 5: Fix operator
        'add_initialization',    # 6: Add variable initialization
        'fix_array_access',      # 7: Fix array access
        'simplify_expression',   # 8: Simplify expression
        'add_return',            # 9: Add return statement
        'fix_condition',         # 10: Fix conditional logic
        'regenerate',            # 11: Request full regeneration
    ]
    
    def __init__(self, state_dim: int = 256, hidden_dim: int = 512, num_actions: int = 12):
        super().__init__()
        
        self.num_actions = num_actions
        
        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Forward pass returning action logits and state value
        shared_features = self.shared(state)
        
        action_logits = self.policy_head(shared_features)
        value = self.value_head(shared_features)
        
        return action_logits, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, float, float]:
        # Sample action from policy
        action_logits, value = self.forward(state)
        
        # Create distribution
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        
        if deterministic:
            action = action_logits.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Evaluate actions for PPO update
        action_logits, values = self.forward(states)
        
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(-1), entropy


class PPOTrainer:
    
    def __init__(
        self,
        state_encoder: CodeStateEncoder,
        policy: PolicyNetwork,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        value_clip_range: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.01,
        device: str = 'cuda'
    ):
        self.state_encoder = state_encoder.to(device)
        self.policy = policy.to(device)
        self.device = device
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_clip_range = value_clip_range
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        
        # Optimizer
        self.optimizer = optim.AdamW(
            list(self.state_encoder.parameters()) + list(self.policy.parameters()),
            lr=lr,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=100, T_mult=2
        )
        
        # Experience buffer
        self.buffer = RolloutBuffer()
        
        # Training stats
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'kl_divergence': [],
            'total_reward': []
        }
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                    dones: List[bool], last_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = []
        gae = 0
        
        # Append last value for bootstrapping
        values = values + [last_value]
        
        # Compute GAE backwards
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = advantages + torch.tensor(values[:-1], dtype=torch.float32, device=self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self, epochs: int = 10, batch_size: int = 64) -> Dict[str, float]:
        if len(self.buffer) < batch_size:
            return {}
        
        # Compute GAE
        advantages, returns = self.compute_gae(
            self.buffer.rewards,
            self.buffer.values,
            self.buffer.dones
        )
        
        # Convert buffer to tensors
        states = torch.stack(self.buffer.states).to(self.device)
        actions = torch.tensor(self.buffer.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device)
        old_values = torch.tensor(self.buffer.values, dtype=torch.float32, device=self.device)
        
        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        num_updates = 0
        
        # Multiple epochs of updates
        for epoch in range(epochs):
            # Generate random indices
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                # Get batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_values = old_values[batch_indices]
                
                # Forward pass through encoder
                state_embeddings = self.state_encoder(batch_states)
                
                # Evaluate current policy
                new_log_probs, new_values, entropy = self.policy.evaluate_actions(
                    state_embeddings, batch_actions
                )
                
                # Compute ratio for PPO
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss with clipping
                value_clipped = batch_old_values + torch.clamp(
                    new_values - batch_old_values,
                    -self.value_clip_range,
                    self.value_clip_range
                )
                value_loss1 = F.mse_loss(new_values, batch_returns)
                value_loss2 = F.mse_loss(value_clipped, batch_returns)
                value_loss = torch.max(value_loss1, value_loss2)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    list(self.state_encoder.parameters()) + list(self.policy.parameters()),
                    self.max_grad_norm
                )
                
                self.optimizer.step()
                
                # Compute KL divergence for early stopping
                with torch.no_grad():
                    kl = (batch_old_log_probs - new_log_probs).mean().item()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_kl += kl
                num_updates += 1
                
                # Early stopping based on KL divergence
                if abs(kl) > 1.5 * self.target_kl:
                    break
            
            # Check KL divergence for epoch early stopping
            if abs(total_kl / max(num_updates, 1)) > 1.5 * self.target_kl:
                break
        
        # Update scheduler
        self.scheduler.step()
        
        # Clear buffer
        self.buffer.clear()
        
        # Compute average metrics
        metrics = {
            'policy_loss': total_policy_loss / max(num_updates, 1),
            'value_loss': total_value_loss / max(num_updates, 1),
            'entropy': total_entropy / max(num_updates, 1),
            'kl_divergence': total_kl / max(num_updates, 1),
            'num_updates': num_updates
        }
        
        # Store stats
        for key, value in metrics.items():
            if key in self.training_stats:
                self.training_stats[key].append(value)
        
        return metrics
    
    def collect_experience(self, state: torch.Tensor, action: int, reward: float,
                           next_state: torch.Tensor, done: bool, 
                           log_prob: float, value: float):
        exp = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=log_prob,
            value=value
        )
        self.buffer.add(exp)
    
    def save_checkpoint(self, path: str, epoch: int):
        checkpoint = {
            'epoch': epoch,
            'state_encoder': self.state_encoder.state_dict(),
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'training_stats': self.training_stats
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str) -> int:
        checkpoint = torch.load(path, map_location=self.device)
        
        self.state_encoder.load_state_dict(checkpoint['state_encoder'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        
        print(f"Loaded checkpoint from {path}, epoch {checkpoint['epoch']}")
        return checkpoint['epoch']


class RewardShaper:
    
    def __init__(self):
        # Reward weights
        self.weights = {
            'compilation': 2.0,
            'syntax': 1.0,
            'execution_match': 5.0,
            'symbolic_equiv': 3.0,
            'code_quality': 1.0,
            'improvement': 2.0
        }
        
        # History for improvement tracking
        self.history = deque(maxlen=100)
    
    def compute_reward(
        self,
        compilation_success: bool,
        syntax_valid: bool,
        execution_match: bool,
        symbolic_equiv: bool,
        code_quality_score: float,
        previous_score: Optional[float] = None
    ) -> float:
        reward = 0.0
        
        # Base rewards
        if compilation_success:
            reward += self.weights['compilation']
        else:
            reward -= 1.0  # Penalty for compilation failure
        
        if syntax_valid:
            reward += self.weights['syntax']
        
        if execution_match:
            reward += self.weights['execution_match']
        
        if symbolic_equiv:
            reward += self.weights['symbolic_equiv']
        
        # Code quality bonus
        reward += self.weights['code_quality'] * code_quality_score
        
        # Improvement bonus
        current_score = self._compute_score(
            compilation_success, execution_match, symbolic_equiv, code_quality_score
        )
        
        if previous_score is not None:
            improvement = current_score - previous_score
            reward += self.weights['improvement'] * improvement
        
        # Store in history
        self.history.append(current_score)
        
        return reward
    
    def _compute_score(
        self,
        compilation_success: bool,
        execution_match: bool,
        symbolic_equiv: bool,
        code_quality: float
    ) -> float:
        score = 0.0
        
        if compilation_success:
            score += 0.2
        if execution_match:
            score += 0.4
        if symbolic_equiv:
            score += 0.3
        
        score += 0.1 * code_quality
        
        return score
    
    def get_potential(self, state_features: Dict) -> float:
        # Features that indicate progress
        num_valid_lines = state_features.get('valid_lines', 0)
        num_total_lines = state_features.get('total_lines', 1)
        type_coverage = state_features.get('type_coverage', 0.0)
        
        # Potential function
        potential = (
            0.5 * (num_valid_lines / max(num_total_lines, 1)) +
            0.3 * type_coverage +
            0.2 * state_features.get('structure_score', 0.0)
        )
        
        return potential


class DecompilationEnvironment:
    
    def __init__(self, verifier, max_steps: int = 10):
        self.verifier = verifier
        self.max_steps = max_steps
        self.current_step = 0
        
        # Current state
        self.pcode_features = None
        self.current_code = None
        self.previous_score = None
        
        # Reward shaper
        self.reward_shaper = RewardShaper()
        
        # Action implementations
        self.action_handlers = {
            0: self._action_keep,
            1: self._action_add_type_cast,
            2: self._action_remove_redundant,
            3: self._action_fix_loop_bounds,
            4: self._action_add_null_check,
            5: self._action_fix_operator,
            6: self._action_add_initialization,
            7: self._action_fix_array_access,
            8: self._action_simplify_expression,
            9: self._action_add_return,
            10: self._action_fix_condition,
            11: self._action_regenerate,
        }
    
    def reset(self, pcode_features: List[Dict], initial_code: str) -> torch.Tensor:
        self.pcode_features = pcode_features
        self.current_code = initial_code
        self.current_step = 0
        self.previous_score = None
        
        return self._get_state_tensor()
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        self.current_step += 1
        
        # Apply action
        handler = self.action_handlers.get(action, self._action_keep)
        self.current_code = handler(self.current_code)
        
        # Verify current code
        verification = self._verify_code(self.current_code)
        
        # Compute reward
        reward = self.reward_shaper.compute_reward(
            compilation_success=verification['compilation_success'],
            syntax_valid=verification.get('syntax_valid', True),
            execution_match=verification.get('execution_match', False),
            symbolic_equiv=verification.get('symbolic_equivalent', False),
            code_quality_score=verification.get('code_quality', 0.5),
            previous_score=self.previous_score
        )
        
        # Update previous score
        self.previous_score = self.reward_shaper._compute_score(
            verification['compilation_success'],
            verification.get('execution_match', False),
            verification.get('symbolic_equivalent', False),
            verification.get('code_quality', 0.5)
        )
        
        # Check termination
        done = (
            self.current_step >= self.max_steps or
            (verification['compilation_success'] and 
             verification.get('execution_match', False))
        )
        
        # Get next state
        next_state = self._get_state_tensor()
        
        info = {
            'verification': verification,
            'step': self.current_step,
            'action': PolicyNetwork.ACTIONS[action]
        }
        
        return next_state, reward, done, info
    
    def _get_state_tensor(self) -> torch.Tensor:
        # Simple tokenization (in production, use proper tokenizer)
        tokens = []
        
        # Encode P-Code features
        for feat in self.pcode_features[:256]:
            if isinstance(feat, dict):
                mnemonic = feat.get('mnemonic', 'NOP')
                tokens.append(hash(mnemonic) % 1024)
            else:
                tokens.append(hash(str(feat)) % 1024)
        
        # Encode current code
        for char in self.current_code[:256]:
            tokens.append(ord(char) % 1024)
        
        # Pad to fixed length
        while len(tokens) < 512:
            tokens.append(0)
        
        return torch.tensor(tokens[:512], dtype=torch.long)
    
    def _verify_code(self, code: str) -> Dict:
        try:
            result = self.verifier.verify(code)
            return result
        except Exception as e:
            return {
                'compilation_success': False,
                'error': str(e)
            }
    
    # Action implementations
    def _action_keep(self, code: str) -> str:
        return code
    
    def _action_add_type_cast(self, code: str) -> str:
        import re
        # Find assignments without casts
        code = re.sub(
            r'(\w+)\s*=\s*(\w+)\s*;',
            r'\1 = (int)\2;',
            code
        )
        return code
    
    def _action_remove_redundant(self, code: str) -> str:
        lines = code.split('\n')
        seen = set()
        filtered = []
        for line in lines:
            stripped = line.strip()
            if stripped and stripped not in seen:
                filtered.append(line)
                seen.add(stripped)
        return '\n'.join(filtered)
    
    def _action_fix_loop_bounds(self, code: str) -> str:
        import re
        code = re.sub(
            r'for\s*\(\s*int\s+(\w+)\s*=\s*0\s*;\s*\1\s*<=\s*(\w+)\s*;',
            r'for (int \1 = 0; \1 < \2;',
            code
        )
        return code
    
    def _action_add_null_check(self, code: str) -> str:
        import re
        code = re.sub(
            r'(\*(\w+))',
            r'((\2 != NULL) ? *\2 : 0)',
            code
        )
        return code
    
    def _action_fix_operator(self, code: str) -> str:
        # Common fix: = vs ==
        import re
        code = re.sub(
            r'if\s*\(\s*(\w+)\s*=\s*(\w+)\s*\)',
            r'if (\1 == \2)',
            code
        )
        return code
    
    def _action_add_initialization(self, code: str) -> str:
        import re
        code = re.sub(
            r'(int\s+\w+)\s*;',
            r'\1 = 0;',
            code
        )
        return code
    
    def _action_fix_array_access(self, code: str) -> str:
        import re
        code = re.sub(
            r'(\w+)\[(\w+)\]',
            r'((\2 < sizeof(\1)/sizeof(\1[0])) ? \1[\2] : 0)',
            code
        )
        return code
    
    def _action_simplify_expression(self, code: str) -> str:
        import re
        # Simplify x + 0 -> x
        code = re.sub(r'(\w+)\s*\+\s*0', r'\1', code)
        # Simplify x * 1 -> x
        code = re.sub(r'(\w+)\s*\*\s*1', r'\1', code)
        return code
    
    def _action_add_return(self, code: str) -> str:
        if 'return' not in code:
            lines = code.split('\n')
            # Find last closing brace and add return before it
            for i in range(len(lines) - 1, -1, -1):
                if '}' in lines[i]:
                    lines.insert(i, '    return 0;')
                    break
            code = '\n'.join(lines)
        return code
    
    def _action_fix_condition(self, code: str) -> str:
        import re
        # Fix common issue: missing parentheses
        code = re.sub(
            r'if\s+(\w+)\s*([<>=!]+)\s*(\w+)',
            r'if (\1 \2 \3)',
            code
        )
        return code
    
    def _action_regenerate(self, code: str) -> str:
        # In production, this would call the LLM
        return code


def train_rl_agent(
    train_data: List[Dict],
    num_epochs: int = 100,
    episodes_per_epoch: int = 10,
    checkpoint_dir: str = './checkpoints',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    # Initialize models
    state_encoder = CodeStateEncoder(
        vocab_size=1024,
        embed_dim=256,
        num_heads=4,
        num_layers=3
    )
    
    policy = PolicyNetwork(
        state_dim=256,
        hidden_dim=512,
        num_actions=len(PolicyNetwork.ACTIONS)
    )
    
    # Initialize trainer
    trainer = PPOTrainer(
        state_encoder=state_encoder,
        policy=policy,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coef=0.01,
        device=device
    )
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize environment (mock verifier for training)
    class MockVerifier:
        def verify(self, code):
            # Simple verification
            compiles = '{' in code and '}' in code
            return {
                'compilation_success': compiles,
                'execution_match': compiles and 'return' in code,
                'symbolic_equivalent': False,
                'code_quality': 0.5 if compiles else 0.0
            }
    
    env = DecompilationEnvironment(MockVerifier(), max_steps=10)
    
    # Training loop
    print(f"Starting RL training on {device}")
    print(f"Training data: {len(train_data)} examples")
    
    for epoch in range(num_epochs):
        epoch_rewards = []
        epoch_lengths = []
        
        for episode in range(episodes_per_epoch):
            # Sample training example
            example = train_data[episode % len(train_data)]
            pcode_features = example.get('features', [])
            initial_code = example.get('initial_code', 'void f() {}')
            
            # Reset environment
            state = env.reset(pcode_features, initial_code)
            episode_reward = 0
            
            done = False
            while not done:
                # Get state embedding
                with torch.no_grad():
                    state_tensor = state.unsqueeze(0).to(device)
                    state_emb = trainer.state_encoder(state_tensor)
                    action, log_prob, value = trainer.policy.get_action(state_emb)
                
                # Take action
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                trainer.collect_experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    log_prob=log_prob,
                    value=value
                )
                
                episode_reward += reward
                state = next_state
            
            epoch_rewards.append(episode_reward)
            epoch_lengths.append(env.current_step)
        
        # PPO update
        if len(trainer.buffer) >= 64:
            metrics = trainer.update(epochs=10, batch_size=64)
        else:
            metrics = {}
        
        # Logging
        avg_reward = np.mean(epoch_rewards)
        avg_length = np.mean(epoch_lengths)
        trainer.training_stats['total_reward'].append(avg_reward)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: avg_reward={avg_reward:.2f}, avg_length={avg_length:.1f}")
            if metrics:
                print(f"  policy_loss={metrics.get('policy_loss', 0):.4f}, "
                      f"value_loss={metrics.get('value_loss', 0):.4f}, "
                      f"entropy={metrics.get('entropy', 0):.4f}")
        
        # Save checkpoint
        if epoch % 50 == 0 and epoch > 0:
            trainer.save_checkpoint(
                os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'),
                epoch
            )
    
    # Final checkpoint
    trainer.save_checkpoint(
        os.path.join(checkpoint_dir, 'final_checkpoint.pt'),
        num_epochs
    )
    
    print("Training complete!")
    return trainer


if __name__ == '__main__':
    # Example training data
    train_data = [
        {
            'features': [{'mnemonic': 'INT_ADD'}, {'mnemonic': 'STORE'}],
            'initial_code': 'void f() { int x; }'
        },
        {
            'features': [{'mnemonic': 'CBRANCH'}, {'mnemonic': 'INT_LESS'}],
            'initial_code': 'void f() { if (x > 0) {} }'
        }
    ]
    
    trainer = train_rl_agent(
        train_data=train_data,
        num_epochs=100,
        episodes_per_epoch=10,
        checkpoint_dir='./rl_checkpoints'
    )
