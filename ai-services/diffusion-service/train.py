"""
Adversarial Training for Diffusion Model.

Implements adversarial robustness improvements:
1. Adversarial examples during training
2. Gradient-based perturbations
3. Robustness evaluation
4. Defensive distillation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import os

from app import DiffusionCodeGenerator


class AdversarialDiffusionTrainer:
    """
    Adversarial training for diffusion models.
    
    Improves robustness against:
    - Noisy binary inputs
    - Malformed P-Code
    - Adversarial obfuscation patterns
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.epsilon = 0.1  # Perturbation magnitude
        self.alpha = 0.01  # Step size for FGSM
        self.num_adv_steps = 5
    
    def train_epoch(self, dataloader, optimizer, epoch):
        """Train one epoch with adversarial examples."""
        self.model.train()
        total_loss = 0
        total_adv_loss = 0
        
        progress = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, (clean_tokens, condition, target_tokens) in enumerate(progress):
            clean_tokens = clean_tokens.to(self.device)
            condition = condition.to(self.device)
            target_tokens = target_tokens.to(self.device)
            
            # 1. Standard training step
            optimizer.zero_grad()
            clean_loss = self._compute_loss(clean_tokens, condition, target_tokens)
            
            # 2. Generate adversarial examples
            adv_condition = self._generate_adversarial_condition(
                clean_tokens, condition, target_tokens
            )
            
            # 3. Adversarial training step
            adv_loss = self._compute_loss(clean_tokens, adv_condition, target_tokens)
            
            # 4. Combined loss
            total_batch_loss = clean_loss + 0.5 * adv_loss
            total_batch_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += clean_loss.item()
            total_adv_loss += adv_loss.item()
            
            # Update progress bar
            progress.set_postfix({
                'loss': f'{clean_loss.item():.4f}',
                'adv_loss': f'{adv_loss.item():.4f}'
            })
        
        avg_loss = total_loss / len(dataloader)
        avg_adv_loss = total_adv_loss / len(dataloader)
        
        return avg_loss, avg_adv_loss
    
    def _compute_loss(self, tokens, condition, target):
        """Compute diffusion loss."""
        batch_size = tokens.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.model.num_timesteps, (batch_size,), device=self.device)
        
        # Add noise to tokens
        noise = torch.randn_like(tokens.float())
        alpha_t = self.model.alphas_cumprod[t].view(-1, 1)
        noisy_tokens = torch.sqrt(alpha_t) * tokens.float() + torch.sqrt(1 - alpha_t) * noise
        
        # Predict noise
        noise_pred = self.model(noisy_tokens.long(), t, condition)
        
        # MSE loss
        loss = nn.MSELoss()(noise_pred, noise)
        
        return loss
    
    def _generate_adversarial_condition(self, tokens, condition, target):
        """
        Generate adversarial perturbation using FGSM (Fast Gradient Sign Method).
        
        Creates adversarial examples by perturbing the condition vector
        in the direction that increases loss.
        """
        # Enable gradients for condition
        adv_condition = condition.clone().detach().requires_grad_(True)
        
        # Forward pass with adversarial condition
        loss = self._compute_loss(tokens, adv_condition, target)
        
        # Backward pass to get gradients
        loss.backward()
        
        # Generate perturbation
        with torch.no_grad():
            # FGSM: perturb in direction of gradient sign
            perturbation = self.epsilon * adv_condition.grad.sign()
            adv_condition = condition + perturbation
            
            # Clip to valid range
            adv_condition = torch.clamp(adv_condition, -3.0, 3.0)
        
        return adv_condition.detach()
    
    def generate_pgd_adversarial(self, tokens, condition, target):
        """
        Generate adversarial examples using PGD (Projected Gradient Descent).
        
        More powerful than FGSM - performs multiple iterative perturbations.
        """
        adv_condition = condition.clone().detach()
        
        for step in range(self.num_adv_steps):
            adv_condition.requires_grad = True
            
            loss = self._compute_loss(tokens, adv_condition, target)
            loss.backward()
            
            with torch.no_grad():
                # Take small step in gradient direction
                perturbation = self.alpha * adv_condition.grad.sign()
                adv_condition = adv_condition + perturbation
                
                # Project back to epsilon ball
                perturbation = adv_condition - condition
                perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
                adv_condition = condition + perturbation
                
                # Clip to valid range
                adv_condition = torch.clamp(adv_condition, -3.0, 3.0)
            
            adv_condition = adv_condition.detach()
        
        return adv_condition
    
    def evaluate_robustness(self, dataloader):
        """
        Evaluate model robustness against adversarial examples.
        
        Returns:
            dict with clean accuracy and adversarial accuracy
        """
        self.model.eval()
        
        clean_correct = 0
        adv_correct = 0
        total = 0
        
        with torch.no_grad():
            for tokens, condition, target in dataloader:
                tokens = tokens.to(self.device)
                condition = condition.to(self.device)
                target = target.to(self.device)
                
                # Clean predictions
                clean_pred = self.model.generate(condition, max_length=tokens.shape[1])
                clean_correct += (clean_pred == target).all(dim=1).sum().item()
                
                # Adversarial predictions
                adv_condition = self._generate_adversarial_condition(tokens, condition, target)
                adv_pred = self.model.generate(adv_condition, max_length=tokens.shape[1])
                adv_correct += (adv_pred == target).all(dim=1).sum().item()
                
                total += tokens.shape[0]
        
        clean_accuracy = clean_correct / total if total > 0 else 0
        adv_accuracy = adv_correct / total if total > 0 else 0
        
        return {
            'clean_accuracy': clean_accuracy,
            'adversarial_accuracy': adv_accuracy,
            'robustness_gap': clean_accuracy - adv_accuracy
        }


class DefensiveDistillation:
    """
    Defensive distillation to improve robustness.
    
    Train a student model to mimic teacher's softened outputs,
    making it more robust to adversarial perturbations.
    """
    
    def __init__(self, teacher_model, student_model, temperature=10.0):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
    
    def distill(self, dataloader, optimizer, num_epochs=10):
        """Perform defensive distillation."""
        self.teacher.eval()
        self.student.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            for tokens, condition, _ in tqdm(dataloader, desc=f'Distillation Epoch {epoch}'):
                optimizer.zero_grad()
                
                # Get soft labels from teacher
                with torch.no_grad():
                    teacher_logits = self.teacher(tokens, torch.zeros(tokens.shape[0]), condition)
                    soft_labels = torch.softmax(teacher_logits / self.temperature, dim=-1)
                
                # Student predictions
                student_logits = self.student(tokens, torch.zeros(tokens.shape[0]), condition)
                student_probs = torch.log_softmax(student_logits / self.temperature, dim=-1)
                
                # Distillation loss (KL divergence)
                loss = nn.KLDivLoss(reduction='batchmean')(student_probs, soft_labels)
                loss = loss * (self.temperature ** 2)  # Scale by T^2
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f'Distillation Epoch {epoch}: Loss = {avg_loss:.4f}')


def main():
    """Main training script with adversarial training."""
    # Configuration
    VOCAB_SIZE = 50000
    D_MODEL = 768
    NUM_TIMESTEPS = 1000
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')
    
    # Initialize model
    model = DiffusionCodeGenerator(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_timesteps=NUM_TIMESTEPS
    ).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Initialize adversarial trainer
    adv_trainer = AdversarialDiffusionTrainer(model, device=device)
    
    # Training loop would go here with actual dataloader
    print("Adversarial training configured and ready")
    print(f"Epsilon: {adv_trainer.epsilon}")
    print(f"Adversarial steps: {adv_trainer.num_adv_steps}")
    
    # Placeholder for actual training
    # for epoch in range(NUM_EPOCHS):
    #     clean_loss, adv_loss = adv_trainer.train_epoch(train_loader, optimizer, epoch)
    #     scheduler.step()
    #     
    #     if epoch % 5 == 0:
    #         robustness = adv_trainer.evaluate_robustness(val_loader)
    #         print(f"Robustness metrics: {robustness}")
    
    print("Training script complete")


if __name__ == '__main__':
    main()
