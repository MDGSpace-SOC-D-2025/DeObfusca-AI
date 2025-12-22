#!/usr/bin/env python3
"""
Data Preprocessing Pipeline for DeObfusca-AI

This script converts raw binaries and source code into the preprocessed
format required for training.

Usage:
    python3 preprocess_data.py --raw-dir ./raw_data --output-dir ./training-data

Input Structure (raw_data/):
    binaries/          - Raw ELF binaries
    ground_truth/      - Corresponding C source files
    
Output Structure (training-data/):
    preprocessed/gnn/        - GNN training data
    preprocessed/llm/        - LLM training data
    preprocessed/diffusion/  - Diffusion training data
    preprocessed/rl/         - RL training data
    splits.json              - Train/val/test splits
    metadata.json            - Dataset metadata
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
import subprocess
import hashlib
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import random

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocesses raw data for training."""
    
    def __init__(self, raw_dir, output_dir):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        for subdir in ['gnn', 'llm', 'diffusion', 'rl']:
            (self.output_dir / 'preprocessed' / subdir).mkdir(parents=True, exist_ok=True)
        
        self.metadata = {
            'dataset_version': '1.0.0',
            'num_samples': 0,
            'obfuscation_types': defaultdict(int),
            'processing_stats': {}
        }
    
    def extract_ghidra_features(self, binary_path):
        """
        Extract features from binary using Ghidra.
        
        In production, this would call Ghidra headless analyzer.
        For now, we simulate the expected output format.
        """
        # TODO: Replace with actual Ghidra integration
        # Command would be:
        # analyzeHeadless /tmp ghidra_project -import binary_path -postScript extract_features.py
        
        logger.info(f"Extracting features from {binary_path.name}...")
        
        # Simulate Ghidra output
        ghidra_output = {
            'disassembly': self._extract_disassembly(binary_path),
            'cfg': self._extract_cfg(binary_path),
            'pcode': self._extract_pcode(binary_path),
            'functions': self._extract_functions(binary_path)
        }
        
        return ghidra_output
    
    def _extract_disassembly(self, binary_path):
        """Extract disassembly using objdump."""
        try:
            result = subprocess.run(
                ['objdump', '-d', str(binary_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout
        except Exception as e:
            logger.error(f"Failed to disassemble {binary_path}: {e}")
            return ""
    
    def _extract_cfg(self, binary_path):
        """
        Extract control flow graph.
        
        In production, use Ghidra's CFG API.
        Here we simulate a simple CFG structure.
        """
        # Simulate CFG
        num_blocks = random.randint(5, 50)
        num_edges = random.randint(num_blocks, num_blocks * 2)
        
        cfg = {
            'num_blocks': num_blocks,
            'num_edges': num_edges,
            'blocks': [
                {
                    'id': i,
                    'address': f'0x{0x401000 + i*16:x}',
                    'size': random.randint(4, 64)
                }
                for i in range(num_blocks)
            ],
            'edges': [
                {
                    'from': random.randint(0, num_blocks-1),
                    'to': random.randint(0, num_blocks-1),
                    'type': random.choice(['sequential', 'branch', 'call', 'return'])
                }
                for _ in range(num_edges)
            ]
        }
        
        return cfg
    
    def _extract_pcode(self, binary_path):
        """Extract P-Code intermediate representation."""
        # Simulate P-Code instructions
        instructions = [
            'COPY', 'LOAD', 'STORE', 'BRANCH', 'CBRANCH',
            'CALL', 'RETURN', 'INT_ADD', 'INT_SUB', 'INT_MULT',
            'INT_DIV', 'INT_AND', 'INT_OR', 'INT_XOR'
        ]
        
        num_instructions = random.randint(50, 500)
        pcode = [
            {
                'opcode': random.choice(instructions),
                'address': f'0x{0x401000 + i*4:x}',
                'operands': random.randint(0, 3)
            }
            for i in range(num_instructions)
        ]
        
        return pcode
    
    def _extract_functions(self, binary_path):
        """Extract function boundaries."""
        # Simulate function list
        functions = [
            {
                'name': 'main',
                'address': '0x401000',
                'size': 256
            },
            {
                'name': 'helper',
                'address': '0x401100',
                'size': 128
            }
        ]
        
        return functions
    
    def create_gnn_sample(self, sample_id, ghidra_output):
        """Create GNN training sample."""
        cfg = ghidra_output['cfg']
        pcode = ghidra_output['pcode']
        
        # Create node features (128-dim)
        num_nodes = cfg['num_blocks']
        node_features = np.random.randn(num_nodes, 128).astype(np.float32)
        
        # Create edge index
        edges = cfg['edges']
        edge_index = [
            [e['from'] for e in edges],
            [e['to'] for e in edges]
        ]
        
        # Create edge attributes (control flow vs data flow)
        edge_attr = []
        for e in edges:
            if e['type'] in ['sequential', 'branch']:
                edge_attr.append([1.0, 0.0])  # Control flow
            else:
                edge_attr.append([0.5, 0.5])  # Mixed
        
        # Generate labels (simulate junk detection)
        # In production, compare with clean CFG to label junk nodes
        labels = [random.randint(0, 1) for _ in range(num_nodes)]
        
        gnn_sample = {
            'sample_id': sample_id,
            'node_features': node_features.tolist(),
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'labels': labels,
            'metadata': {
                'num_nodes': num_nodes,
                'num_edges': len(edges),
                'num_junk': sum(labels),
                'cyclomatic_complexity': len(edges) - num_nodes + 2
            }
        }
        
        return gnn_sample
    
    def create_llm_sample(self, sample_id, ghidra_output, source_code, cfg_embedding):
        """Create LLM training sample."""
        # Extract assembly for main function
        disassembly = ghidra_output['disassembly']
        
        # Truncate if too long
        max_asm_lines = 100
        asm_lines = disassembly.split('\n')[:max_asm_lines]
        assembly = '\n'.join(asm_lines)
        
        llm_sample = {
            'sample_id': sample_id,
            'assembly': assembly,
            'cfg_embedding': cfg_embedding,
            'source_code': source_code,
            'metadata': {
                'num_instructions': len(ghidra_output['pcode']),
                'num_basic_blocks': ghidra_output['cfg']['num_blocks'],
                'function_name': ghidra_output['functions'][0]['name'] if ghidra_output['functions'] else 'unknown',
                'has_loops': random.choice([True, False]),
                'has_conditionals': random.choice([True, False])
            }
        }
        
        return llm_sample
    
    def create_diffusion_sample(self, sample_id, source_code, assembly_embedding, tokenizer=None):
        """Create diffusion training sample."""
        # Tokenize source code
        # In production, use actual CodeLlama tokenizer
        if tokenizer is None:
            # Simulate tokenization (word-level for demo)
            tokens = [hash(word) % 50000 for word in source_code.split()]
            # Pad to 256 tokens
            tokens = tokens[:256] + [0] * max(0, 256 - len(tokens))
        else:
            tokens = tokenizer.encode(source_code, max_length=256, padding='max_length')
        
        diffusion_sample = {
            'sample_id': sample_id,
            'tokens': tokens,
            'token_length': len([t for t in tokens if t != 0]),
            'condition': {
                'assembly_embedding': assembly_embedding,
                'cfg_features': {
                    'num_blocks': random.randint(3, 20),
                    'num_edges': random.randint(3, 30),
                    'complexity': random.randint(1, 10),
                    'has_loops': random.choice([True, False])
                },
                'llm_confidence': random.uniform(0.6, 0.95),
                'verification_hints': {
                    'failed_constraints': [],
                    'suggested_fixes': []
                }
            },
            'source_code': source_code,
            'metadata': {
                'is_refinement': False,
                'iteration': 0,
                'previous_errors': []
            }
        }
        
        return diffusion_sample
    
    def create_rl_sample(self, sample_id, ghidra_output):
        """Create RL training sample (episode trajectory)."""
        # Simulate episode trajectory
        num_steps = random.randint(1, 5)
        
        initial_state = np.random.randn(128).tolist()
        trajectory = []
        
        state = initial_state
        total_reward = 0
        
        for step in range(num_steps):
            action = random.randint(0, 3)  # 4 possible actions
            reward = random.uniform(0, 10.5)
            total_reward += reward
            next_state = np.random.randn(128).tolist()
            done = (step == num_steps - 1)
            
            trajectory.append({
                'step': step,
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'info': {
                    'compilation_success': reward > 5.0,
                    'z3_sat': reward > 7.0,
                    'num_violations': max(0, int((10.5 - reward) * 2))
                }
            })
            
            state = next_state
        
        rl_sample = {
            'sample_id': sample_id,
            'initial_state': {
                'pcode_features': initial_state,
                'cfg_complexity': ghidra_output['cfg']['num_blocks'],
                'num_constraints': random.randint(5, 20),
                'initial_confidence': random.uniform(0.5, 0.9)
            },
            'trajectory': trajectory,
            'total_reward': total_reward,
            'num_steps': num_steps,
            'success': total_reward > 20.0
        }
        
        return rl_sample
    
    def process_sample(self, binary_path, source_path, sample_id):
        """Process a single binary-source pair."""
        try:
            # Read source code
            with open(source_path) as f:
                source_code = f.read()
            
            # Extract features with Ghidra
            ghidra_output = self.extract_ghidra_features(binary_path)
            
            # Create GNN sample
            gnn_sample = self.create_gnn_sample(sample_id, ghidra_output)
            with open(self.output_dir / 'preprocessed' / 'gnn' / f'{sample_id}.json', 'w') as f:
                json.dump(gnn_sample, f)
            
            # Simulate GNN embedding (768-dim)
            cfg_embedding = np.random.randn(768).astype(np.float32).tolist()
            
            # Create LLM sample
            llm_sample = self.create_llm_sample(sample_id, ghidra_output, source_code, cfg_embedding)
            with open(self.output_dir / 'preprocessed' / 'llm' / f'{sample_id}.json', 'w') as f:
                json.dump(llm_sample, f)
            
            # Create Diffusion sample
            diffusion_sample = self.create_diffusion_sample(sample_id, source_code, cfg_embedding)
            with open(self.output_dir / 'preprocessed' / 'diffusion' / f'{sample_id}.json', 'w') as f:
                json.dump(diffusion_sample, f)
            
            # Create RL sample
            rl_sample = self.create_rl_sample(sample_id, ghidra_output)
            with open(self.output_dir / 'preprocessed' / 'rl' / f'{sample_id}.json', 'w') as f:
                json.dump(rl_sample, f)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process {sample_id}: {e}")
            return False
    
    def create_splits(self, sample_ids, train_ratio=0.8, val_ratio=0.1):
        """Create train/val/test splits."""
        random.shuffle(sample_ids)
        
        n = len(sample_ids)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        splits = {
            'train': sample_ids[:train_end],
            'validation': sample_ids[train_end:val_end],
            'test': sample_ids[val_end:],
            'split_ratios': {
                'train': train_ratio,
                'validation': val_ratio,
                'test': 1 - train_ratio - val_ratio
            }
        }
        
        with open(self.output_dir / 'splits.json', 'w') as f:
            json.dump(splits, f, indent=2)
        
        logger.info(f"Created splits: Train={len(splits['train'])}, "
                   f"Val={len(splits['validation'])}, Test={len(splits['test'])}")
        
        return splits
    
    def save_metadata(self):
        """Save dataset metadata."""
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Saved metadata: {self.metadata['num_samples']} samples")
    
    def run(self):
        """Run preprocessing pipeline."""
        logger.info("Starting data preprocessing...")
        
        # Find all binary-source pairs
        binary_dir = self.raw_dir / 'binaries'
        source_dir = self.raw_dir / 'ground_truth'
        
        if not binary_dir.exists() or not source_dir.exists():
            logger.error(f"Raw data directories not found in {self.raw_dir}")
            return False
        
        # Match binaries with source files
        binary_files = sorted(binary_dir.glob('*.bin'))
        sample_ids = []
        
        for binary_path in tqdm(binary_files, desc='Processing samples'):
            # Find corresponding source file
            source_name = binary_path.stem + '.c'
            source_path = source_dir / source_name
            
            if not source_path.exists():
                logger.warning(f"No source file for {binary_path.name}, skipping")
                continue
            
            sample_id = binary_path.stem
            
            if self.process_sample(binary_path, source_path, sample_id):
                sample_ids.append(sample_id)
                self.metadata['num_samples'] += 1
        
        # Create splits
        self.create_splits(sample_ids)
        
        # Save metadata
        self.save_metadata()
        
        logger.info(f"Preprocessing complete! Processed {self.metadata['num_samples']} samples")
        return True


def main():
    parser = argparse.ArgumentParser(description='Preprocess DeObfusca-AI training data')
    parser.add_argument('--raw-dir', type=str, required=True,
                       help='Directory containing raw binaries and source code')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for preprocessed data')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training set ratio (default: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Validation set ratio (default: 0.1)')
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = DataPreprocessor(args.raw_dir, args.output_dir)
    
    # Run preprocessing
    success = preprocessor.run()
    
    if success:
        logger.info("=" * 80)
        logger.info("Preprocessing successful!")
        logger.info(f"Preprocessed data saved to: {args.output_dir}")
        logger.info("You can now run: python3 train_all.py --data-dir {}".format(args.output_dir))
        logger.info("=" * 80)
        return 0
    else:
        logger.error("Preprocessing failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
