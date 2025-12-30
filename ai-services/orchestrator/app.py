from flask import Flask, request, jsonify
import requests
import os
from typing import Dict, List
import traceback

app = Flask(__name__)

# Service URLs - Advanced Architecture
GHIDRA_URL = os.getenv('GHIDRA_SERVICE_URL', 'http://ghidra-service:5001')
CPG_URL = os.getenv('CPG_SERVICE_URL', 'http://cpg-service:5005')
GNN_URL = os.getenv('GNN_SERVICE_URL', 'http://gnn-service:5002')
LLM_URL = os.getenv('LLM_SERVICE_URL', 'http://llm-service:5003')
RL_URL = os.getenv('RL_SERVICE_URL', 'http://rl-service:5004')
DIFFUSION_URL = os.getenv('DIFFUSION_SERVICE_URL', 'http://diffusion-service:5006')
MULTI_AGENT_URL = os.getenv('MULTI_AGENT_SERVICE_URL', 'http://multi-agent-service:5007')
COT_URL = os.getenv('COT_SERVICE_URL', 'http://cot-service:5008')

# Verify-Refine Loop Configuration
MAX_REFINEMENT_ITERATIONS = int(os.getenv('MAX_REFINEMENT_ITERATIONS', '3'))
REWARD_THRESHOLD = float(os.getenv('REWARD_THRESHOLD', '10.5'))

# Timeout configurations (in seconds)
GHIDRA_TIMEOUT = 300
CPG_TIMEOUT = 60
GNN_TIMEOUT = 60
LLM_TIMEOUT = 180
RL_TIMEOUT = 60
DIFFUSION_TIMEOUT = 120
MULTI_AGENT_TIMEOUT = 120
COT_TIMEOUT = 60

def safe_request(url, method='POST', json_data=None, timeout=30, service_name=''):
    try:
        if method == 'POST':
            response = requests.post(url, json=json_data, timeout=timeout)
        else:
            response = requests.get(url, timeout=timeout)
        
        if response.status_code >= 500:
            return None, f"{service_name} server error: {response.status_code}"
        
        return response, None
    except requests.Timeout:
        return None, f"{service_name} timeout (>{timeout}s)"
    except requests.ConnectionError:
        return None, f"{service_name} connection failed (unreachable)"
    except Exception as e:
        return None, f"{service_name} error: {str(e)}"

@app.route('/health', methods=['GET'])
def health():
    statuses = {}
    all_healthy = True
    
    services = [
        ('ghidra', GHIDRA_URL),
        ('cpg', CPG_URL),
        ('gnn', GNN_URL),
        ('llm', LLM_URL),
        ('rl', RL_URL),
        ('diffusion', DIFFUSION_URL),
        ('multi_agent', MULTI_AGENT_URL),
        ('cot', COT_URL)
    ]
    
    for name, url in services:
        try:
            resp = requests.get(f'{url}/health', timeout=5)
            if resp.status_code == 200:
                statuses[name] = {'status': 'ok', 'healthy': True}
            else:
                statuses[name] = {'status': 'error', 'code': resp.status_code, 'healthy': False}
                all_healthy = False
        except requests.Timeout:
            statuses[name] = {'status': 'timeout', 'healthy': False}
            all_healthy = False
        except Exception as e:
            statuses[name] = {'status': 'error', 'message': str(e), 'healthy': False}
            all_healthy = False
    
    return jsonify({
        'status': 'ok' if all_healthy else 'degraded',
        'all_services_healthy': all_healthy,
        'architecture': 'verify-refine-loop-enhanced',
        'services': statuses,
        'features': ['diffusion', 'multi-agent', 'chain-of-thought']
    })

@app.route('/sanitize', methods=['POST'])
def sanitize():
    try:
        data = request.json
        file_path = data.get('file_path')
        enable_refinement = data.get('enable_refinement', True)
        max_iterations = data.get('max_iterations', MAX_REFINEMENT_ITERATIONS)
        
        if not file_path:
            return jsonify({'error': 'file_path required'}), 400
        
        # STEP 1: Ghidra Analysis - Extract P-Code
        print("Step 1: Ghidra Analysis...")
        try:
            ghidra_resp, ghidra_err = safe_request(
                f'{GHIDRA_URL}/analyze',
                json_data={'file_path': file_path, 'project_name': os.path.basename(file_path)},
                timeout=GHIDRA_TIMEOUT,
                service_name='Ghidra'
            )
            
            if ghidra_err:
                return jsonify({'error': f'Ghidra analysis failed: {ghidra_err}'}), 503
            
            if ghidra_resp.status_code != 200:
                return jsonify({'error': f'Ghidra returned {ghidra_resp.status_code}'}), 500
            
            analysis_data = ghidra_resp.json()
            
            if not analysis_data.get('functions'):
                return jsonify({'error': 'No functions found in binary'}), 400
        except Exception as e:
            return jsonify({'error': f'Ghidra step failed: {str(e)}'}), 500
        
        # STEP 2: Build Code Property Graph (CPG)
        print("Step 2: Building CPG (Hypergraph)...")
        cpg_analysis = {}
        
        for func in analysis_data['functions']:
            pcode = func.get('pcode', [])
            cfg = func.get('cfg', {})
            
            try:
                cpg_resp, cpg_err = safe_request(
                    f'{CPG_URL}/build-cpg',
                    json_data={'pcode': pcode, 'cfg': cfg},
                    timeout=CPG_TIMEOUT,
                    service_name='CPG'
                )
                
                if cpg_err:
                    print(f"CPG warning: {cpg_err}")
                    cpg_analysis[func['name']] = None
                elif cpg_resp.status_code == 200:
                    cpg_analysis[func['name']] = cpg_resp.json().get('cpg')
                else:
                    cpg_analysis[func['name']] = None
            except Exception as e:
                print(f"CPG error for {func['name']}: {e}")
                cpg_analysis[func['name']] = None
        
        # STEP 3: Edge-Augmented Graph Transformer (EAGT) - Detect obfuscation
        print("Step 3: Graph Transformer Sanitization...")
        sanitized_functions = []
        
        for func in analysis_data['functions']:
            func_cpg = cpg_analysis.get(func['name'], {})
            
            try:
                gnn_resp, gnn_err = safe_request(
                    f'{GNN_URL}/sanitize',
                    json_data={'pcode': func.get('pcode', []), 'cfg': func.get('cfg', {}), 'cpg': func_cpg},
                    timeout=GNN_TIMEOUT,
                    service_name='GNN'
                )
                
                if gnn_err:
                    print(f"GNN warning: {gnn_err}")
                    sanitized_functions.append({'name': func['name'], 'sanitized_features': func.get('pcode', [])})
                elif gnn_resp.status_code == 200:
                    sanitized_data = gnn_resp.json()
                    sanitized_functions.append({
                        'name': func['name'],
                        'sanitized_features': sanitized_data.get('sanitized_features', func.get('pcode', [])),
                        'summary': f"Function with {len(sanitized_data.get('sanitized_features', []))} instructions"
                    })
                else:
                    sanitized_functions.append({'name': func['name'], 'sanitized_features': func.get('pcode', [])})
            except Exception as e:
                print(f"GNN error: {e}")
                sanitized_functions.append({'name': func['name'], 'sanitized_features': func.get('pcode', [])})
        
        # STEP 4 & 5: Parallel Candidate Generation + Collaborative Refinement Loop
        # NEW ARCHITECTURE: All 3 generators (LLM, Diffusion, Multi-Agent) produce candidates,
        # RL verifier scores them, and ALL generators refine the BEST candidate together.
        print("Step 4: Parallel Candidate Generation (LLM + Diffusion + Multi-Agent)...")
        refinement_history = []
        best_decompilation = None
        best_reward = -float('inf')
        current_best_code = {}  # Track best code per function for collaborative refinement
        
        for iteration in range(max_iterations):
            print(f"\n{'='*60}")
            print(f"  ITERATION {iteration + 1}/{max_iterations}")
            print(f"{'='*60}")
            
            # ─────────────────────────────────────────────────────────────
            # PHASE A: Generate candidates from ALL three services in parallel
            # ─────────────────────────────────────────────────────────────
            print("  Phase A: Generating candidates from LLM, Diffusion, Multi-Agent...")
            
            candidates = {}  # {func_name: [{source: 'llm'|'diffusion'|'multi-agent', code: str, reward: float}]}
            
            for func in sanitized_functions:
                func_name = func['name']
                candidates[func_name] = []
                features = func.get('sanitized_features', [])
                cpg = cpg_analysis.get(func_name, {})
                
                # Base code for refinement iterations (empty on first, best on subsequent)
                base_code = current_best_code.get(func_name, '')
                
                # ── Candidate 1: LLM ──
                try:
                    if iteration == 0 or not base_code:
                        # First iteration: fresh generation
                        llm_resp, llm_err = safe_request(
                            f'{LLM_URL}/decompile',
                            json_data={'sanitized_features': features},
                            timeout=LLM_TIMEOUT,
                            service_name='LLM'
                        )
                    else:
                        # Subsequent iterations: refine best code
                        llm_resp, llm_err = safe_request(
                            f'{LLM_URL}/refine',
                            json_data={
                                'current_code': base_code,
                                'sanitized_features': features,
                                'feedback': func.get('refinement_feedback', '')
                            },
                            timeout=LLM_TIMEOUT,
                            service_name='LLM'
                        )
                    
                    if not llm_err and llm_resp and llm_resp.status_code == 200:
                        llm_code = llm_resp.json().get('code', llm_resp.json().get('refined_code', ''))
                        if llm_code:
                            candidates[func_name].append({'source': 'llm', 'code': llm_code, 'reward': 0.0})
                            print(f"    [+] LLM candidate for {func_name}")
                    else:
                        print(f"    [-] LLM failed for {func_name}: {llm_err}")
                except Exception as e:
                    print(f"    [-] LLM error for {func_name}: {e}")
                
                # ── Candidate 2: Diffusion ──
                try:
                    if iteration == 0 or not base_code:
                        diff_resp, diff_err = safe_request(
                            f'{DIFFUSION_URL}/generate',
                            json_data={'binary_features': features, 'max_length': 512},
                            timeout=DIFFUSION_TIMEOUT,
                            service_name='Diffusion'
                        )
                    else:
                        diff_resp, diff_err = safe_request(
                            f'{DIFFUSION_URL}/refine',
                            json_data={
                                'current_code': base_code,
                                'sanitized_features': features,
                                'feedback': func.get('refinement_feedback', '')
                            },
                            timeout=DIFFUSION_TIMEOUT,
                            service_name='Diffusion'
                        )
                    
                    if not diff_err and diff_resp and diff_resp.status_code == 200:
                        diff_code = diff_resp.json().get('code', diff_resp.json().get('refined_code', ''))
                        if diff_code:
                            candidates[func_name].append({'source': 'diffusion', 'code': diff_code, 'reward': 0.0})
                            print(f"    [+] Diffusion candidate for {func_name}")
                    else:
                        print(f"    [-] Diffusion failed for {func_name}: {diff_err}")
                except Exception as e:
                    print(f"    [-] Diffusion error for {func_name}: {e}")
                
                # ── Candidate 3: Multi-Agent ──
                try:
                    if iteration == 0 or not base_code:
                        ma_resp, ma_err = safe_request(
                            f'{MULTI_AGENT_URL}/analyze',
                            json_data={'features': features, 'cpg': cpg},
                            timeout=MULTI_AGENT_TIMEOUT,
                            service_name='Multi-Agent'
                        )
                    else:
                        ma_resp, ma_err = safe_request(
                            f'{MULTI_AGENT_URL}/refine',
                            json_data={
                                'current_code': base_code,
                                'features': features,
                                'cpg': cpg,
                                'feedback': func.get('refinement_feedback', '')
                            },
                            timeout=MULTI_AGENT_TIMEOUT,
                            service_name='Multi-Agent'
                        )
                    
                    if not ma_err and ma_resp and ma_resp.status_code == 200:
                        ma_code = ma_resp.json().get('code', ma_resp.json().get('refined_code', ''))
                        if ma_code:
                            candidates[func_name].append({'source': 'multi-agent', 'code': ma_code, 'reward': 0.0})
                            print(f"    [+] Multi-Agent candidate for {func_name}")
                    else:
                        print(f"    [-] Multi-Agent failed for {func_name}: {ma_err}")
                except Exception as e:
                    print(f"    [-] Multi-Agent error for {func_name}: {e}")
                
                # Fallback: if no candidates, keep base code or generate placeholder
                if not candidates[func_name]:
                    if base_code:
                        candidates[func_name].append({'source': 'fallback', 'code': base_code, 'reward': 0.0})
                    else:
                        candidates[func_name].append({'source': 'fallback', 'code': f'// Decompilation failed for {func_name}', 'reward': 0.0})
            
            # ─────────────────────────────────────────────────────────────
            # PHASE B: Score ALL candidates with RL Verifier
            # ─────────────────────────────────────────────────────────────
            print("\n  Phase B: Scoring all candidates with RL Verifier (Z3)...")
            
            for func_name, func_candidates in candidates.items():
                for candidate in func_candidates:
                    try:
                        verify_resp, verify_err = safe_request(
                            f'{RL_URL}/verify',
                            json_data={
                                'source_code': candidate['code'],
                                'original_binary_path': file_path,
                                'use_symbolic': True
                            },
                            timeout=RL_TIMEOUT,
                            service_name='RL'
                        )
                        
                        if not verify_err and verify_resp and verify_resp.status_code == 200:
                            verification = verify_resp.json()
                            candidate['reward'] = verification.get('reward', 0.0)
                            candidate['verification'] = verification
                            candidate['feedback'] = verification.get('feedback', '')
                        else:
                            candidate['reward'] = 0.0
                            candidate['verification'] = {'error': verify_err}
                            candidate['feedback'] = ''
                    except Exception as e:
                        candidate['reward'] = 0.0
                        candidate['verification'] = {'error': str(e)}
                        candidate['feedback'] = ''
                
                # Sort candidates by reward (descending)
                func_candidates.sort(key=lambda c: c['reward'], reverse=True)
                
                best_candidate = func_candidates[0]
                print(f"    {func_name}: Best={best_candidate['source']} (reward={best_candidate['reward']:.2f})")
                for c in func_candidates[1:]:
                    print(f"      -> {c['source']}: reward={c['reward']:.2f}")
            
            # ─────────────────────────────────────────────────────────────
            # PHASE C: Select BEST candidate per function for next iteration
            # ─────────────────────────────────────────────────────────────
            print("\n  Phase C: Selecting best candidates for collaborative refinement...")
            
            iteration_decompilation = {}
            iteration_verification = {}
            total_reward = 0
            
            for func_name, func_candidates in candidates.items():
                best = func_candidates[0]
                iteration_decompilation[func_name] = best['code']
                iteration_verification[func_name] = best.get('verification', {})
                total_reward += best['reward']
                
                # Update current_best_code for next iteration's collaborative refinement
                current_best_code[func_name] = best['code']
                
                # Store feedback for refinement
                for sf in sanitized_functions:
                    if sf['name'] == func_name:
                        sf['refinement_feedback'] = best.get('feedback', '')
                        sf['best_source'] = best['source']
            
            # Record this iteration
            refinement_history.append({
                'iteration': iteration + 1,
                'reward': total_reward,
                'decompilation': iteration_decompilation,
                'verification': iteration_verification,
                'candidate_breakdown': {
                    fn: [{'source': c['source'], 'reward': c['reward']} for c in cands]
                    for fn, cands in candidates.items()
                }
            })
            
            print(f"\n  Iteration {iteration + 1} Total Reward: {total_reward:.2f}")
            
            # Update global best
            if total_reward > best_reward:
                best_reward = total_reward
                best_decompilation = iteration_decompilation.copy()
                print(f"  * New best! (reward={best_reward:.2f})")
            
            # Check if we've reached acceptable quality
            if total_reward >= REWARD_THRESHOLD or not enable_refinement:
                print(f"\n  [OK] Acceptable quality reached (reward: {total_reward:.2f} >= threshold: {REWARD_THRESHOLD})")
                break
            
            if iteration < max_iterations - 1:
                print(f"\n  -> Continuing to iteration {iteration + 2} (all generators will refine best code)...")
        
        # Return best result
        return jsonify({
            'cpg_analysis': cpg_analysis,
            'decompilation': best_decompilation or {},
            'verification': refinement_history[-1]['verification'] if refinement_history else {},
            'refinement_history': refinement_history,
            'final_reward': best_reward,
            'iterations_used': len(refinement_history),
            'architecture': 'parallel-candidate-collaborative-refinement',
            'success': best_reward >= REWARD_THRESHOLD
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/decompile', methods=['POST'])
def decompile():
    try:
        data = request.json
        features = data.get('features')
        
        if not features:
            return jsonify({'error': 'features required'}), 400
        
        # Sanitize with GNN
        gnn_resp = requests.post(
            f'{GNN_URL}/sanitize',
            json={'pcode': features, 'cfg': {'edges': []}},
            timeout=60
        )
        
        sanitized = gnn_resp.json()['sanitized_features'] if gnn_resp.status_code == 200 else features
        
        # Decompile with LLM
        llm_resp = requests.post(
            f'{LLM_URL}/decompile',
            json={'sanitized_features': sanitized},
            timeout=120
        )
        
        if llm_resp.status_code != 200:
            return jsonify({'error': 'Decompilation failed'}), 500
        
        return jsonify(llm_resp.json())
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def ai_chat():
    try:
        data = request.json
        message = data.get('message', '')
        context = data.get('context', {})
        
        # Use CoT service for reasoning
        cot_resp = requests.post(
            f'{COT_URL}/explain',
            json={
                'step_number': 1,
                'code_fragment': context.get('code', '')
            },
            timeout=30
        )
        
        if cot_resp.status_code == 200:
            explanation = cot_resp.json()
            return jsonify({
                'response': f"Based on my analysis: {explanation.get('detailed_reasoning', [])}",
                'reasoning': explanation
            })
        else:
            return jsonify({
                'response': 'I can help you understand this code. What would you like to know?'
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/advanced-decompile', methods=['POST'])
def advanced_decompile():
    try:
        data = request.json
        binary_features = data.get('binary_features', [])
        method = data.get('method', 'multi-agent')  # 'diffusion', 'multi-agent', or 'cot'
        
        if method == 'diffusion':
            # Use diffusion model
            resp = requests.post(
                f'{DIFFUSION_URL}/generate',
                json={'binary_features': binary_features, 'max_length': 512},
                timeout=120
            )
        elif method == 'multi-agent':
            # Use multi-agent system
            resp = requests.post(
                f'{MULTI_AGENT_URL}/decompile',
                json={'code_fragment': str(binary_features), 'context': {}},
                timeout=120
            )
        elif method == 'cot':
            # Use Chain-of-Thought
            resp = requests.post(
                f'{COT_URL}/reason',
                json={'binary_code': str(binary_features), 'context': {}},
                timeout=120
            )
        else:
            return jsonify({'error': 'Invalid method'}), 400
        
        if resp.status_code == 200:
            result = resp.json()
            result['method_used'] = method
            return jsonify(result)
        else:
            return jsonify({'error': 'Advanced decompilation failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
