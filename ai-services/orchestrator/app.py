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
    """
    Safe request wrapper with error handling and fallbacks.
    """
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
    """Check health of all services in advanced architecture."""
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
    """
    Advanced Verify-Refine Loop Pipeline with comprehensive error handling.
    Binary → Ghidra → CPG → Graph Transformer → Hierarchical LLM → Symbolic Verifier
    
    With iterative refinement based on verification feedback.
    """
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
        
        # STEP 4 & 5: Verify-Refine Loop
        print("Step 4: Hierarchical LLM Decompilation with Verify-Refine Loop...")
        refinement_history = []
        best_decompilation = None
        best_reward = -float('inf')
        
        for iteration in range(max_iterations):
            print(f"  Iteration {iteration + 1}/{max_iterations}")
            
            # Decompile entire binary
            try:
                llm_resp, llm_err = safe_request(
                    f'{LLM_URL}/decompile-binary',
                    json_data={'functions': sanitized_functions},
                    timeout=LLM_TIMEOUT,
                    service_name='LLM'
                )
                
                if llm_err:
                    print(f"LLM error: {llm_err}")
                    if iteration == 0:
                        return jsonify({'error': f'LLM decompilation failed: {llm_err}'}), 503
                    break
                
                if llm_resp.status_code != 200:
                    print(f"LLM returned {llm_resp.status_code}")
                    break
                
                decompiled = llm_resp.json().get('decompiled', {})
            except Exception as e:
                print(f"LLM step error: {e}")
                if iteration == 0:
                    return jsonify({'error': f'LLM decompilation error: {str(e)}'}), 500
                break
            
            # STEP 5: Neural-Symbolic Verification with Z3
            print(f"  Step 5: Verifying with symbolic execution...")
            verification_results = {}
            total_reward = 0
            
            for func_name, source_code in decompiled.items():
                try:
                    verify_resp, verify_err = safe_request(
                        f'{RL_URL}/verify',
                        json_data={'source_code': source_code, 'original_binary_path': file_path, 'use_symbolic': True},
                        timeout=RL_TIMEOUT,
                        service_name='RL'
                    )
                    
                    if verify_err:
                        print(f"RL warning: {verify_err}")
                        verification_results[func_name] = {'reward': 0.0, 'error': verify_err}
                    elif verify_resp.status_code == 200:
                        verification = verify_resp.json()
                        verification_results[func_name] = verification
                        total_reward += verification.get('reward', 0)
                    else:
                        verification_results[func_name] = {'reward': 0.0, 'error': f'RL returned {verify_resp.status_code}'}
                except Exception as e:
                    print(f"RL error: {e}")
                    verification_results[func_name] = {'reward': 0.0, 'error': str(e)}
            
            # Record this iteration
            refinement_history.append({
                'iteration': iteration + 1,
                'reward': total_reward,
                'decompilation': decompiled,
                'verification': verification_results
            })
            
            # Check if this is best so far
            if total_reward > best_reward:
                best_reward = total_reward
                best_decompilation = decompiled
            
            # Check if we've reached acceptable quality
            if total_reward >= REWARD_THRESHOLD or not enable_refinement:
                print(f"  ✓ Acceptable quality reached (reward: {total_reward})")
                break
            
            # STEP 6: Generate feedback for refinement
            print(f"  Generating feedback for refinement (reward: {total_reward})...")
            feedback_prompts = []
            for func_name, verification in verification_results.items():
                if verification.get('feedback'):
                    feedback_prompts.append({'function': func_name, 'feedback': verification['feedback']})
            
            # Select refinement strategy based on iteration
            if feedback_prompts and iteration < max_iterations - 1:
                refinement_method = ['diffusion', 'multi-agent', 'cot'][iteration % 3]
                print(f"  Applying {refinement_method} refinement...")
                
                for sf in sanitized_functions:
                    for fp in feedback_prompts:
                        if sf['name'] == fp['function']:
                            sf['refinement_feedback'] = fp['feedback']
                            
                            # Call appropriate refinement service
                            try:
                                if refinement_method == 'diffusion':
                                    refine_resp, refine_err = safe_request(
                                        f'{DIFFUSION_URL}/refine',
                                        json_data={'sanitized_features': sf.get('sanitized_features', []), 'feedback': fp['feedback']},
                                        timeout=DIFFUSION_TIMEOUT,
                                        service_name='Diffusion'
                                    )
                                elif refinement_method == 'multi-agent':
                                    refine_resp, refine_err = safe_request(
                                        f'{MULTI_AGENT_URL}/refine',
                                        json_data={'current_code': decompiled.get(sf['name'], ''), 'feedback': fp['feedback']},
                                        timeout=MULTI_AGENT_TIMEOUT,
                                        service_name='Multi-Agent'
                                    )
                                else:  # cot
                                    refine_resp, refine_err = safe_request(
                                        f'{COT_URL}/refine',
                                        json_data={'current_code': decompiled.get(sf['name'], ''), 'feedback': fp['feedback']},
                                        timeout=COT_TIMEOUT,
                                        service_name='CoT'
                                    )
                                
                                if refine_err:
                                    print(f"Refinement warning: {refine_err}")
                            except Exception as e:
                                print(f"Refinement error: {e}")
        
        # Return best result
        return jsonify({
            'cpg_analysis': cpg_analysis,
            'decompilation': best_decompilation or {},
            'verification': refinement_history[-1]['verification'] if refinement_history else {},
            'refinement_history': refinement_history,
            'final_reward': best_reward,
            'iterations_used': len(refinement_history),
            'success': best_reward >= REWARD_THRESHOLD
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500
        # STEP 2: Build Code Property Graph (CPG)
        print("Step 2: Building CPG (Hypergraph)...")
        cpg_analysis = {}
        
        for func in analysis_data['functions']:
            pcode = func['pcode']
            cfg = func['cfg']
            
            cpg_resp = requests.post(
                f'{CPG_URL}/build-cpg',
                json={'pcode': pcode, 'cfg': cfg},
                timeout=60
            )
            
            if cpg_resp.status_code == 200:
                cpg_analysis[func['name']] = cpg_resp.json()['cpg']
        
        # STEP 3: Edge-Augmented Graph Transformer (EAGT) - Detect obfuscation
        print("Step 3: Graph Transformer Sanitization...")
        sanitized_functions = []
        
        for func in analysis_data['functions']:
            func_cpg = cpg_analysis.get(func['name'], {})
            
            gnn_resp = requests.post(
                f'{GNN_URL}/sanitize',
                json={
                    'pcode': func['pcode'],
                    'cfg': func['cfg'],
                    'cpg': func_cpg  # Pass CPG for dominator-aware attention
                },
                timeout=60
            )
            
            if gnn_resp.status_code == 200:
                sanitized_data = gnn_resp.json()
                sanitized_functions.append({
                    'name': func['name'],
                    'sanitized_features': sanitized_data['sanitized_features'],
                    'summary': f"Function with {len(sanitized_data['sanitized_features'])} instructions"
                })
        
        # STEP 4: Hierarchical LLM with RAG - Decompile with verify-refine loop
        print("Step 4: Hierarchical LLM Decompilation with Verify-Refine Loop...")
        refinement_history = []
        best_decompilation = None
        best_reward = -float('inf')
        
        for iteration in range(max_iterations):
            print(f"  Iteration {iteration + 1}/{max_iterations}")
            
            # Decompile entire binary
            llm_resp = requests.post(
                f'{LLM_URL}/decompile-binary',
                json={'functions': sanitized_functions},
                timeout=180
            )
            
            if llm_resp.status_code != 200:
                break
            
            decompiled = llm_resp.json()['decompiled']
            
            # STEP 5: Neural-Symbolic Verification with Z3
            print(f"  Step 5: Verifying with symbolic execution...")
            verification_results = {}
            total_reward = 0
            
            for func_name, source_code in decompiled.items():
                verify_resp = requests.post(
                    f'{RL_URL}/verify',
                    json={
                        'source_code': source_code,
                        'original_binary_path': file_path,
                        'use_symbolic': True
                    },
                    timeout=60
                )
                
                if verify_resp.status_code == 200:
                    verification = verify_resp.json()
                    verification_results[func_name] = verification
                    total_reward += verification.get('reward', 0)
            
            # Record this iteration
            refinement_history.append({
                'iteration': iteration + 1,
                'reward': total_reward,
                'decompilation': decompiled,
                'verification': verification_results
            })
            
            # Check if this is best so far
            if total_reward > best_reward:
                best_reward = total_reward
                best_decompilation = decompiled
            
            # Check if we've reached acceptable quality
            if total_reward >= REWARD_THRESHOLD or not enable_refinement:
                print(f"  ✓ Acceptable quality reached (reward: {total_reward})")
                break
            
            # STEP 6: Generate feedback for refinement
            print(f"  Generating feedback for refinement (reward: {total_reward})...")
            feedback_prompts = []
            for func_name, verification in verification_results.items():
                if verification.get('feedback'):
                    feedback_prompts.append({
                        'function': func_name,
                        'feedback': verification['feedback']
                    })
            
            # Update sanitized functions with feedback for next iteration
            if feedback_prompts:
                for sf in sanitized_functions:
                    for fp in feedback_prompts:
                        if sf['name'] == fp['function']:
                            sf['refinement_feedback'] = fp['feedback']
        
        # Return best result
        return jsonify({
            'cpg_analysis': cpg_analysis,
            'decompilation': best_decompilation or {},
            'verification': refinement_history[-1]['verification'] if refinement_history else {},
            'refinement_history': refinement_history,
            'final_reward': best_reward,
            'iterations_used': len(refinement_history),
            'success': best_reward >= REWARD_THRESHOLD
        })
        
    except requests.Timeout:
        return jsonify({'error': 'Service timeout'}), 504
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/decompile', methods=['POST'])
def decompile():
    """
    Legacy endpoint for direct decompilation (bypasses Ghidra).
    Used by Node.js backend for pre-analyzed binaries.
    """
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
    """
    AI chat assistant for code explanation and Q&A.
    Uses Chain-of-Thought reasoning for complex questions.
    """
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
    """
    Advanced decompilation using all new techniques:
    - Diffusion model for generation
    - Multi-agent collaboration
    - Chain-of-Thought reasoning
    """
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
