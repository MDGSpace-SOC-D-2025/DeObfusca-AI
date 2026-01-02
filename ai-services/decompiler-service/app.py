"""
External Decompiler Service
===========================
Integrates with external decompilers (Ghidra, RetDec, Snowman/SK2) to convert
sanitized assembly into C code. This replaces the LLM-based approach.

Pipeline:
  1. Receive sanitized assembly (junk removed by GNN)
  2. Call external decompiler
  3. Post-process and return C code
"""

from flask import Flask, request, jsonify
import subprocess
import tempfile
import os
import re
import shutil
from typing import Optional, Dict, List, Tuple

app = Flask(__name__)

# Configuration
GHIDRA_PATH = os.getenv('GHIDRA_PATH', '/opt/ghidra')
RETDEC_PATH = os.getenv('RETDEC_PATH', '/opt/retdec/bin/retdec-decompiler')
SNOWMAN_PATH = os.getenv('SNOWMAN_PATH', '/opt/snowman/nocode')
TIMEOUT = int(os.getenv('DECOMPILER_TIMEOUT', '120'))


class ExternalDecompiler:
    """Interface to external decompilers."""

    def __init__(self):
        self.available_decompilers = self._detect_available()
        print(f"Available decompilers: {self.available_decompilers}")

    def _detect_available(self) -> List[str]:
        available = []
        
        # Check Ghidra
        ghidra_headless = os.path.join(GHIDRA_PATH, 'support', 'analyzeHeadless')
        if os.path.exists(ghidra_headless) or shutil.which('analyzeHeadless'):
            available.append('ghidra')
        
        # Check RetDec
        if os.path.exists(RETDEC_PATH) or shutil.which('retdec-decompiler'):
            available.append('retdec')
        
        # Check Snowman
        if os.path.exists(SNOWMAN_PATH) or shutil.which('nocode'):
            available.append('snowman')
        
        # Always have fallback
        available.append('pattern')
        
        return available

    def decompile(
        self,
        binary_path: str = None,
        assembly: str = None,
        sanitized_indices: List[int] = None,
        prefer: str = 'auto'
    ) -> Dict:
        """
        Decompile binary or assembly to C code.
        
        Args:
            binary_path: Path to binary file
            assembly: Raw assembly text (if no binary)
            sanitized_indices: Indices of instructions to keep (from GNN)
            prefer: Preferred decompiler ('ghidra', 'retdec', 'snowman', 'auto')
        
        Returns:
            Dict with 'code', 'decompiler_used', 'success', 'errors'
        """
        # Select decompiler
        if prefer == 'auto':
            decompiler = self._select_best_decompiler()
        elif prefer in self.available_decompilers:
            decompiler = prefer
        else:
            decompiler = self._select_best_decompiler()

        # If we have assembly but no binary, create temp binary
        if assembly and not binary_path:
            binary_path = self._assemble_to_binary(assembly, sanitized_indices)
            if not binary_path:
                return self._pattern_decompile(assembly, sanitized_indices)

        # Call appropriate decompiler
        if decompiler == 'ghidra':
            return self._ghidra_decompile(binary_path)
        elif decompiler == 'retdec':
            return self._retdec_decompile(binary_path)
        elif decompiler == 'snowman':
            return self._snowman_decompile(binary_path)
        else:
            return self._pattern_decompile(assembly, sanitized_indices)

    def _select_best_decompiler(self) -> str:
        # Priority: retdec > ghidra > snowman > pattern
        priority = ['retdec', 'ghidra', 'snowman', 'pattern']
        for d in priority:
            if d in self.available_decompilers:
                return d
        return 'pattern'

    def _ghidra_decompile(self, binary_path: str) -> Dict:
        """Use Ghidra's decompiler."""
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                project_dir = os.path.join(tmpdir, 'ghidra_project')
                os.makedirs(project_dir, exist_ok=True)
                
                output_file = os.path.join(tmpdir, 'decompiled.c')
                
                # Ghidra headless script for decompilation
                script_content = '''
from ghidra.app.decompiler import DecompInterface
from ghidra.util.task import ConsoleTaskMonitor

decomp = DecompInterface()
decomp.openProgram(currentProgram)

with open("{output}", "w") as f:
    fm = currentProgram.getFunctionManager()
    for func in fm.getFunctions(True):
        results = decomp.decompileFunction(func, 60, ConsoleTaskMonitor())
        if results.decompileCompleted():
            f.write("// Function: " + func.getName() + "\\n")
            f.write(results.getDecompiledFunction().getC() + "\\n\\n")
'''.format(output=output_file.replace('\\', '/'))
                
                script_path = os.path.join(tmpdir, 'decompile_script.py')
                with open(script_path, 'w') as f:
                    f.write(script_content)
                
                # Run Ghidra headless
                ghidra_headless = os.path.join(GHIDRA_PATH, 'support', 'analyzeHeadless')
                if not os.path.exists(ghidra_headless):
                    ghidra_headless = shutil.which('analyzeHeadless') or 'analyzeHeadless'
                
                cmd = [
                    ghidra_headless,
                    project_dir, 'temp_project',
                    '-import', binary_path,
                    '-postScript', script_path,
                    '-deleteProject'
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT
                )
                
                if os.path.exists(output_file):
                    with open(output_file, 'r') as f:
                        code = f.read()
                    return {
                        'code': self._postprocess_ghidra(code),
                        'decompiler_used': 'ghidra',
                        'success': True,
                        'errors': []
                    }
                else:
                    return {
                        'code': '',
                        'decompiler_used': 'ghidra',
                        'success': False,
                        'errors': [result.stderr or 'No output generated']
                    }
                    
        except subprocess.TimeoutExpired:
            return {'code': '', 'decompiler_used': 'ghidra', 'success': False, 
                    'errors': ['Timeout']}
        except Exception as e:
            return {'code': '', 'decompiler_used': 'ghidra', 'success': False,
                    'errors': [str(e)]}

    def _retdec_decompile(self, binary_path: str) -> Dict:
        """Use RetDec decompiler."""
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_file = os.path.join(tmpdir, 'output.c')
                
                retdec = RETDEC_PATH if os.path.exists(RETDEC_PATH) else 'retdec-decompiler'
                
                cmd = [
                    retdec,
                    '-o', output_file,
                    '--cleanup',
                    binary_path
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT
                )
                
                if os.path.exists(output_file):
                    with open(output_file, 'r') as f:
                        code = f.read()
                    return {
                        'code': self._postprocess_retdec(code),
                        'decompiler_used': 'retdec',
                        'success': True,
                        'errors': []
                    }
                else:
                    return {
                        'code': '',
                        'decompiler_used': 'retdec',
                        'success': False,
                        'errors': [result.stderr or 'No output generated']
                    }
                    
        except subprocess.TimeoutExpired:
            return {'code': '', 'decompiler_used': 'retdec', 'success': False,
                    'errors': ['Timeout']}
        except Exception as e:
            return {'code': '', 'decompiler_used': 'retdec', 'success': False,
                    'errors': [str(e)]}

    def _snowman_decompile(self, binary_path: str) -> Dict:
        """Use Snowman (nocode CLI) decompiler."""
        try:
            snowman = SNOWMAN_PATH if os.path.exists(SNOWMAN_PATH) else 'nocode'
            
            result = subprocess.run(
                [snowman, binary_path],
                capture_output=True,
                text=True,
                timeout=TIMEOUT
            )
            
            if result.returncode == 0 and result.stdout:
                return {
                    'code': self._postprocess_snowman(result.stdout),
                    'decompiler_used': 'snowman',
                    'success': True,
                    'errors': []
                }
            else:
                return {
                    'code': '',
                    'decompiler_used': 'snowman',
                    'success': False,
                    'errors': [result.stderr or 'Decompilation failed']
                }
                
        except subprocess.TimeoutExpired:
            return {'code': '', 'decompiler_used': 'snowman', 'success': False,
                    'errors': ['Timeout']}
        except Exception as e:
            return {'code': '', 'decompiler_used': 'snowman', 'success': False,
                    'errors': [str(e)]}

    def _pattern_decompile(
        self,
        assembly: str,
        sanitized_indices: List[int] = None
    ) -> Dict:
        """
        Pattern-based assembly to C translation.
        Used as fallback when no external decompiler is available.
        """
        if not assembly:
            return {
                'code': '// No assembly provided',
                'decompiler_used': 'pattern',
                'success': False,
                'errors': ['No input']
            }
        
        lines = assembly.strip().split('\n')
        
        # Filter to sanitized indices if provided
        if sanitized_indices:
            lines = [lines[i] for i in sanitized_indices if i < len(lines)]
        
        c_lines = ['int decompiled_function(void) {']
        var_counter = 0
        declared_vars = set()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith(';') or line.startswith('#'):
                continue
            
            c_stmt = self._translate_instruction(line, declared_vars, var_counter)
            if c_stmt:
                c_lines.append(f'    {c_stmt}')
                var_counter += 1
        
        c_lines.append('    return 0;')
        c_lines.append('}')
        
        return {
            'code': '\n'.join(c_lines),
            'decompiler_used': 'pattern',
            'success': True,
            'errors': []
        }

    def _translate_instruction(
        self,
        instr: str,
        declared_vars: set,
        var_idx: int
    ) -> Optional[str]:
        """Translate a single x86 instruction to C statement."""
        instr = instr.lower().strip()
        
        # Skip labels
        if instr.endswith(':'):
            return f'// label: {instr}'
        
        # Parse instruction
        parts = re.split(r'[\s,]+', instr)
        if not parts:
            return None
        
        mnem = parts[0]
        ops = parts[1:] if len(parts) > 1 else []
        
        # Normalize operands
        def norm_op(op):
            op = op.strip()
            # Register to variable
            if op in ['eax', 'rax', 'ax', 'al']:
                return 'result'
            if op in ['ebx', 'rbx', 'bx', 'bl']:
                return 'var_b'
            if op in ['ecx', 'rcx', 'cx', 'cl']:
                return 'var_c'
            if op in ['edx', 'rdx', 'dx', 'dl']:
                return 'var_d'
            if op in ['esi', 'rsi', 'si']:
                return 'var_src'
            if op in ['edi', 'rdi', 'di']:
                return 'var_dst'
            # Immediate
            if op.startswith('0x') or op.isdigit():
                return op
            # Memory reference
            if '[' in op:
                return f'mem[{var_idx}]'
            return op
        
        ops = [norm_op(o) for o in ops]
        
        # Declare variables
        for op in ops:
            if op.startswith('var_') or op == 'result':
                if op not in declared_vars:
                    declared_vars.add(op)
        
        # Translate by mnemonic
        if mnem == 'mov' and len(ops) >= 2:
            return f'{ops[0]} = {ops[1]};'
        elif mnem == 'add' and len(ops) >= 2:
            return f'{ops[0]} += {ops[1]};'
        elif mnem == 'sub' and len(ops) >= 2:
            return f'{ops[0]} -= {ops[1]};'
        elif mnem == 'mul' and len(ops) >= 1:
            return f'result *= {ops[0]};'
        elif mnem == 'imul' and len(ops) >= 2:
            return f'{ops[0]} *= {ops[1]};'
        elif mnem == 'div' and len(ops) >= 1:
            return f'result /= {ops[0]};'
        elif mnem == 'xor' and len(ops) >= 2:
            if ops[0] == ops[1]:
                return f'{ops[0]} = 0;'
            return f'{ops[0]} ^= {ops[1]};'
        elif mnem == 'and' and len(ops) >= 2:
            return f'{ops[0]} &= {ops[1]};'
        elif mnem == 'or' and len(ops) >= 2:
            return f'{ops[0]} |= {ops[1]};'
        elif mnem == 'shl' and len(ops) >= 2:
            return f'{ops[0]} <<= {ops[1]};'
        elif mnem == 'shr' and len(ops) >= 2:
            return f'{ops[0]} >>= {ops[1]};'
        elif mnem == 'push':
            return f'// push {ops[0] if ops else ""}'
        elif mnem == 'pop':
            return f'// pop {ops[0] if ops else ""}'
        elif mnem == 'call':
            return f'{ops[0] if ops else "func"}();'
        elif mnem == 'ret':
            return 'return result;'
        elif mnem == 'jmp':
            return f'goto {ops[0] if ops else "label"};'
        elif mnem in ['je', 'jz']:
            return f'if (zero_flag) goto {ops[0] if ops else "label"};'
        elif mnem in ['jne', 'jnz']:
            return f'if (!zero_flag) goto {ops[0] if ops else "label"};'
        elif mnem == 'cmp' and len(ops) >= 2:
            return f'zero_flag = ({ops[0]} == {ops[1]});'
        elif mnem == 'test' and len(ops) >= 2:
            return f'zero_flag = (({ops[0]} & {ops[1]}) == 0);'
        elif mnem == 'lea' and len(ops) >= 2:
            return f'{ops[0]} = &{ops[1]};'
        elif mnem == 'nop':
            return '// nop'
        elif mnem in ['inc']:
            return f'{ops[0]}++;' if ops else '// inc'
        elif mnem in ['dec']:
            return f'{ops[0]}--;' if ops else '// dec'
        elif mnem == 'neg' and ops:
            return f'{ops[0]} = -{ops[0]};'
        elif mnem == 'not' and ops:
            return f'{ops[0]} = ~{ops[0]};'
        else:
            return f'// {instr}'

    def _assemble_to_binary(
        self,
        assembly: str,
        sanitized_indices: List[int] = None
    ) -> Optional[str]:
        """Assemble sanitized assembly into a temporary binary."""
        try:
            lines = assembly.strip().split('\n')
            if sanitized_indices:
                lines = [lines[i] for i in sanitized_indices if i < len(lines)]
            
            asm_content = '\n'.join(lines)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.s', delete=False) as f:
                f.write(asm_content)
                asm_path = f.name
            
            bin_path = asm_path.replace('.s', '.o')
            
            # Try nasm first, then as
            try:
                subprocess.run(
                    ['nasm', '-f', 'elf64', '-o', bin_path, asm_path],
                    capture_output=True, timeout=30, check=True
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                subprocess.run(
                    ['as', '-o', bin_path, asm_path],
                    capture_output=True, timeout=30, check=True
                )
            
            os.unlink(asm_path)
            
            if os.path.exists(bin_path):
                return bin_path
            return None
            
        except Exception as e:
            print(f"Assembly failed: {e}")
            return None

    def _postprocess_ghidra(self, code: str) -> str:
        """Clean up Ghidra decompiler output."""
        # Remove Ghidra-specific annotations
        code = re.sub(r'/\* WARNING:.*?\*/', '', code, flags=re.DOTALL)
        code = re.sub(r'undefined\d*', 'int', code)
        code = re.sub(r'DAT_[0-9a-fA-F]+', 'global_var', code)
        code = re.sub(r'FUN_[0-9a-fA-F]+', 'func', code)
        return code.strip()

    def _postprocess_retdec(self, code: str) -> str:
        """Clean up RetDec output."""
        # Remove RetDec metadata comments
        lines = code.split('\n')
        cleaned = [l for l in lines if not l.strip().startswith('//')]
        return '\n'.join(cleaned).strip()

    def _postprocess_snowman(self, code: str) -> str:
        """Clean up Snowman output."""
        return code.strip()


# Global decompiler instance
decompiler = ExternalDecompiler()


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'available_decompilers': decompiler.available_decompilers
    })


@app.route('/decompile', methods=['POST'])
def decompile():
    """
    Decompile binary or assembly to C code.
    
    Request JSON:
        binary_path: Path to binary file (optional)
        assembly: Raw assembly text (optional)
        sanitized_indices: List of instruction indices to keep (from GNN)
        prefer: Preferred decompiler ('ghidra', 'retdec', 'snowman', 'auto')
    
    Response JSON:
        code: Decompiled C code
        decompiler_used: Which decompiler was used
        success: Boolean success flag
        errors: List of error messages
    """
    data = request.get_json(force=True)
    
    binary_path = data.get('binary_path')
    assembly = data.get('assembly')
    sanitized_indices = data.get('sanitized_indices')
    prefer = data.get('prefer', 'auto')
    
    if not binary_path and not assembly:
        return jsonify({
            'code': '',
            'decompiler_used': None,
            'success': False,
            'errors': ['Either binary_path or assembly required']
        }), 400
    
    result = decompiler.decompile(
        binary_path=binary_path,
        assembly=assembly,
        sanitized_indices=sanitized_indices,
        prefer=prefer
    )
    
    return jsonify(result)


@app.route('/decompile-function', methods=['POST'])
def decompile_function():
    """
    Decompile a single function from sanitized P-Code.
    
    Request JSON:
        pcode: List of P-Code operations
        sanitized_mask: Boolean mask of which instructions to keep
        function_name: Name of the function
    """
    data = request.get_json(force=True)
    
    pcode = data.get('pcode', [])
    sanitized_mask = data.get('sanitized_mask', [True] * len(pcode))
    func_name = data.get('function_name', 'decompiled_func')
    
    # Filter P-Code by mask
    sanitized_pcode = [
        op for op, keep in zip(pcode, sanitized_mask) if keep
    ]
    
    # Convert P-Code to assembly-like representation
    assembly_lines = []
    for op in sanitized_pcode:
        mnem = op.get('mnemonic', 'NOP')
        inputs = op.get('inputs', [])
        output = op.get('output')
        
        input_strs = [str(i.get('value', i.get('name', '?'))) for i in inputs]
        output_str = str(output.get('value', output.get('name', 'out'))) if output else ''
        
        line = f"{mnem} {output_str}, {', '.join(input_strs)}"
        assembly_lines.append(line)
    
    assembly = '\n'.join(assembly_lines)
    
    # Decompile
    result = decompiler._pattern_decompile(assembly, None)
    
    # Replace function name
    result['code'] = result['code'].replace('decompiled_function', func_name)
    result['function_name'] = func_name
    result['original_instruction_count'] = len(pcode)
    result['sanitized_instruction_count'] = len(sanitized_pcode)
    
    return jsonify(result)


if __name__ == '__main__':
    port = int(os.getenv('DECOMPILER_SERVICE_PORT', '5009'))
    app.run(host='0.0.0.0', port=port, debug=False)
