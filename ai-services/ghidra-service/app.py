from flask import Flask, request, jsonify
import subprocess
import json
import os
from pathlib import Path

app = Flask(__name__)

GHIDRA_INSTALL_PATH = os.getenv('GHIDRA_INSTALL_PATH', '/opt/ghidra')
SCRIPT_PATH = Path(__file__).parent / 'extract_pcode.py'

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'ghidra-analyzer'})

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze a binary file and extract P-Code, CFG, and function information.
    
    Request body:
    {
        "file_path": "/path/to/binary",
        "project_name": "analysis_project"
    }
    
    Returns:
    {
        "pcode": [...],
        "cfg": {...},
        "functions": [...]
    }
    """
    try:
        data = request.json
        file_path = data.get('file_path')
        project_name = data.get('project_name', 'temp_project')
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': 'Invalid file_path'}), 400
        
        # Create temp project directory
        project_dir = f'/tmp/ghidra_projects/{project_name}'
        os.makedirs(project_dir, exist_ok=True)
        
        # Run Ghidra headless analyzer
        cmd = [
            f'{GHIDRA_INSTALL_PATH}/support/analyzeHeadless',
            project_dir,
            project_name,
            '-import', file_path,
            '-postScript', str(SCRIPT_PATH),
            '-scriptPath', str(SCRIPT_PATH.parent)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 min timeout
        )
        
        # Parse output
        output_file = Path(project_dir) / f'{Path(file_path).stem}_analysis.json'
        
        if output_file.exists():
            with open(output_file) as f:
                analysis_data = json.load(f)
            return jsonify(analysis_data)
        else:
            # Fallback to parsing stdout
            return jsonify({
                'pcode': [],
                'cfg': {},
                'functions': [],
                'stdout': result.stdout,
                'stderr': result.stderr
            })
            
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Analysis timeout'}), 504
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
