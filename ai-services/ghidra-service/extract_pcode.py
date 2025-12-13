# Ghidra script to extract P-Code and CFG
# This script is run by Ghidra's headless analyzer

from ghidra.program.model.pcode import PcodeOp
from ghidra.app.decompiler import DecompInterface
from ghidra.util.task import ConsoleTaskMonitor
import json

def extract_pcode_from_function(func):
    """Extract P-Code operations from a function."""
    pcode_ops = []
    
    # Initialize decompiler
    decompiler = DecompInterface()
    decompiler.openProgram(currentProgram)
    
    # Decompile function
    results = decompiler.decompileFunction(func, 30, ConsoleTaskMonitor())
    
    if results and results.decompileCompleted():
        high_func = results.getHighFunction()
        
        # Iterate through basic blocks
        for basic_block in high_func.getBasicBlocks():
            iterator = basic_block.getIterator()
            
            while iterator.hasNext():
                pcode_op = iterator.next()
                
                op_info = {
                    'opcode': pcode_op.getOpcode(),
                    'mnemonic': pcode_op.getMnemonic(),
                    'inputs': [],
                    'output': None,
                    'address': str(pcode_op.getSeqnum().getTarget())
                }
                
                # Extract inputs
                for i in range(pcode_op.getNumInputs()):
                    varnode = pcode_op.getInput(i)
                    op_info['inputs'].append({
                        'offset': varnode.getOffset(),
                        'size': varnode.getSize()
                    })
                
                # Extract output
                if pcode_op.getOutput():
                    output = pcode_op.getOutput()
                    op_info['output'] = {
                        'offset': output.getOffset(),
                        'size': output.getSize()
                    }
                
                pcode_ops.append(op_info)
    
    return pcode_ops

def extract_cfg(func):
    """Extract control flow graph from a function."""
    cfg = {
        'nodes': [],
        'edges': []
    }
    
    # Get function body
    body = func.getBody()
    
    # Build basic block graph
    block_model = ghidra.program.model.block.BasicBlockModel(currentProgram)
    blocks = block_model.getCodeBlocksContaining(body, ConsoleTaskMonitor())
    
    node_id = 0
    block_map = {}
    
    while blocks.hasNext():
        block = blocks.next()
        
        node_info = {
            'id': node_id,
            'address': str(block.getFirstStartAddress()),
            'num_instructions': block.getNumAddresses()
        }
        
        cfg['nodes'].append(node_info)
        block_map[str(block.getFirstStartAddress())] = node_id
        node_id += 1
        
        # Extract edges
        destinations = block.getDestinations(ConsoleTaskMonitor())
        while destinations.hasNext():
            dest = destinations.next()
            dest_addr = str(dest.getDestinationAddress())
            
            # Edge will be added after all nodes are processed
            cfg['edges'].append({
                'from': str(block.getFirstStartAddress()),
                'to': dest_addr,
                'flow_type': str(dest.getFlowType())
            })
    
    # Map edge addresses to node IDs
    for edge in cfg['edges']:
        edge['from'] = block_map.get(edge['from'], -1)
        edge['to'] = block_map.get(edge['to'], -1)
    
    return cfg

def run():
    """Main script entry point."""
    
    analysis_results = {
        'program_name': currentProgram.getName(),
        'functions': []
    }
    
    # Get function manager
    func_manager = currentProgram.getFunctionManager()
    
    # Iterate through all functions
    for func in func_manager.getFunctions(True):
        func_info = {
            'name': func.getName(),
            'entry_point': str(func.getEntryPoint()),
            'pcode': extract_pcode_from_function(func),
            'cfg': extract_cfg(func)
        }
        
        analysis_results['functions'].append(func_info)
    
    # Write results to file
    output_path = '/tmp/ghidra_analysis_output.json'
    with open(output_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print("Analysis complete. Results written to: " + output_path)

# Run the script
run()
