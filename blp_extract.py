#!/usr/bin/env python3
"""
Extract Python code from LaTeX verbatim blocks and create standalone Python file.
"""

import re

def extract_code_blocks(latex_file):
    """Extract all code from verbatim blocks with question context."""
    with open(latex_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all verbatim blocks
    pattern = r'\\begin\{verbatim\}(.*?)\\end\{verbatim\}'
    blocks = []
    
    for match in re.finditer(pattern, content, re.DOTALL):
        code = match.group(1)
        start_pos = match.start()
        
        # Find question number before this block
        before_text = content[:start_pos]
        q_matches = list(re.finditer(r'\\item\[(\d+)\.\]', before_text))
        q_num = q_matches[-1].group(1) if q_matches else None
        
        blocks.append({
            'question': q_num,
            'code': code,
            'position': start_pos
        })
    
    return blocks

def write_python_file(blocks, output_file):
    """Write extracted code blocks to Python file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        # Header
        f.write('#!/usr/bin/env python3\n')
        f.write('"""\n')
        f.write('BLP Homework - Complete Python Code\n')
        f.write('Economics 600a, Fall 2025\n')
        f.write('Marek Chadim (marek.chadim@yale.edu)\n\n')
        f.write('Code extracted from LaTeX homework document.\n')
        f.write('"""\n\n')
        
        # Imports
        f.write('import numpy as np\n')
        f.write('import pandas as pd\n')
        f.write('import scipy.optimize as opt\n')
        f.write('import pyblp\n')
        f.write('import warnings\n')
        f.write('warnings.filterwarnings("ignore")\n\n')
        
        # Write blocks
        current_q = None
        for block in blocks:
            code = block['code'].strip()
            if not code:
                continue
                
            q = block['question']
            if q and q != current_q:
                f.write(f'\n{"#"*80}\n')
                f.write(f'# QUESTION {q}\n')
                f.write(f'{"#"*80}\n\n')
                current_q = q
            
            f.write(code + '\n\n')

if __name__ == '__main__':
    blocks = extract_code_blocks('BLP_hw_chadim.tex')
    write_python_file(blocks, 'BLP_homework_code.py')
    print(f'✓ Created BLP_homework_code.py with {len(blocks)} code blocks')
    print(f'  Questions covered: {sorted(set(b["question"] for b in blocks if b["question"]))}')

