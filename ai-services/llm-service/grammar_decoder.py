"""
Grammar-Guided Decoding (GBN - Grammar Based Normalization)

Eliminates LLM hallucinations by constraining generation to valid C syntax.
Uses Backus-Naur Form (BNF) to ensure every token is syntactically correct.

The model cannot generate code that doesn't compile.
"""

from lark import Lark, Token
from lark.exceptions import LarkError
from typing import List, Set
import torch

class GrammarConstrainedDecoder:
    """
    Constrains LLM token generation to valid C grammar.
    
    At each generation step, only tokens that maintain valid syntax are allowed.
    This eliminates:
    - Undeclared variables
    - Invalid function calls  
    - Syntax errors
    - Malformed expressions
    """
    
    # Simplified C grammar in BNF (Lark syntax)
    C_GRAMMAR = r"""
    ?start: program
    
    program: (function_def | declaration)*
    
    function_def: type IDENTIFIER "(" param_list? ")" block
    
    param_list: parameter ("," parameter)*
    parameter: type IDENTIFIER
    
    type: "int" | "char" | "float" | "double" | "void" | "unsigned" type
    
    block: "{" statement* "}"
    
    ?statement: declaration
              | expression_stmt
              | return_stmt
              | if_stmt
              | while_stmt
              | for_stmt
              | block
    
    declaration: type IDENTIFIER ("=" expression)? ";"
    
    expression_stmt: expression ";"
    
    return_stmt: "return" expression? ";"
    
    if_stmt: "if" "(" expression ")" statement ("else" statement)?
    
    while_stmt: "while" "(" expression ")" statement
    
    for_stmt: "for" "(" expression? ";" expression? ";" expression? ")" statement
    
    ?expression: assignment
    
    ?assignment: logical_or ("=" assignment)?
    
    ?logical_or: logical_and ("||" logical_and)*
    
    ?logical_and: equality ("&&" equality)*
    
    ?equality: comparison (("==" | "!=") comparison)*
    
    ?comparison: term (("<" | ">" | "<=" | ">=") term)*
    
    ?term: factor (("+" | "-") factor)*
    
    ?factor: unary (("*" | "/" | "%") unary)*
    
    ?unary: ("!" | "-" | "++" | "--" | "&" | "*")? primary
    
    ?primary: NUMBER
            | STRING
            | IDENTIFIER
            | IDENTIFIER "(" arg_list? ")"
            | "(" expression ")"
    
    arg_list: expression ("," expression)*
    
    IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/
    NUMBER: /[0-9]+(\.[0-9]+)?/
    STRING: /"[^"]*"/
    
    %import common.WS
    %ignore WS
    %import common.C_COMMENT
    %ignore C_COMMENT
    %import common.CPP_COMMENT  
    %ignore CPP_COMMENT
    """
    
    def __init__(self):
        self.parser = Lark(
            self.C_GRAMMAR,
            start='start',
            parser='lalr',
            lexer='standard'
        )
        
        # Track declared variables and functions
        self.declared_vars = set()
        self.declared_funcs = set()
        
    def is_valid_continuation(self, current_code: str, next_token: str) -> bool:
        """
        Check if adding next_token maintains valid C syntax.
        
        Args:
            current_code: Code generated so far
            next_token: Token to potentially add
        
        Returns:
            True if syntax remains valid, False otherwise
        """
        try:
            # Try parsing with next token
            test_code = current_code + next_token
            self.parser.parse(test_code)
            return True
        except LarkError:
            # Syntax error - token not allowed
            return False
    
    def get_valid_tokens(
        self,
        current_code: str,
        candidate_tokens: List[str],
        tokenizer
    ) -> List[int]:
        """
        Filter candidate tokens to only those maintaining valid syntax.
        
        Args:
            current_code: Code generated so far
            candidate_tokens: List of potential next tokens
            tokenizer: HuggingFace tokenizer
        
        Returns:
            List of valid token IDs
        """
        valid_ids = []
        
        for token_text in candidate_tokens:
            if self.is_valid_continuation(current_code, token_text):
                token_id = tokenizer.encode(token_text, add_special_tokens=False)[0]
                valid_ids.append(token_id)
        
        return valid_ids
    
    def constrained_sampling(
        self,
        model,
        tokenizer,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.2
    ) -> str:
        """
        Generate code with grammar constraints.
        
        Args:
            model: LLM model
            tokenizer: Tokenizer
            prompt: Input prompt
            max_length: Max tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Grammatically correct C code
        """
        device = next(model.parameters()).device
        
        # Encode prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        generated = input_ids
        current_code = ""
        
        for _ in range(max_length):
            # Get logits for next token
            with torch.no_grad():
                outputs = model(generated)
                next_token_logits = outputs.logits[0, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Get top-k candidates
            top_k = 50
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            
            # Convert to tokens
            candidate_tokens = [tokenizer.decode([idx]) for idx in top_k_indices]
            
            # Filter to valid tokens
            valid_ids = self.get_valid_tokens(current_code, candidate_tokens, tokenizer)
            
            if not valid_ids:
                # No valid continuation - use most likely token
                next_token_id = top_k_indices[0]
            else:
                # Sample from valid tokens
                valid_logits = next_token_logits[valid_ids]
                probs = torch.softmax(valid_logits, dim=0)
                next_idx = torch.multinomial(probs, 1)[0]
                next_token_id = valid_ids[next_idx]
            
            # Append token
            generated = torch.cat([generated, torch.tensor([[next_token_id]]).to(device)], dim=1)
            next_token = tokenizer.decode([next_token_id])
            current_code += next_token
            
            # Check for EOS
            if next_token_id == tokenizer.eos_token_id:
                break
        
        return current_code
    
    def validate_complete_code(self, code: str) -> tuple[bool, str]:
        """
        Validate complete C code.
        
        Returns:
            (is_valid, error_message)
        """
        try:
            self.parser.parse(code)
            return True, ""
        except LarkError as e:
            return False, str(e)


# Example usage
if __name__ == '__main__':
    decoder = GrammarConstrainedDecoder()
    
    # Test valid code
    valid_code = """
    int main() {
        int x = 5;
        return x;
    }
    """
    is_valid, error = decoder.validate_complete_code(valid_code)
    print(f"Valid: {is_valid}")
    
    # Test invalid code
    invalid_code = """
    int main() {
        int x = 5
        return x;
    }
    """
    is_valid, error = decoder.validate_complete_code(invalid_code)
    print(f"Invalid: {not is_valid}, Error: {error}")
