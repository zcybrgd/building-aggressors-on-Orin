import re
from dataclasses import dataclass

@dataclass
class PTXInstruction:
    opcode: str           # ld, st, mad, add, ...
    operand_types: tuple  # (reg, mem, imm, ...)
    memory_space: str     # .global, .shared, .local
    data_type: str        # .f32, .s32, .u64, ...
    predicated: bool      # If statement?
    
    def to_token(self):
        """Convert to vocabulary token"""
        token_parts = [
            f"OP_{self.opcode}",
            f"MEM_{self.memory_space}" if self.memory_space else "COMPUTE",
            f"TYPE_{self.data_type}",
            "PRED" if self.predicated else "UNPRED"
        ]
        return "_".join(token_parts)

class PTXTokenizer:
    def __init__(self):
        self.vocab = self._build_vocab()
        self.instruction_patterns = {
            'memory_load': r'ld\.(.*?)\s+(.*?),\s*\[(.*?)\]',
            'memory_store': r'st\.(.*?)\s+\[(.*?)\],\s*(.*?)',
            'compute': r'(mad|add|mul|div|fma)\.(.*?)\s+%r\d+',
            'branch': r'@(%p\d+)\s+(bra|call)',
        }
    
    def tokenize_ptx(self, ptx_code: str):
        """Convert PTX assembly → token sequence"""
        tokens = []
        lines = ptx_code.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            
            instr = self._parse_instruction(line)
            tokens.append(instr.to_token())
        
        return tokens
    
    def _parse_instruction(self, line: str) -> PTXInstruction:
        # Parse each instruction type
        for pattern_name, pattern in self.instruction_patterns.items():
            match = re.match(pattern, line)
            if match:
                return self._build_instruction(pattern_name, match)
        
        # Default: treat as compute
        return PTXInstruction(
            opcode='generic',
            operand_types=tuple(),
            memory_space=None,
            data_type='unknown',
            predicated='@' in line
        )
    
    def _build_vocab(self):
        """Create token vocabulary"""
        opcodes = ['ld', 'st', 'mad', 'add', 'mul', 'div', 'fma', 'bra', 'call']
        mem_spaces = ['.global', '.shared', '.local', '.const']
        data_types = ['.f32', '.s32', '.u64', '.u32', '.f64']
        
        vocab = {}
        token_id = 0
        
        for opcode in opcodes:
            for mem_space in ['compute'] + mem_spaces:
                for dtype in data_types:
                    token = f"OP_{opcode}_MEM_{mem_space}_TYPE_{dtype}"
                    vocab[token] = token_id
                    token_id += 1
        
        return vocab
    
    def encode_tokens(self, tokens: list) -> list:
        """Convert token strings → integer IDs"""
        return [self.vocab.get(t, self.vocab['<UNK>']) for t in tokens]

# Usage
tokenizer = PTXTokenizer()
ptx_code = """
    ld.global.f32    %f1, [%rd1];
    mad.f32          %f2, %f1, %f0, %f2;
    st.global.f32    [%rd2], %f2;
"""
tokens = tokenizer.tokenize_ptx(ptx_code)
token_ids = tokenizer.encode_tokens(tokens)