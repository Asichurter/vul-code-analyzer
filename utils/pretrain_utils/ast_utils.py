from typing import List
from allennlp.data.tokenizers import Token, Tokenizer
import random

from tree_sitter import Node as ASTNode
from utils.data_utils.changed_func_extraction import parse_tree
from utils import GlobalLogger as mylogger

def decode_bytes(bytes_):
    return bytes_.decode('utf-8')

def is_terminal_node(node: ASTNode):
    return len(node.children) == 0

def make_unixcoder_internal_node_repr(node: ASTNode, is_left: bool):
    text = f"AST#{node.type}#{'Left' if is_left else 'Right'}"
    return text

def unixcoder_ast_serialization_tokenization(node: ASTNode, tokenizer: Tokenizer, non_terminal_dropout: float) -> List[Token]:
    if is_terminal_node(node):
        return tokenizer.tokenize(decode_bytes(node.text))
    else:
        serial_tokens = []
        drop_prob = random.random()
        # Here we use one sampled prob to control both left and right side of the dropout of non-terminal nodes
        if drop_prob > non_terminal_dropout:
            serial_tokens.append(Token(make_unixcoder_internal_node_repr(node, is_left=True)))
        for n in node.children:
            # Recur
            child_ret = unixcoder_ast_serialization_tokenization(n, tokenizer, non_terminal_dropout)
            serial_tokens.extend(child_ret)
        if drop_prob > non_terminal_dropout:
            serial_tokens.append(Token(make_unixcoder_internal_node_repr(node, is_left=False)))
        return serial_tokens

ast_serial_tokenize_dispatcher = {
    'unixcoder': unixcoder_ast_serialization_tokenization,

    None: None
}

def dispatch_ast_serial_tokenize_method(method_name: str):
    if method_name not in ast_serial_tokenize_dispatcher:
        mylogger.error('dispatch_ast_serial_tokenize_method',
                     f"Method name '{method_name}' not in accepted list.")
        raise ValueError
    return ast_serial_tokenize_dispatcher[method_name]

if __name__ == '__main__':
    from allennlp.data.tokenizers import PretrainedTransformerTokenizer

    code = 'lpss_dma_filter(struct dma_chan *chan, void *param)\n{\n\tstruct dw_dma_slave *dws = param;\n\tif (dws->dma_dev != chan->device->dev)\n\t\treturn false;\n\tchan->private = dws;\n\treturn true;\n}'
    tokenizer = PretrainedTransformerTokenizer('microsoft/unixcoder-base', add_special_tokens=False)
    tree = parse_tree(code)
    code_serial = unixcoder_ast_serialization_tokenization(tree.root_node, tokenizer, 1)
    # code_serial_join = ' '.join(code_serial)
    # tokens = tokenizer.tokenize(code_serial_join)
