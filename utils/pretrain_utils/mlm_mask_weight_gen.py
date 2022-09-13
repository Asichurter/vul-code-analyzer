from typing import List, Tuple

import torch
from allennlp.data.tokenizers import Token
from pygments.lexers.c_cpp import CppLexer

from utils.pretrain_utils.lexer_based_token_analyse_utils import get_filtered_token_span_by_cpplexer

cpp_lexer = CppLexer()

def uniform_mlm_gen_mask_weights(raw_code: str, tokens: List[Token]) -> torch.Tensor:
    """
    Equal weights for all tokens.
    """
    return torch.ones(len(tokens),)


def intersect_allennlp_and_filtered_lexer_tokens(raw_code: str,
                                                 allennlp_tokens: List[Token],
                                                 filtered_types: List[str]):
    """
    Given raw code and allennlp tokenized tokens, this function first analyze
    raw code using lexer to get token types, and filter them based on given
    filtered_types.

    Then unfiltered tokens will be intersected with allennlp tokens to determine
    exactly which allennlp tokens are unfiltered based on resulted spans of lexer
    analysis.

    Finally, indices of these intersected allennlp tokens will be returned.
    """
    target_token_char_spans = get_filtered_token_span_by_cpplexer(raw_code, filtered_types)
    span_i = 0
    allennlp_target_token_indices = []
    for token_i, token in enumerate(allennlp_tokens):
        idx, idx_end = token.idx, token.idx_end

        # Skip speicial tokens
        if idx is None or idx_end is None:
            continue
        # No more spans to check, break
        if span_i >= len(target_token_char_spans):
            break

        cur_span = target_token_char_spans[span_i]
        # Check span intersection
        if (idx - cur_span[1]) * (idx_end - cur_span[0]) < 0:
            allennlp_target_token_indices.append(token_i)
        # Check if current token span ends and move to next span
        if idx_end >= cur_span[1]:
            span_i += 1
    return allennlp_target_token_indices

def basic_lexer_filter_mlm_gen_mask_weights(raw_code: str, tokens: List[Token]) -> Tuple[torch.Tensor, List]:
    """
    Filter whitespaces, operators, punctuations and literals.
    """
    filtered_types = ['Token.Punctuation', 'Token.Text.Whitespace',
                      'Token.Literal.String', 'Token.Operator', 'Token.Literal.Number.Integer']
    unfiltered_indices = intersect_allennlp_and_filtered_lexer_tokens(raw_code, tokens, filtered_types)
    weights = torch.zeros(len(tokens),)
    weights[unfiltered_indices] += 1
    return weights, unfiltered_indices

from utils import GlobalLogger as mylogger

mlm_weight_gen_dispatcher = {
    'uniform': uniform_mlm_gen_mask_weights,
    'basic_lexer_filter': basic_lexer_filter_mlm_gen_mask_weights,

    'none': uniform_mlm_gen_mask_weights,
    None: uniform_mlm_gen_mask_weights,
}

def dispatch_mlm_weight_gen_method(method: str = 'uniform'):
    if method not in mlm_weight_gen_dispatcher:
        mylogger.warning('mlm_weight_gen_dispatch',
                         f'No such method when dispatching mlm weight gen: {method}')
    return mlm_weight_gen_dispatcher.get(method, uniform_mlm_gen_mask_weights)


if __name__ == "__main__":
    from allennlp.data.tokenizers import PretrainedTransformerTokenizer

    tokenizer = PretrainedTransformerTokenizer('microsoft/codebert-base')
    code = 'uv_nmi_dump_state(int cpu, struct pt_regs *regs, int master)\n{\n if (master) {\n int tcpu;\n int ignored = 0;\n int saved_console_loglevel = console_loglevel;\n pr_alert("UV: tracing %s for %d CPUs from CPU %d ",\n uv_nmi_action_is("ips") ? "IPs" : "processes",\n atomic_read(&uv_nmi_cpus_in_nmi), cpu);\n console_loglevel = uv_nmi_loglevel;\n atomic_set(&uv_nmi_slave_continue, SLAVE_EXIT);\n for_each_online_cpu(tcpu) {\n if (cpumask_test_cpu(tcpu, uv_nmi_cpu_mask))\n ignored++;\n else if (tcpu == cpu)\n uv_nmi_dump_state_cpu(tcpu, regs);\n else\n uv_nmi_trigger_dump(tcpu);\n }\n if (ignored)\n pr_alert("UV: %d CPUs ignored NMI ", ignored);\n console_loglevel = saved_console_loglevel;\n pr_alert("UV: process trace complete ");\n } else {\n while (!atomic_read(&uv_nmi_slave_continue))\n cpu_relax();\n while (this_cpu_read(uv_cpu_nmi.state) != UV_NMI_STATE_DUMP)\n cpu_relax();\n uv_nmi_dump_state_cpu(cpu, regs);\n }\n uv_nmi_sync_exit(master);\n}'
    allennlp_tokens = tokenizer.tokenize(code)
    weights, indices = basic_lexer_filter_mlm_gen_mask_weights(code, allennlp_tokens)
    a = 0