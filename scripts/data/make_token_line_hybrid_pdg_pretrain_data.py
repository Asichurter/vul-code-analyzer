import os
from typing import List, Tuple, Iterable
from tqdm import tqdm

from allennlp.data.tokenizers import PretrainedTransformerTokenizer

from utils.file import load_pickle, dump_pickle, load_text
from utils.joern_utils.joern_dev_pdg_parse_utils import build_token_level_pdg_struct, build_line_level_pdg_struct
from utils.joern_utils.joern_dev_parse import convert_func_signature_to_one_line
from utils.allennlp_utils.tokenizer_vocab_sensitive_utils import pre_handle_special_tokenizer_tokens, post_handle_special_tokenizer_tokens

def extract_line_ctrl_dependencies(line_edges: Iterable[str]) -> List[Tuple[int,int]]:
    line_ctrl_edges = []
    for edge in line_edges:
        seline, line_type = edge.split()
        if line_type in ['2', '3']:
            sline, eline = seline.split(',')
            line_ctrl_edges.append((int(sline), int(eline)))
    return line_ctrl_edges

def parse_dependency_strs(edges: Iterable[str]) -> List[Tuple[int, int]]:
    parsed_edges = []
    for edge in edges:
        sidx, eidx = edge.split()
        parsed_edges.append((int(sidx), int(eidx)))
    return parsed_edges

def extract_data_id(file_path: str) -> str:
    file_name = file_path.split('/')[-1]
    data_id = file_name.split('.')[0]
    return data_id

# def read_raw_code_file(vol, file_id):
#     cpp_file_path = os.path.join(raw_code_base_vol_path, f'vol{vol}', f'{file_id}.cpp')
#     c_file_path = os.path.join(raw_code_base_vol_path, f'vol{vol}', f'{file_id}.c')
#
#     for file_path in [cpp_file_path, c_file_path]:
#         if not os.path.exists(file_path):
#             continue
#         # Adapt joern-parse format
#         convert_func_signature_to_one_line(file_path, redump=True)
#         raw_code = load_text(file_path)
#         return raw_code
#
#     raise FileExistsError(f'Vol {vol}, File {file_id}')


def generate_data_from_scratch():
    raw_code_base_vol_path = '/data1/zhijietang/vul_data/datasets/docker/cppfiles/'
    line_level_data_path_temp = '/data1/zhijietang/vul_data/datasets/joern_vulberta/packed_vol_data/packed_vol_{}.pkl'
    token_level_data_path_temp = '/data1/zhijietang/vul_data/datasets/docker/joern_dev_analysis_results/joern_parsed_raw_vol{}.pkl'
    tgt_dump_base_path_temp = '/data1/zhijietang/vul_data/datasets/joern_vulberta/packed_process_hybrid_data/packed_hybrid_vol_{}.pkl'
    # tgt_dump_base_path_temp = '/data1/zhijietang/vul_data/datasets/joern_vulberta/packed_process_hybrid_data_allvs/packed_hybrid_vol_{}.pkl'

    multi_vs_multi_strategy = 'first'

    tokenizer_name = 'microsoft/codebert-base'
    tokenizer = PretrainedTransformerTokenizer(tokenizer_name)
    vols = list(range(0,229))

    for vol in vols:
        line_vol_data_path = line_level_data_path_temp.format(vol)
        token_vol_data_path = token_level_data_path_temp.format(vol)
        line_vol_data = load_pickle(line_vol_data_path)
        token_vol_data = load_pickle(token_vol_data_path)

        vol_hybrid_data = []
        line_vol_data_id_map = {o['id']: o for o in line_vol_data}
        for token_data in tqdm(token_vol_data):
            data_id = extract_data_id(token_data['file_path'])
            if data_id not in line_vol_data_id_map:
                print(f'[Warn] Missing line-level item for vol {vol}, id {data_id}')
                continue

            line_data = line_vol_data_id_map[data_id]
            raw_code = convert_func_signature_to_one_line(code=line_data['code'], redump=False)
            tokens = tokenizer.tokenize(raw_code)
            _, token_data_edges = build_token_level_pdg_struct(raw_code, tokens,
                                                               token_data['nodes'], token_data['edges'],
                                                               multi_vs_multi_strategy=multi_vs_multi_strategy,
                                                               to_build_token_ctrl_edges=False)
            pdg_data_edges = parse_dependency_strs(token_data_edges)
            line_ctrl_edges, _ = build_line_level_pdg_struct(token_data['nodes'], token_data['edges'])
            pdg_line_edges = parse_dependency_strs(line_ctrl_edges)
            # pdg_ctrl_edges = extract_line_ctrl_dependencies(line_data['edges'])
            # raw_code = read_raw_code_file(vol, data_id)
            hybrid_line_token_data = {
                'line_edges': pdg_line_edges,           # PDG line-level ctrl edges
                # 'line_edges': line_data['edges'],     # Line edges contain both data and ctrl edges
                # 'token_nodes': token_data['nodes'],   # Token nodes are input of "build_token_level_pdg_struct"
                # 'token_edges': token_data['edges'],   # Token edges are input of "build_token_level_pdg_struct"
                'id': data_id,
                # 'raw_code': line_data['code'],        # Really raw code, no space trimming
                'raw_code': raw_code,                   # Signature-converted is indispensable
                'processed_token_data_edges': {         # Dump processed token data edges
                    tokenizer_name: pdg_data_edges      # Since it is sensitive to tokenizer, we have to highlight the tokenizer name here
                }

            }
            vol_hybrid_data.append(hybrid_line_token_data)

        print(f'Vol. {vol} ({len(vol_hybrid_data)} items) saved to {tgt_dump_base_path_temp.format(vol)}')
        dump_pickle(vol_hybrid_data, tgt_dump_base_path_temp.format(vol))


def update_token_data_tokenizer(check_overwrite=True):
    raw_code_base_vol_path = '/data1/zhijietang/vul_data/datasets/docker/cppfiles/'
    line_level_data_path_temp = '/data1/zhijietang/vul_data/datasets/joern_vulberta/packed_vol_data/packed_vol_{}.pkl'
    token_level_data_path_temp = '/data1/zhijietang/vul_data/datasets/docker/joern_dev_analysis_results/joern_parsed_raw_vol{}.pkl'
    tgt_dump_base_path_temp = '/data1/zhijietang/vul_data/datasets/joern_vulberta/packed_process_hybrid_data/packed_hybrid_vol_{}.pkl'
    # tgt_dump_base_path_temp = '/data1/zhijietang/vul_data/datasets/joern_vulberta/packed_process_hybrid_data_allvs/packed_hybrid_vol_{}.pkl'

    multi_vs_multi_strategy = 'first'

    tokenizer_name = 'microsoft/unixcoder-base-nine'
    tokenizer_type = 'unixcoder_base_nine'
    tokenizer = PretrainedTransformerTokenizer(tokenizer_name)
    vols = list(range(200,206))

    for vol in vols:
        line_vol_data_path = line_level_data_path_temp.format(vol)
        token_vol_data_path = token_level_data_path_temp.format(vol)
        tgt_vol_data_path = tgt_dump_base_path_temp.format(vol)
        line_vol_data = load_pickle(line_vol_data_path)
        token_vol_data = load_pickle(token_vol_data_path)
        tgt_vol_data = load_pickle(tgt_vol_data_path)

        line_vol_data_id_map = {o['id']: o for o in line_vol_data}
        id_to_new_token_data_edges_idx_map = {}
        new_token_data_edges = []

        for token_data in tqdm(token_vol_data):
            data_id = extract_data_id(token_data['file_path'])
            if data_id not in line_vol_data_id_map:
                print(f'[Warn] Missing line-level item for vol {vol}, id {data_id}')
                continue

            line_data = line_vol_data_id_map[data_id]
            raw_code = convert_func_signature_to_one_line(code=line_data['code'], redump=False)
            tokens = tokenizer.tokenize(raw_code)
            # BugFix: Since we will do some special token preprocesses on input tokens,
            #         we need to simulate these during preprocessing to accurately analyze the token intersections.
            # ------------------------------------------------------------------------------------------------
            tokens = pre_handle_special_tokenizer_tokens(tokenizer_type, tokens)
            tokens, _ = post_handle_special_tokenizer_tokens(tokenizer_type, (tokens,), None, '<encoder-only>') # mode is only for placeholder
            # ------------------------------------------------------------------------------------------------
            _, token_data_edges = build_token_level_pdg_struct(raw_code, tokens,
                                                               token_data['nodes'], token_data['edges'],
                                                               multi_vs_multi_strategy=multi_vs_multi_strategy,
                                                               to_build_token_ctrl_edges=False)
            pdg_data_edges = parse_dependency_strs(token_data_edges)
            id_to_new_token_data_edges_idx_map[data_id] = len(new_token_data_edges)
            new_token_data_edges.append(pdg_data_edges)

        for tgt_data in tgt_vol_data:
            data_id = tgt_data['id']
            new_token_data_edge = new_token_data_edges[id_to_new_token_data_edges_idx_map[data_id]]

            if check_overwrite and tokenizer_name in tgt_data['processed_token_data_edges']:
                assert False, f'{new_token_data_edge} has existed in processed_token_data_edges!'
            tgt_data['processed_token_data_edges'][tokenizer_name] = new_token_data_edge

        print(f'Vol. {vol} ({len(tgt_vol_data)} items) saved to {tgt_dump_base_path_temp.format(vol)}')
        dump_pickle(tgt_vol_data, tgt_dump_base_path_temp.format(vol))


if __name__ == '__main__':
    # generate_data_from_scratch()
    update_token_data_tokenizer(False)