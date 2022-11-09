from utils.file import dump_text
import subprocess

_gcc_comment_remove_cmd = 'gcc -fpreprocessed -dD -E -P {}'

def gcc_preprocess_remove_c_cpp_comment(raw_code: str,
                                        temp_src_path='_temp_file_src.cpp') -> str:
    dump_text(raw_code, temp_src_path)
    try:
        p = subprocess.Popen(_gcc_comment_remove_cmd.format(temp_src_path), stdout=subprocess.PIPE, shell=True, )
        p.wait()
        preprocessed_code = p.stdout.read().decode()
    except Exception as e:
        print(f'[Error] Err: {e}, code: {raw_code}')
        return raw_code
    return preprocessed_code