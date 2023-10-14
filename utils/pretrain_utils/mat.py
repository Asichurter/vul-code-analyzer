import re
import torch

def remove_consecutive_lines(code):
    if code[-1] != '\n':
        code += '\n'
    # Del line indices start from 0
    del_line_indices = []
    new_line_indices = []
    new_code = ''
    code_ptr = 0
    for m in re.finditer('\n', code):  # 3
        new_line_indices.append(m.start())
    if len(new_line_indices) > 0:
        for i in range(len(new_line_indices) - 1):
            # Consecutive lines
            # if new_line_indices[i] == new_line_indices[i + 1] + 1:
            if code[new_line_indices[i]:new_line_indices[i + 1] + 1].strip() == '':
                del_line_indices.append(i + 1)
                new_code += code[code_ptr:new_line_indices[i]+1]
                code_ptr = new_line_indices[i + 1] + 1
            else:
                new_code += code[code_ptr:new_line_indices[i]+1]
                code_ptr = new_line_indices[i]+1

    new_code += code[code_ptr:]

    # while '\n\n' in code:
    #     code = code.replace('\n\n', '\n')
    return new_code, del_line_indices

def shift_graph_matrix(mat, del_lines, shift=0):
    assert len(mat.shape) == 2 and mat.size(0) == mat.size(1)
    line_spans = [-1] + [n-shift for n in del_lines] + [len(mat)]
    # print(line_spans)

    new_rows = []
    for start, end in zip(line_spans[:-1],line_spans[1:]):
        if start < end:
            new_rows.append(mat[start+1:end])
    new_mat = torch.cat(new_rows, dim=0)

    new_cols = []
    for start, end in zip(line_spans[:-1],line_spans[1:]):
        if start < end:
            new_cols.append(new_mat[:, start+1:end])
    new_mat = torch.cat(new_cols, dim=1)
    return new_mat

def shift_edges_in_matrix(edges, del_line_indices, shift=-1):
    """
        This method is used to shift the edges in a matrix after certain lines
        were removed.
        Edges are input as compressed form, rather than matrix form.

        E.g.:
        With Del Lines: [3]   ,
        Edges: [(1,2),(1,4),(4,5)]    →     Shifted edges: [(1,2),(1,3),(3,4)]

        Demostration：
          1 2 3 4 5                            1 2   3 4
        [[0,1,0,1,0],  1                     [[0,1,×,1,0],      1
         [0,0,0,0,0],  2                      [0,0,×,0,0],      2
         [0,0,0,0,0],  3             →        [×,×,×,×,×],
         [0,0,0,0,1],  4                      [0,0,×,0,1],      3
         [0,0,0,0,0]]  5                      [0,0,×,0,0]]      4

        Note: Shift=-1 works for edges start from 1 but del_indices start from 0.
    """
    if len(edges) == 0:
        return edges
    edges_t = torch.LongTensor(edges)
    e_max = int(edges_t.max().item())
    temp_mat = torch.zeros((e_max+1, e_max+1))
    temp_mat[edges_t[:,0], edges_t[:,1]] = 1
    # Make up start from 0 and start from 1
    temp_mat = shift_graph_matrix(temp_mat, del_line_indices, shift)
    return temp_mat.nonzero().tolist()

if __name__ == '__main__':
    from utils.joern_utils.pretty_print_utils import print_code_with_line_num

    # code = 'static int xpm_decode_frame(AVCodecContext *avctx, void *data, int *got_frame, AVPacket *avpkt)\n{\n     XPMDecContext *x = avctx->priv_data;\n     AVFrame *p=data;\n    const uint8_t *end, *ptr = avpkt->data;\n     int ncolors, cpp, ret, i, j;\n     int64_t size;\n     uint32_t *dst;\n \n     avctx->pix_fmt = AV_PIX_FMT_BGRA;\n \n    end = avpkt->data + avpkt->size;\n    while (memcmp(ptr, "/* XPM */", 9) && ptr < end - 9)\n         ptr++;\n \n    if (ptr >= end) {\n         av_log(avctx, AV_LOG_ERROR, "missing signature\\n");\n         return AVERROR_INVALIDDATA;\n     }\n\n    ptr += mod_strcspn(ptr, "\\"");\n    if (sscanf(ptr, "\\"%u %u %u %u\\",",\n               &avctx->width, &avctx->height, &ncolors, &cpp) != 4) {\n        av_log(avctx, AV_LOG_ERROR, "missing image parameters\\n");\n        return AVERROR_INVALIDDATA;\n    }\n\n    if ((ret = ff_set_dimensions(avctx, avctx->width, avctx->height)) < 0)\n        return ret;\n\n    if ((ret = ff_get_buffer(avctx, p, 0)) < 0)\n        return ret;\n\n    if (cpp <= 0 || cpp >= 5) {\n        av_log(avctx, AV_LOG_ERROR, "unsupported/invalid number of chars per pixel: %d\\n", cpp);\n        return AVERROR_INVALIDDATA;\n    }\n \n     size = 1;\n     for (i = 0; i < cpp; i++)\n        size *= 94;\n \n     if (ncolors <= 0 || ncolors > size) {\n         av_log(avctx, AV_LOG_ERROR, "invalid number of colors: %d\\n", ncolors);\n        return AVERROR_INVALIDDATA;\n    }\n\n    size *= 4;\n\n    av_fast_padded_malloc(&x->pixels, &x->pixels_size, size);\n    if (!x->pixels)\n         return AVERROR(ENOMEM);\n \n     ptr += mod_strcspn(ptr, ",") + 1;\n     for (i = 0; i < ncolors; i++) {\n         const uint8_t *index;\n         int len;\n \n         ptr += mod_strcspn(ptr, "\\"") + 1;\n        if (ptr + cpp > end)\n             return AVERROR_INVALIDDATA;\n         index = ptr;\n         ptr += cpp;\n\n        ptr = strstr(ptr, "c ");\n        if (ptr) {\n            ptr += 2;\n        } else {\n            return AVERROR_INVALIDDATA;\n        }\n\n        len = strcspn(ptr, "\\" ");\n\n        if ((ret = ascii2index(index, cpp)) < 0)\n            return ret;\n \n         x->pixels[ret] = color_string_to_rgba(ptr, len);\n         ptr += mod_strcspn(ptr, ",") + 1;\n     }\n \n     for (i = 0; i < avctx->height; i++) {\n         dst = (uint32_t *)(p->data[0] + i * p->linesize[0]);\n         ptr += mod_strcspn(ptr, "\\"") + 1;\n \n         for (j = 0; j < avctx->width; j++) {\n            if (ptr + cpp > end)\n                 return AVERROR_INVALIDDATA;\n \n             if ((ret = ascii2index(ptr, cpp)) < 0)\n                return ret;\n\n            *dst++ = x->pixels[ret];\n            ptr += cpp;\n        }\n        ptr += mod_strcspn(ptr, ",") + 1;\n    }\n\n    p->key_frame = 1;\n    p->pict_type = AV_PICTURE_TYPE_I;\n\n    *got_frame = 1;\n\n    return avpkt->size;\n}\n'
    code = 'static int pmcraid_copy_sglist( struct pmcraid_sglist *sglist, unsigned long buffer, u32 len, int direction )\n{\n\tstruct scatterlist *scatterlist;\n\tvoid *kaddr;\n\tint bsize_elem;\n\tint i;\n\tint rc = 0;\n\n\t/* Determine the actual number of bytes per element */\n\tbsize_elem = PAGE_SIZE * (1 << sglist->order);\n\n\tscatterlist = sglist->scatterlist;\n\n\tfor (i = 0; i < (len / bsize_elem); i++, buffer += bsize_elem) {\n\t\tstruct page *page = sg_page(&scatterlist[i]);\n\n\t\tkaddr = kmap(page);\n\t\tif (direction == DMA_TO_DEVICE)\n\t\t\trc = __copy_from_user(kaddr,\n\t\t\t\t\t      (void *)buffer,\n\t\t\t\t\t      bsize_elem);\n\t\telse\n\t\t\trc = __copy_to_user((void *)buffer, kaddr, bsize_elem);\n\n\t\tkunmap(page);\n\n\t\tif (rc) {\n\t\t\tpmcraid_err("failed to copy user data into sg list\\n");\n\t\t\treturn -EFAULT;\n\t\t}\n\n\t\tscatterlist[i].length = bsize_elem;\n\t}\n\n\tif (len % bsize_elem) {\n\t\tstruct page *page = sg_page(&scatterlist[i]);\n\n\t\tkaddr = kmap(page);\n\n\t\tif (direction == DMA_TO_DEVICE)\n\t\t\trc = __copy_from_user(kaddr,\n\t\t\t\t\t      (void *)buffer,\n\t\t\t\t\t      len % bsize_elem);\n\t\telse\n\t\t\trc = __copy_to_user((void *)buffer,\n\t\t\t\t\t    kaddr,\n\t\t\t\t\t    len % bsize_elem);\n\n\t\tkunmap(page);\n\n\t\tscatterlist[i].length = len % bsize_elem;\n\t}\n\n\tif (rc) {\n\t\tpmcraid_err("failed to copy user data into sg list\\n");\n\t\trc = -EFAULT;\n\t}\n\n\treturn rc;\n}\n'
    rm_code, del_lines = remove_consecutive_lines(code)