from .utils import load_csv

def get_idx_match(file_path, entities, map_pattern, id):
    l_id, r_id = map_pattern.split(',')
    id_to_idxs = [{getattr(e, id): idx for idx, e in enumerate(es)}
                  for es in entities]
    l_id_prefix = entities[0][0].id_prefix
    r_id_prefix = entities[1][0].id_prefix

    def match(row):
        idx_pair = [id_to_idxs[0].get(l_id_prefix + str(row[l_id])),
                    id_to_idxs[1].get(r_id_prefix + str(row[r_id]))]
        if None in idx_pair:
            return []
        return idx_pair

    idx_pairs = list(filter(lambda x: len(x),
                            load_csv(file_path, match)))
    idx_match = {}
    for l_idx, r_idx in idx_pairs:
        if idx_match.__contains__(l_idx):
            idx_match[l_idx].append(r_idx)
        else:
            idx_match[l_idx] = [r_idx]

    return idx_match

