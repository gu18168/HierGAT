from tqdm import tqdm

def block_full(entities, idx_match, limit):
    l_entities, r_entities = entities

    pos_matches = []
    neg_matches = []
    for l_idx, l_e in tqdm(enumerate(l_entities)):
        pos_match = idx_match.get(l_idx, [])
        neg_match = []

        l_tokens = set(l_e.to_string().split(' '))
        while len(l_tokens) != -1 and len(neg_match) < limit:
            new_l_tokens = l_tokens
            r_index = -1

            for r_idx, r_e in enumerate(r_entities):
                if r_idx in neg_match or r_idx in pos_match:
                    continue

                r_tokens = set(r_e.to_string().split(' '))
                res_tokens = (l_tokens ^ r_tokens) & l_tokens
                if len(res_tokens) < len(new_l_tokens):
                    new_l_tokens = res_tokens
                    r_index = r_idx

            if len(new_l_tokens) == len(l_tokens):
                break
            else:
                l_tokens = new_l_tokens
                neg_match.append(r_index)

        l_tokens = set(l_e.to_string().split(' '))
        for _ in range(limit - len(neg_match)):
            r_index = -1
            max_num = -1
            for r_idx, r_e in enumerate(r_entities):
                if r_idx in neg_match or r_idx in pos_match:
                    continue

                r_tokens = set(r_e.to_string().split(' '))
                overlay_num = len(l_tokens & r_tokens)

                if overlay_num > max_num:
                    max_num = overlay_num
                    r_index = r_idx

            if max_num == -1:
                break

            neg_match.append(r_index)

        pos_matches.append(pos_match)
        neg_matches.append(neg_match)

    return pos_matches, neg_matches