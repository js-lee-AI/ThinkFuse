REPLACEMENT = "�"


def select_aligned_prefix(base_client, other_clients, new_token_ids, mode="loose"):
    if not new_token_ids:
        return "", 0

    prefixes = [new_token_ids[:i] for i in range(1, len(new_token_ids) + 1)]
    prefix_strs = base_client.batch_decode(prefixes)
    valid = [REPLACEMENT not in s for s in prefix_strs]

    for other in other_clients:
        reencoded = [other.encode(s) for s in prefix_strs]
        decoded = other.batch_decode(reencoded)
        for i, (s, d) in enumerate(zip(prefix_strs, decoded)):
            if valid[i] and (s != d or REPLACEMENT in d):
                valid[i] = False

    selected_idx = None
    if mode == "strict":
        for i, v in enumerate(valid):
            if v and prefix_strs[i].strip():
                selected_idx = i
                break
    else:
        for i in range(len(valid) - 1, -1, -1):
            if valid[i] and prefix_strs[i].strip():
                selected_idx = i
                break

    if selected_idx is None:
        for i, s in enumerate(prefix_strs):
            if REPLACEMENT not in s and s.strip():
                return s, i + 1
        return "", 0

    return prefix_strs[selected_idx], selected_idx + 1
