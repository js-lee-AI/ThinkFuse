def select_by_primary_ppl(client, context_ids, candidate_texts):
    if not candidate_texts:
        raise ValueError("No candidates to score.")
    ppls = [client.perplexity(context_ids, text) for text in candidate_texts]
    best = min(range(len(ppls)), key=lambda i: ppls[i])
    return best, ppls
