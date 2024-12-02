# colors!
purple = '\033[95m';blue = '\033[94m';cyan = '\033[96m';lime = '\033[92m';yellow = '\033[93m';red = "\033[38;5;196m";pink = "\033[38;5;206m";orange = "\033[38;5;202m";green = "\033[38;5;34m";gray = "\033[38;5;8m";bold = '\033[1m';underline = '\033[4m';endc = '\033[0m'

# converts list/1d tensor of token ids into list of strings for those tokens
def to_str_tokens(tokenizer, input):
    toks = tokenizer(input)['input_ids']
    return [tokenizer.decode(tok) for tok in toks]