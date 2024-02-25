import gadgets as gd
tokenizer = gd.tok()
for line in open("method/best_guesses.txt"):
    p = line.strip().split(",")
    print([tokenizer.encode(u, add_special_tokens=False) for u in p])
    print()