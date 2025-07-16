with open("char_list.txt", encoding="utf-8") as f:
    lines = [l.rstrip("\n") for l in f]
print(f"Space present: {' ' in lines}")