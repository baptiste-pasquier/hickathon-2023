with open("src/hickathon_2023/features_md/features_after_extraction.md") as f:
    a = f.readlines()

a = [elem.replace("\n", "") for elem in a]
str_a = "FEATURES = " + str(a)

with open("src/hickathon_2023/features_md/features_to_onehotencode.md") as f:
    b = f.readlines()

b = [elem.replace("\n", "").strip() for elem in b]
str_b = "FEATURES_ONEHOT = " + str(b)

for feature in b:
    if feature not in a:
        raise ValueError(feature)

with open("src/hickathon_2023/features.py", "w") as f:
    f.write(str_a + "\n\n" + str_b)
