import nlpaug.augmenter.char as nac
import numpy as np

if __name__ == "__main__":
    with open("var.txt") as f:
        variables = [line.strip() for line in f.readlines()]
    np.random.seed(42)
    variables = np.random.choice(variables, 1024)
    aug = nac.KeyboardAug(
        aug_char_max=1,
        include_special_char=False,
        include_numeric=False,
        include_upper_case=False,
    )
    with open("typo_corr.txt", "w") as f, open("typo_var.txt", "w") as f_var:
        for variable in variables:
            aug_var = aug.augment(variable)
            variable, aug_var = variable.replace(" ", ""), aug_var.replace(" ", "")
            f.write(f"{aug_var} {variable}\n")
            f_var.write(f"{aug_var}\n")
