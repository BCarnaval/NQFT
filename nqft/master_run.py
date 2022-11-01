import os


electrons_number = ['half', 'near']
shapes = [[2, 2], [3, 4]]
interactions = [0.5, 1.0, 2.0, 8.0]

file = "./nqft/qcm.py"
py = "/Users/antoinedelagrave/.virtualenvs/nqft-uG72oM7R-py3.10/bin/python3"

for shape in shapes:
    # Iterating through shapes
    if shape[0] % 2 != 0 or shape[1] % 2 != 0:
        shift = 1
    else:
        shift = 0

    for electron in electrons_number:
        # Determining filling
        if electron == 'half':
            e_number = shape[0] * shape[1]
        elif electron == 'near':
            e_number = shape[0] * shape[1] - 2

        for inter in interactions:

            os.system(
                f"{py} {file} {shape[0]} {shape[1]} {e_number} {inter} {shift}"
            )

            os.system("git add -A")
            os.system("git commit -m 'Data files'")
            os.system("git push")
