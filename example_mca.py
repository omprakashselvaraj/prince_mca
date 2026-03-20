import pandas as pd

from custom_mca import MCA


def main():
    X = pd.DataFrame(
        {
            "color": ["red", "blue", "red", "green", "blue", "green"],
            "shape": ["circle", "square", "triangle", "circle", "square", "triangle"],
            "size": ["S", "M", "L", "S", "M", "L"],
        }
    )

    mca = MCA(n_components=2, random_state=42, correction="benzecri")
    mca.fit(X)

    print("Row coordinates")
    print(mca.row_coordinates(X))
    print()

    print("Column coordinates")
    print(mca.column_coordinates(X))
    print()

    print("Eigenvalue summary")
    print(mca.eigenvalues_summary)


if __name__ == "__main__":
    main()
