import csv
import matplotlib.pyplot as plt
import scienceplots

def main():
    plt.style.use(['science', 'nature'])
    iter_energy = []
    with open("out/energies.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            iter_energy.append([int(row[0]), float(row[1])])

    final_energy = iter_energy[-1][1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([x[0] for x in iter_energy], [x[1] for x in iter_energy])
    ax.set_xlabel("Iteration", fontsize=20)
    ax.set_ylabel("Energy", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.set_xlim(1, 3e3)
    # ax.set_ylim(1e-3, 1e2)
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    main()
