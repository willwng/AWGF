import csv
import matplotlib.pyplot as plt
import scienceplots


def collect_data(filepath):
    iter_energy = []
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            iter_energy.append([int(row[0]), float(row[1])])
    return iter_energy


def plot_data(data, ax, label, linestyle="-"):
    ax.plot([x[0] for x in data], [x[1] for x in data], label=label, linestyle=linestyle)
    return


def main():
    plt.style.use(['science', 'nature'])
    fig, ax = plt.subplots(figsize=(7, 5))

    expt1_data = collect_data("out/expt1/energies.csv")
    expt2_data = collect_data("out/expt2/energies.csv")
    expt3_data = collect_data("out/expt3/energies.csv")
    final_energy = expt3_data[-1][1]
    # subtract
    expt1_data = [[x[0], x[1] - final_energy] for x in expt1_data]
    expt2_data = [[x[0], x[1] - final_energy] for x in expt2_data]
    expt3_data = [[x[0], x[1] - final_energy] for x in expt3_data]
    plot_data(expt1_data, ax, label="GD")
    plot_data(expt2_data, ax, label="Accelerated")
    plot_data(expt3_data, ax, label="Accelerated w/ Steering")

    ax.set_xlabel("Iteration", fontsize=20)
    ax.set_ylabel(r"Energy Residual", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(fontsize=16)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 5e3)
    ax.set_ylim(3e-4, 1e2)
    plt.tight_layout()
    plt.savefig("out/convergence-log.pdf")
    plt.show()
    return


if __name__ == "__main__":
    main()
